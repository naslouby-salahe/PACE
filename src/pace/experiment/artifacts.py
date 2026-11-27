import fcntl
import hashlib
import os
import tempfile
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import structlog
from pydantic import ValidationError, model_validator
from structlog.stdlib import BoundLogger

from pace.config import ResolvedConfiguration, ScientificSettings, SplitSettings
from pace.datasets.common import digest_dataset_clients
from pace.experiment.implementation import (
    digest_implementation,
    require_implementation_unchanged,
    require_numerical_policy,
)
from pace.experiment.source_integrity import describe_input_difference
from pace.experiment.sources import SourceBaseline
from pace.types import (
    CURRENT_ARTIFACT_SCHEMA,
    AlertDecision,
    ArtifactConflictError,
    ArtifactFileName,
    ArtifactKind,
    ArtifactPath,
    ArtifactPayload,
    ArtifactReuseState,
    ArtifactSchemaVersion,
    ArtifactSerializationMode,
    ArtifactValidationError,
    ArtifactVerification,
    CampaignWorkflowResult,
    CommandName,
    DatasetClientData,
    DatasetClientId,
    DatasetName,
    DatasetValidationError,
    DetectionRate,
    DisplayText,
    ExecutionDevice,
    FalseAlertRate,
    FixtureKind,
    FrozenStrictModel,
    InputDigestMaterial,
    JointMultiplier,
    LearningSeed,
    LengthPrefixedHasher,
    LockFileSuffix,
    LogEvent,
    ObservationCount,
    OutputDirectory,
    ProvenanceActive,
    ProvenanceDifference,
    ProvenanceField,
    ReplacementPolicy,
    ScientificDigest,
    SeedDomain,
    SourceFileRecord,
    SourceVerification,
    StudyStratum,
    TargetPerformance,
    TextEncoding,
    Threshold,
    VerificationState,
    big_endian_bytes,
    big_endian_value,
    canonical_json,
    default_study_stratum,
    payload_digest,
    placeholder_digest,
    short_digest,
)

logger = structlog.get_logger()


class SyntheticOutcome(FrozenStrictModel, frozen=True):
    fixture_kind: FixtureKind
    row_count: ObservationCount
    detected_count: ObservationCount
    local_alert_count: ObservationCount
    peer_alert_count: ObservationCount
    benign_alert_rate: FalseAlertRate
    attack_detection_rate: DetectionRate
    local_threshold: Threshold
    expert_thresholds: tuple[Threshold, ...]
    joint_multiplier: JointMultiplier
    peer_block_alert_rates: tuple[FalseAlertRate, ...]
    alerts: tuple[AlertDecision, ...]


class CampaignOutcome(FrozenStrictModel, frozen=True):
    targets: tuple[TargetPerformance, ...]

    @model_validator(mode="after")
    def validate_target_coverage(self) -> "CampaignOutcome":
        identifiers = tuple((target.target, target.seed) for target in self.targets)
        if not identifiers or len(set(identifiers)) != len(identifiers):
            raise ValueError("Campaign artifacts require unique target and seed results")
        target_sets_by_seed = {
            frozenset(target.target for target in self.targets if target.seed == seed)
            for seed in dict.fromkeys(target.seed for target in self.targets)
        }
        if len(target_sets_by_seed) != 1:
            raise ValueError("Campaign target coverage must match across learning seeds")
        if len({target.study_stratum for target in self.targets}) != 1:
            raise ValueError("A campaign outcome must contain exactly one study stratum")
        schedules = {
            tuple(point.multiplier for point in target.budget_sweep) for target in self.targets
        }
        if len(schedules) != 1:
            raise ValueError("Campaign budget sweep coverage must match across targets")
        return self

    @property
    def target_count(self) -> ObservationCount:
        return len({target.target for target in self.targets})

    @classmethod
    def from_targets(cls, targets: tuple[TargetPerformance, ...]) -> "CampaignOutcome":
        return cls(targets=targets)


class EvidenceArtifact(FrozenStrictModel, frozen=True):
    schema_version: ArtifactSchemaVersion
    artifact_kind: ArtifactKind
    scientific_configuration: ScientificSettings
    splitting_configuration: SplitSettings
    configuration_digest: ScientificDigest
    input_digest: ScientificDigest
    implementation_digest: ScientificDigest
    execution_device: ExecutionDevice
    source_verification: SourceVerification
    source_manifest: tuple[SourceFileRecord, ...]
    preprocessing_digest: ScientificDigest | None
    campaign_stratum: StudyStratum | None
    verification_state: VerificationState
    outcome: SyntheticOutcome | CampaignOutcome
    content_digest: ScientificDigest

    @model_validator(mode="after")
    def campaign_sweeps_match_configuration(self) -> "EvidenceArtifact":
        if isinstance(self.outcome, CampaignOutcome):
            strata = {target.study_stratum for target in self.outcome.targets}
            if len(strata) != 1 or self.campaign_stratum not in strata:
                raise ValueError("Campaign artifact stratum differs from its target results")
            expected = self.scientific_configuration.budget_sweep_multipliers.for_alpha(
                self.scientific_configuration.alpha
            )
            if any(
                tuple(point.multiplier for point in target.budget_sweep) != expected
                for target in self.outcome.targets
            ):
                raise ValueError("Campaign budget sweeps do not match the scientific configuration")
        elif self.campaign_stratum is not None:
            raise ValueError("Synthetic artifacts cannot carry a campaign study stratum")
        return self


@dataclass(frozen=True, slots=True)
class Provenance:
    scientific: ScientificSettings
    splitting: SplitSettings
    configuration_digest: ScientificDigest
    input_digest: ScientificDigest
    implementation_digest: ScientificDigest
    artifact_kind: ArtifactKind = ArtifactKind.SYNTHETIC_DECISION
    campaign_dataset: DatasetName | None = None
    execution_device: ExecutionDevice = ExecutionDevice.CPU
    preprocessing_digest: ScientificDigest | None = None
    campaign_stratum: StudyStratum | None = None
    provenance_active: ProvenanceActive = True
    input_fingerprint: SourceBaseline | None = None

    @property
    def verification(self) -> SourceVerification:
        return SourceVerification.for_provenance(self.provenance_active)


def _content_digest(artifact: EvidenceArtifact) -> ScientificDigest:
    return payload_digest(canonical_json(artifact, ArtifactSerializationMode.OMIT_DIGEST))


def _validate_kind_specific(artifact: EvidenceArtifact, provenance: Provenance) -> None:
    if provenance.artifact_kind is ArtifactKind.SYNTHETIC_DECISION:
        if not isinstance(artifact.outcome, SyntheticOutcome):
            raise ArtifactValidationError("Synthetic artifact has an incompatible outcome schema")
        if artifact.preprocessing_digest is not None:
            raise ArtifactValidationError("Synthetic artifacts cannot carry dataset preprocessing")
        if artifact.campaign_stratum is not None:
            raise ArtifactValidationError("Synthetic artifacts cannot carry a campaign stratum")
    elif provenance.artifact_kind is ArtifactKind.DATASET_CAMPAIGN:
        if provenance.campaign_dataset is None:
            raise ArtifactValidationError("Dataset campaign validation requires a dataset")
        _validate_campaign_outcome(artifact.outcome, provenance)


def _digest_difference(
    field: ProvenanceField, recorded: ScientificDigest | None, expected: ScientificDigest | None
) -> ProvenanceDifference | None:
    if recorded == expected:
        return None
    return ProvenanceDifference(
        f"{field} (artifact {short_digest(recorded)}, operation {short_digest(expected)})"
    )


def provenance_differences(
    artifact: EvidenceArtifact, provenance: Provenance
) -> tuple[ProvenanceDifference, ...]:
    differences = (
        _digest_difference(
            ProvenanceField.CONFIGURATION_DIGEST,
            artifact.configuration_digest,
            provenance.configuration_digest,
        ),
        _digest_difference(
            ProvenanceField.INPUT_DIGEST, artifact.input_digest, provenance.input_digest
        ),
        _digest_difference(
            ProvenanceField.IMPLEMENTATION_DIGEST,
            artifact.implementation_digest,
            provenance.implementation_digest,
        ),
        None
        if provenance.preprocessing_digest is None
        else _digest_difference(
            ProvenanceField.PREPROCESSING_DIGEST,
            artifact.preprocessing_digest,
            provenance.preprocessing_digest,
        ),
        None
        if artifact.scientific_configuration == provenance.scientific
        else ProvenanceDifference(ProvenanceField.SCIENTIFIC_CONFIGURATION),
        None
        if artifact.splitting_configuration == provenance.splitting
        else ProvenanceDifference(ProvenanceField.SPLITTING_CONFIGURATION),
        None
        if artifact.execution_device is provenance.execution_device
        else ProvenanceDifference(
            f"{ProvenanceField.EXECUTION_DEVICE} (artifact {artifact.execution_device.name}, "
            f"operation {provenance.execution_device.name})"
        ),
    )
    return tuple(difference for difference in differences if difference is not None)


def _input_diagnostics(artifact: EvidenceArtifact, provenance: Provenance) -> DisplayText:
    fingerprint = provenance.input_fingerprint
    if fingerprint is None or artifact.input_digest == provenance.input_digest:
        return DisplayText("")
    return DisplayText(f"\n{describe_input_difference(artifact.source_manifest, fingerprint)}")


def validate_artifact(artifact: EvidenceArtifact, provenance: Provenance) -> None:
    if (
        artifact.schema_version != CURRENT_ARTIFACT_SCHEMA
        or artifact.artifact_kind is not provenance.artifact_kind
    ):
        raise ArtifactValidationError("Artifact schema or kind is not supported")
    provenance.verification.require_usable(
        artifact.source_verification, DisplayText("Existing artifact")
    )
    _validate_kind_specific(artifact, provenance)
    campaign_stratum = provenance.campaign_stratum
    if campaign_stratum is not None and artifact.campaign_stratum is not campaign_stratum:
        raise ArtifactConflictError("Existing artifact belongs to a different study stratum")
    differences = (
        provenance_differences(artifact, provenance) if provenance.provenance_active else ()
    )
    if differences:
        raise ArtifactConflictError(
            f"Existing artifact provenance does not match this operation: "
            f"{'; '.join(differences)}{_input_diagnostics(artifact, provenance)}"
        )
    if artifact.verification_state is not VerificationState.VERIFIED:
        raise ArtifactValidationError("Artifact has no successful verification state")
    if artifact.content_digest != _content_digest(artifact):
        raise ArtifactValidationError("Artifact content hash does not match its contents")


def _validate_campaign_outcome(
    outcome: SyntheticOutcome | CampaignOutcome, provenance: Provenance
) -> None:
    if not isinstance(outcome, CampaignOutcome):
        raise ArtifactValidationError("Campaign artifact has an incompatible outcome schema")
    identifiers = tuple((target.target, target.seed) for target in outcome.targets)
    if not identifiers or len(set(identifiers)) != len(identifiers):
        raise ArtifactValidationError("Campaign artifact has missing or repeated target results")
    if any(target.target.dataset is not provenance.campaign_dataset for target in outcome.targets):
        raise ArtifactValidationError("Campaign targets differ from the artifact dataset")
    strata = {target.study_stratum for target in outcome.targets}
    campaign_stratum = provenance.campaign_stratum
    if len(strata) != 1 or (campaign_stratum is not None and strata != {campaign_stratum}):
        raise ArtifactValidationError("Campaign targets differ from the artifact study stratum")
    configured_seeds = provenance.scientific.learning_seeds
    if {target.seed for target in outcome.targets} != set(configured_seeds):
        raise ArtifactValidationError("Campaign artifact does not match configured learning seeds")
    targets_by_seed = {
        tuple(sorted(target.target.name for target in outcome.targets if target.seed == seed))
        for seed in configured_seeds
    }
    if len(targets_by_seed) != 1:
        raise ArtifactValidationError("Campaign target coverage differs across learning seeds")


def read_verified_artifact(path: Path, provenance: Provenance) -> EvidenceArtifact:
    try:
        artifact = EvidenceArtifact.model_validate_json(path.read_bytes(), strict=True)
    except (OSError, ValidationError, ValueError) as error:
        raise ArtifactValidationError(f"Existing artifact is invalid: {path}") from error
    validate_artifact(artifact, provenance)
    logger.debug(
        LogEvent.ARTIFACT_VERIFIED,
        artifact_path=path.as_posix(),
        artifact_kind=artifact.artifact_kind.name,
        content_digest=artifact.content_digest,
    )
    return artifact


def _build_artifact(
    outcome: SyntheticOutcome | CampaignOutcome, provenance: Provenance
) -> EvidenceArtifact:
    selected_stratum = provenance.campaign_stratum
    if provenance.artifact_kind is ArtifactKind.DATASET_CAMPAIGN and selected_stratum is None:
        if not isinstance(outcome, CampaignOutcome):
            raise ArtifactValidationError("Dataset campaign requires a campaign outcome")
        strata = {target.study_stratum for target in outcome.targets}
        if len(strata) != 1:
            raise ArtifactValidationError("Dataset campaign requires exactly one study stratum")
        selected_stratum = next(iter(strata))

    def artifact(content_digest: ScientificDigest) -> EvidenceArtifact:
        return EvidenceArtifact(
            schema_version=CURRENT_ARTIFACT_SCHEMA,
            artifact_kind=provenance.artifact_kind,
            scientific_configuration=provenance.scientific,
            splitting_configuration=provenance.splitting,
            configuration_digest=provenance.configuration_digest,
            input_digest=provenance.input_digest,
            implementation_digest=provenance.implementation_digest,
            execution_device=provenance.execution_device,
            source_verification=provenance.verification,
            source_manifest=(
                () if provenance.input_fingerprint is None else provenance.input_fingerprint.records
            ),
            preprocessing_digest=provenance.preprocessing_digest,
            campaign_stratum=selected_stratum,
            verification_state=VerificationState.VERIFIED,
            outcome=outcome,
            content_digest=content_digest,
        )

    return artifact(_content_digest(artifact(placeholder_digest())))


def reuse_or_write_artifact(
    path: Path,
    outcome: SyntheticOutcome | CampaignOutcome,
    provenance: Provenance,
    replacement_policy: ReplacementPolicy,
) -> tuple[EvidenceArtifact, ArtifactReuseState]:
    if path.exists() and replacement_policy is ReplacementPolicy.PRESERVE_VALID:
        return read_verified_artifact(path, provenance), ArtifactReuseState.REUSED
    if provenance.provenance_active and provenance.artifact_kind is ArtifactKind.DATASET_CAMPAIGN:
        require_implementation_unchanged()
    artifact = _build_artifact(outcome, provenance)

    def validate_written(path_to_validate: Path) -> None:
        if read_verified_artifact(path_to_validate, provenance) != artifact:
            raise ArtifactValidationError("Serialized artifact changed during validation")

    with _exclusive_destination(path):
        if path.exists() and replacement_policy is ReplacementPolicy.PRESERVE_VALID:
            concurrent = read_verified_artifact(path, provenance)
            if concurrent.content_digest != artifact.content_digest:
                raise ArtifactConflictError(
                    "A concurrent writer persisted different content for identical provenance "
                    f"(existing {short_digest(concurrent.content_digest)}, computed "
                    f"{short_digest(artifact.content_digest)}); the computation is not "
                    "deterministic"
                )
            return concurrent, ArtifactReuseState.REUSED
        _write_atomically(path, ArtifactPayload(canonical_json(artifact) + b"\n"), validate_written)
    return artifact, ArtifactReuseState.CREATED


@contextmanager
def _exclusive_destination(path: Path) -> Generator[None]:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        handle = (path.parent / f".{path.name}{LockFileSuffix.LOCK}").open("ab")
    except OSError as error:
        raise ArtifactValidationError(f"Could not write artifact atomically: {path}") from error
    with handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            for stale in path.parent.glob(f".{path.name}.*"):
                if stale.suffix != LockFileSuffix.LOCK:
                    stale.unlink(missing_ok=True)
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def write_bytes_atomically(
    path: Path, serialized: ArtifactPayload, validate: Callable[[Path], None]
) -> None:
    with _exclusive_destination(path):
        _write_atomically(path, serialized, validate)


def _write_atomically(
    path: Path, serialized: ArtifactPayload, validate: Callable[[Path], None]
) -> None:
    temporary_path: Path | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            temporary_file.write(serialized)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        validate(temporary_path)
        temporary_path.replace(path)
        temporary_path = None
        directory_handle = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_handle)
        finally:
            os.close(directory_handle)
        validate(path)
        logger.debug(
            LogEvent.ARTIFACT_PERSISTED, artifact_path=path.as_posix(), byte_count=len(serialized)
        )
    except OSError as error:
        raise ArtifactValidationError(f"Could not write artifact atomically: {path}") from error
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def artifact_path(output_root: Path, configuration_digest: ScientificDigest) -> ArtifactPath:
    return ArtifactPath(
        output_root / OutputDirectory.SMOKE / configuration_digest / ArtifactFileName.DECISION
    )


def campaign_artifact_path(
    output_root: Path,
    campaign: DatasetName | StudyStratum,
    configuration_digest: ScientificDigest,
) -> ArtifactPath:
    stratum = campaign if isinstance(campaign, StudyStratum) else default_study_stratum(campaign)
    return ArtifactPath(
        output_root
        / OutputDirectory.CAMPAIGNS
        / stratum.name.lower()
        / configuration_digest
        / ArtifactFileName.CAMPAIGN_RESULTS
    )


def execution_log_path(output_root: Path, command: CommandName) -> Path:
    return output_root / OutputDirectory.LOGS / command.log_file_name


def digest_inputs(material: InputDigestMaterial) -> ScientificDigest:
    hasher = LengthPrefixedHasher()
    for chunk in material.chunks:
        hasher.framed(chunk)
    return hasher.hexdigest()


def derive_learning_seed(
    seed: LearningSeed,
    client: DatasetClientId,
    domain: SeedDomain,
) -> LearningSeed:
    material = b"\0".join(
        (
            big_endian_bytes(seed, max(1, (seed.bit_length() + 7) // 8)),
            client.dataset.name.encode(TextEncoding.ASCII),
            client.name.encode(TextEncoding.UTF8),
            domain.name.encode(TextEncoding.ASCII),
        )
    )
    return big_endian_value(ArtifactPayload(hashlib.sha256(material).digest()[:4]))


def reused_campaign(
    destination: ArtifactPath, provenance: Provenance, logger: BoundLogger
) -> CampaignWorkflowResult:
    artifact = read_verified_artifact(destination.path, provenance)
    if not isinstance(artifact.outcome, CampaignOutcome):
        raise DatasetValidationError("Verified campaign artifact has the wrong result schema")
    logger.info(LogEvent.CAMPAIGN_REUSED, target_count=artifact.outcome.target_count)
    return workflow_result(
        destination, artifact.content_digest, ArtifactReuseState.REUSED, artifact.outcome
    )


def campaign_provenance(
    configuration: ResolvedConfiguration,
    baseline: SourceBaseline,
    identity: tuple[DatasetName, StudyStratum],
    clients: tuple[DatasetClientData, ...] | Callable[[], tuple[DatasetClientData, ...]],
) -> Provenance:
    require_numerical_policy(configuration.scientific.numerical_threads)
    dataset, stratum = identity
    return Provenance(
        configuration.scientific,
        configuration.splitting,
        configuration.scientific_digest,
        baseline.digest,
        digest_implementation(),
        ArtifactKind.DATASET_CAMPAIGN,
        dataset,
        configuration.runtime.execution_device,
        None if callable(clients) else digest_dataset_clients(clients),
        stratum,
        configuration.runtime.provenance_active,
        baseline,
    )


def workflow_result(
    destination: ArtifactPath,
    content_digest: ScientificDigest,
    reuse_state: ArtifactReuseState,
    outcome: CampaignOutcome,
) -> CampaignWorkflowResult:
    return CampaignWorkflowResult(
        destination,
        ArtifactVerification(VerificationState.VERIFIED, content_digest),
        reuse_state,
        outcome.target_count,
    )
