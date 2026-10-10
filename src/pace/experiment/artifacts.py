import hashlib
import importlib.metadata
import os
import platform
import sys
import tempfile
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import structlog
from pydantic import ValidationError, model_validator
from structlog.stdlib import BoundLogger

from pace.config import ResolvedConfiguration, ScientificSettings, SplitSettings
from pace.datasets.common import digest_dataset_clients
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
    ByteBlockSize,
    CampaignWorkflowResult,
    CommandName,
    DatasetClientData,
    DatasetClientId,
    DatasetName,
    DatasetValidationError,
    DetectionRate,
    DigestComponentName,
    DigestNamespace,
    EnvironmentFile,
    ExecutionDevice,
    FalseAlertRate,
    FixtureKind,
    FrozenStrictModel,
    ImplementationScope,
    InputDigestMaterial,
    JointMultiplier,
    LearningSeed,
    LengthPrefixedHasher,
    LogEvent,
    MachineArchitecture,
    MissingRuntimePackage,
    ObservationCount,
    OutputDirectory,
    PlatformName,
    ProvenanceActive,
    PythonVersionText,
    ReplacementPolicy,
    RuntimePackage,
    RuntimePackageVersion,
    ScientificDigest,
    SeedDomain,
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
)

logger = structlog.get_logger()


class RuntimeEnvironment(FrozenStrictModel, frozen=True):
    python: PythonVersionText
    platform: PlatformName
    machine: MachineArchitecture
    distributions: dict[RuntimePackage, RuntimePackageVersion | MissingRuntimePackage]


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


def _content_digest(artifact: EvidenceArtifact) -> ScientificDigest:
    return payload_digest(canonical_json(artifact, ArtifactSerializationMode.OMIT_DIGEST))


def _runtime_environment() -> RuntimeEnvironment:
    versions: dict[RuntimePackage, RuntimePackageVersion | MissingRuntimePackage] = {}
    for package in RuntimePackage:
        try:
            versions[package] = RuntimePackageVersion(importlib.metadata.version(package))
        except importlib.metadata.PackageNotFoundError:
            versions[package] = MissingRuntimePackage.UNAVAILABLE
    return RuntimeEnvironment(
        python=PythonVersionText(sys.version),
        platform=PlatformName(platform.system()),
        machine=MachineArchitecture(platform.machine()),
        distributions=versions,
    )


def _implementation_components(
    scope: ImplementationScope,
) -> Iterator[tuple[DigestComponentName, ArtifactPayload]]:
    package_root = Path(__file__).resolve().parents[1]
    excluded = scope.excluded_sources
    for source_path in sorted(package_root.rglob("*.py")):
        relative_path = PurePosixPath(source_path.relative_to(package_root).as_posix())
        if not any(relative_path.is_relative_to(prefix) for prefix in excluded):
            yield (
                DigestComponentName(f"{DigestNamespace.SOURCE}/{relative_path}"),
                ArtifactPayload(source_path.read_bytes()),
            )
    repository_root = next(
        (
            parent
            for parent in package_root.parents
            if (parent / EnvironmentFile.PYPROJECT).is_file()
        ),
        None,
    )
    if repository_root is not None:
        for environment_file in (EnvironmentFile.PYPROJECT, EnvironmentFile.LOCK):
            configuration_path = repository_root / environment_file
            if configuration_path.is_file():
                yield (
                    DigestComponentName(f"{DigestNamespace.ENVIRONMENT}/{environment_file}"),
                    ArtifactPayload(configuration_path.read_bytes()),
                )
    yield (
        DigestComponentName(f"{DigestNamespace.ENVIRONMENT}/{EnvironmentFile.RUNTIME}"),
        canonical_json(_runtime_environment()),
    )


def digest_implementation(
    scope: ImplementationScope = ImplementationScope.CAMPAIGN,
) -> ScientificDigest:
    hasher = LengthPrefixedHasher()
    for name, content in _implementation_components(scope):
        hasher.framed_label(name)
        hasher.framed(content)
    return hasher.hexdigest()


def digest_files(paths: Sequence[Path], root: Path) -> ScientificDigest:
    hasher = LengthPrefixedHasher()
    for path in sorted(paths):
        hasher.framed_path(path.relative_to(root))
        with path.open("rb") as source:
            while chunk := source.read(ByteBlockSize.DIGEST):
                hasher.framed(ArtifactPayload(chunk))
    return hasher.hexdigest()


def digest_source_files(source_files: Sequence[Path], data_root: Path) -> ScientificDigest:
    resolved_root = data_root.resolve()
    resolved_files = {path.resolve() for path in source_files}
    if (
        not source_files
        or len(resolved_files) != len(source_files)
        or any(
            not path.is_file() or not path.is_relative_to(resolved_root) for path in resolved_files
        )
    ):
        raise DatasetValidationError(
            "Dataset source files must be unique and exist under the configured raw data root"
        )
    return digest_files(source_files, data_root)


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


def validate_artifact(artifact: EvidenceArtifact, provenance: Provenance) -> None:
    if (
        artifact.schema_version != CURRENT_ARTIFACT_SCHEMA
        or artifact.artifact_kind is not provenance.artifact_kind
    ):
        raise ArtifactValidationError("Artifact schema or kind is not supported")
    _validate_kind_specific(artifact, provenance)
    campaign_stratum = provenance.campaign_stratum
    if campaign_stratum is not None and artifact.campaign_stratum is not campaign_stratum:
        raise ArtifactConflictError("Existing artifact belongs to a different study stratum")
    if provenance.provenance_active and (
        artifact.scientific_configuration != provenance.scientific
        or artifact.splitting_configuration != provenance.splitting
        or artifact.configuration_digest != provenance.configuration_digest
        or artifact.input_digest != provenance.input_digest
        or artifact.implementation_digest != provenance.implementation_digest
        or artifact.execution_device is not provenance.execution_device
        or (
            provenance.preprocessing_digest is not None
            and artifact.preprocessing_digest != provenance.preprocessing_digest
        )
    ):
        raise ArtifactConflictError("Existing artifact provenance does not match this operation")
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
    artifact = _build_artifact(outcome, provenance)

    def validate_written(path_to_validate: Path) -> None:
        if read_verified_artifact(path_to_validate, provenance) != artifact:
            raise ArtifactValidationError("Serialized artifact changed during validation")

    write_bytes_atomically(
        path,
        ArtifactPayload(canonical_json(artifact) + b"\n"),
        validate_written,
    )
    return artifact, ArtifactReuseState.CREATED


def write_bytes_atomically(
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
    input_digest: ScientificDigest,
    identity: tuple[DatasetName, StudyStratum],
    clients: tuple[DatasetClientData, ...] | Callable[[], tuple[DatasetClientData, ...]],
) -> Provenance:
    dataset, stratum = identity
    return Provenance(
        configuration.scientific,
        configuration.splitting,
        configuration.scientific_digest,
        input_digest,
        digest_implementation(),
        ArtifactKind.DATASET_CAMPAIGN,
        dataset,
        configuration.runtime.execution_device,
        None if callable(clients) else digest_dataset_clients(clients),
        stratum,
        configuration.runtime.provenance_active,
    )


def require_unchanged_sources(
    source_files: tuple[Path, ...],
    configuration: ResolvedConfiguration,
    input_digest: ScientificDigest,
) -> None:
    if digest_files(source_files, configuration.runtime.raw_data_root) != input_digest:
        raise DatasetValidationError("Dataset source files changed during campaign execution")


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
