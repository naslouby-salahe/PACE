import copy
from pathlib import Path

import pytest

from pace.config import ResolvedConfiguration
from pace.experiment.artifacts import (
    CampaignOutcome,
    EvidenceArtifact,
    Provenance,
    read_verified_artifact,
    validate_artifact,
    write_bytes_atomically,
)
from pace.types import (
    AlphaLevel,
    ArtifactConflictError,
    ArtifactKind,
    ArtifactPayload,
    ArtifactValidationError,
    BudgetSweepPerformance,
    ClientName,
    DatasetClientId,
    DatasetName,
    ExecutionDevice,
    ProvenanceActive,
    ScientificDigest,
    StudyStratum,
)
from tests.support import (
    IMPLEMENTATION,
    INPUT,
    build_campaign_artifact,
    campaign_results,
    with_attributes,
)


@pytest.fixture(scope="module")
def campaign_artifact(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[EvidenceArtifact, ResolvedConfiguration]:
    return build_campaign_artifact(tmp_path_factory.mktemp("campaign"))


def _validate(
    artifact: EvidenceArtifact,
    configuration: ResolvedConfiguration,
    kind: ArtifactKind = ArtifactKind.DATASET_CAMPAIGN,
    dataset: DatasetName | None = DatasetName.N_BAIOT,
    stratum: StudyStratum | None = StudyStratum.N_BAIOT,
    implementation: ScientificDigest = IMPLEMENTATION,
    provenance_active: ProvenanceActive = True,
    device: ExecutionDevice = ExecutionDevice.CPU,
) -> None:
    validate_artifact(
        artifact,
        Provenance(
            configuration.scientific,
            configuration.splitting,
            configuration.scientific_digest,
            INPUT,
            implementation,
            kind,
            dataset,
            device,
            None,
            stratum,
            provenance_active,
        ),
    )


def _outcome(artifact: EvidenceArtifact) -> CampaignOutcome:
    assert isinstance(artifact.outcome, CampaignOutcome)
    return artifact.outcome


def test_valid_campaign_artifact_round_trips(
    campaign_artifact: tuple[EvidenceArtifact, ResolvedConfiguration],
) -> None:
    artifact, configuration = campaign_artifact
    _validate(artifact, configuration)


def test_campaign_validation_requires_a_dataset(
    campaign_artifact: tuple[EvidenceArtifact, ResolvedConfiguration],
) -> None:
    artifact, configuration = campaign_artifact
    with pytest.raises(ArtifactValidationError, match="requires a dataset"):
        _validate(artifact, configuration, dataset=None)


def test_kind_and_outcome_must_agree(
    campaign_artifact: tuple[EvidenceArtifact, ResolvedConfiguration],
) -> None:
    artifact, configuration = campaign_artifact
    with pytest.raises(ArtifactValidationError, match="schema or kind"):
        _validate(artifact, configuration, ArtifactKind.SYNTHETIC_DECISION)
    synthetic_kind = artifact.model_copy(update={"artifact_kind": ArtifactKind.SYNTHETIC_DECISION})
    with pytest.raises(ArtifactValidationError, match="incompatible outcome"):
        _validate(synthetic_kind, configuration, ArtifactKind.SYNTHETIC_DECISION)


def test_campaign_targets_must_match_dataset_stratum_and_seeds(
    campaign_artifact: tuple[EvidenceArtifact, ResolvedConfiguration],
) -> None:
    artifact, configuration = campaign_artifact
    outcome = _outcome(artifact)
    foreign = DatasetClientId(DatasetName.IOT_23, ClientName("7-1"))
    foreign_target = copy.copy(outcome.targets[0])
    object.__setattr__(foreign_target, "target", foreign)
    variant = artifact.model_copy(
        update={"outcome": CampaignOutcome.model_construct(targets=(foreign_target,))}
    )
    with pytest.raises(ArtifactValidationError, match="differ from the artifact dataset"):
        _validate(
            variant,
            configuration,
        )
    with pytest.raises(ArtifactValidationError, match="study stratum"):
        _validate(artifact, configuration, stratum=StudyStratum.IOT_23)
    seed_zero = tuple(target for target in outcome.targets if target.seed == 0)
    variant = artifact.model_copy(
        update={"outcome": CampaignOutcome.model_construct(targets=seed_zero)}
    )
    with pytest.raises(ArtifactValidationError, match="configured learning seeds"):
        _validate(
            variant,
            configuration,
        )
    uneven = (*outcome.targets[:-1],)
    variant = artifact.model_copy(
        update={"outcome": CampaignOutcome.model_construct(targets=uneven)}
    )
    with pytest.raises(ArtifactValidationError, match="coverage differs across learning seeds"):
        _validate(
            variant,
            configuration,
        )
    variant = artifact.model_copy(
        update={
            "outcome": CampaignOutcome.model_construct(
                targets=(outcome.targets[0], outcome.targets[0])
            )
        }
    )
    with pytest.raises(ArtifactValidationError, match="missing or repeated"):
        _validate(
            variant,
            configuration,
        )


def test_provenance_mismatches_are_conflicts(
    campaign_artifact: tuple[EvidenceArtifact, ResolvedConfiguration],
) -> None:
    artifact, configuration = campaign_artifact
    variant_input = artifact.model_copy(update={"execution_device": ExecutionDevice.CUDA})
    with pytest.raises(ArtifactConflictError):
        _validate(variant_input, configuration)
    variant_input = artifact.model_copy(update={"campaign_stratum": StudyStratum.IOT_23})
    with pytest.raises(ArtifactConflictError):
        _validate(variant_input, configuration)


def test_inactive_provenance_tolerates_stale_digests_but_keeps_integrity_checks(
    campaign_artifact: tuple[EvidenceArtifact, ResolvedConfiguration],
) -> None:
    artifact, configuration = campaign_artifact
    stale = "9" * 64
    with pytest.raises(ArtifactConflictError, match="provenance does not match"):
        _validate(artifact, configuration, implementation=stale)
    _validate(artifact, configuration, implementation=stale, provenance_active=False)
    with pytest.raises(ArtifactConflictError):
        _validate(artifact, configuration, device=ExecutionDevice.CUDA)
    _validate(artifact, configuration, device=ExecutionDevice.CUDA, provenance_active=False)

    other_stratum = artifact.model_copy(update={"campaign_stratum": StudyStratum.IOT_23})
    with pytest.raises(ArtifactConflictError, match="different study stratum"):
        _validate(other_stratum, configuration, provenance_active=False)
    tampered = artifact.model_copy(update={"input_digest": stale})
    with pytest.raises(ArtifactValidationError, match="content hash"):
        _validate(tampered, configuration, provenance_active=False)


def test_unverified_state_and_corrupt_files_are_rejected(
    campaign_artifact: tuple[EvidenceArtifact, ResolvedConfiguration], tmp_path: Path
) -> None:
    artifact, configuration = campaign_artifact
    corrupt = tmp_path / "corrupt.json"
    corrupt.write_bytes(b"{not json")
    Provenance_input = Provenance(
        configuration.scientific,
        configuration.splitting,
        configuration.scientific_digest,
        INPUT,
        IMPLEMENTATION,
        ArtifactKind.DATASET_CAMPAIGN,
        DatasetName.N_BAIOT,
    )
    with pytest.raises(ArtifactValidationError, match="Existing artifact is invalid"):
        read_verified_artifact(
            corrupt,
            Provenance_input,
        )
    assert artifact.content_digest


def test_atomic_writes_clean_up_and_reject_failed_validation(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "artifact.json"

    def reject(_path: Path) -> None:
        raise ArtifactValidationError("rejected")

    ArtifactPayload_input = ArtifactPayload(b"data")
    with pytest.raises(ArtifactValidationError, match="rejected"):
        write_bytes_atomically(target, ArtifactPayload_input, reject)
    assert not target.exists()
    assert tuple(target.parent.iterdir()) == ()
    blocked = tmp_path / "file"
    blocked.write_text("x", encoding="utf-8")
    ArtifactPayload_input = ArtifactPayload(b"data")
    with pytest.raises(ArtifactValidationError, match="Could not write artifact atomically"):
        write_bytes_atomically(blocked / "artifact.json", ArtifactPayload_input, lambda _p: None)


def test_outcome_and_artifact_validators_reject_inconsistent_content(tmp_path: Path) -> None:
    results = campaign_results()
    with pytest.raises(ValueError, match="unique target and seed"):
        CampaignOutcome(targets=())
    with pytest.raises(ValueError, match="unique target and seed"):
        CampaignOutcome(targets=(results[0], results[0]))
    seed_zero = tuple(result for result in results if result.seed == 0)
    seed_one = tuple(result for result in results if result.seed == 1)
    uneven = (*seed_zero, *seed_one[:-1])
    with pytest.raises(ValueError, match="coverage must match across learning seeds"):
        CampaignOutcome(targets=uneven)
    foreign = with_attributes(results[0], study_stratum=StudyStratum.IOT_23)
    with pytest.raises(ValueError, match="exactly one study stratum"):
        CampaignOutcome(targets=(foreign, *results[1:]))
    shifted = tuple(
        BudgetSweepPerformance(
            multiplier=point.multiplier + 0.01,
            detection_rate=point.detection_rate,
            false_alert_rate=point.false_alert_rate,
        )
        for point in results[0].budget_sweep
    )
    variant_input = with_attributes(results[0], budget_sweep=shifted)
    with pytest.raises(ValueError, match="sweep coverage must match across targets"):
        CampaignOutcome(targets=(variant_input, *results[1:]))

    artifact, _ = build_campaign_artifact(tmp_path)
    fields = {name: getattr(artifact, name) for name in EvidenceArtifact.model_fields}
    with pytest.raises(ValueError, match="stratum differs"):
        EvidenceArtifact.model_validate({**fields, "campaign_stratum": StudyStratum.IOT_23})
    variant_input = artifact.scientific_configuration.model_copy(
        update={"alpha": AlphaLevel.ALPHA_001}
    )
    with pytest.raises(ValueError, match="do not match the scientific configuration"):
        EvidenceArtifact.model_validate(
            {
                **fields,
                "scientific_configuration": variant_input,
            }
        )
