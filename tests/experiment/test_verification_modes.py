from pathlib import Path

import pytest

from pace.config import ResolvedConfiguration
from pace.experiment.artifacts import (
    CampaignOutcome,
    EvidenceArtifact,
    Provenance,
    read_verified_artifact,
    reuse_or_write_artifact,
    validate_artifact,
)
from pace.experiment.sources import (
    SourceBaseline,
    baseline_source_files,
    fingerprint_source_files,
)
from pace.types import (
    ArtifactConflictError,
    ArtifactKind,
    ArtifactReuseState,
    ArtifactValidationError,
    DatasetName,
    ReplacementPolicy,
    SourceVerification,
    StudyStratum,
)
from tests.support import IMPLEMENTATION, INPUT, build_campaign_artifact

Campaign = tuple[EvidenceArtifact, ResolvedConfiguration]


@pytest.fixture(scope="module")
def campaign(tmp_path_factory: pytest.TempPathFactory) -> Campaign:
    return build_campaign_artifact(tmp_path_factory.mktemp("verification-campaign"))


def _provenance(
    configuration: ResolvedConfiguration, *, provenance_active: bool = True
) -> Provenance:
    return Provenance(
        configuration.scientific,
        configuration.splitting,
        configuration.scientific_digest,
        INPUT,
        IMPLEMENTATION,
        ArtifactKind.DATASET_CAMPAIGN,
        DatasetName.N_BAIOT,
        campaign_stratum=StudyStratum.N_BAIOT,
        provenance_active=provenance_active,
    )


def _outcome(artifact: EvidenceArtifact) -> CampaignOutcome:
    assert isinstance(artifact.outcome, CampaignOutcome)
    return artifact.outcome


def test_strict_is_the_default(campaign: Campaign) -> None:
    artifact, configuration = campaign
    assert artifact.source_verification is SourceVerification.STRICT
    assert SourceVerification.STRICT.is_official
    assert not SourceVerification.DEVELOPMENT.is_official
    validate_artifact(artifact, _provenance(configuration))
    validate_artifact(artifact, _provenance(configuration, provenance_active=False))


def test_development_artifact_fails_strict_validation(campaign: Campaign, tmp_path: Path) -> None:
    artifact, configuration = campaign
    development = _provenance(configuration, provenance_active=False)
    strict = _provenance(configuration)
    path = tmp_path / "campaign.json"
    written, state = reuse_or_write_artifact(
        path, _outcome(artifact), development, ReplacementPolicy.PRESERVE_VALID
    )
    assert state is ArtifactReuseState.CREATED
    assert written.source_verification is SourceVerification.DEVELOPMENT
    assert written.content_digest != artifact.content_digest
    assert read_verified_artifact(path, development).content_digest == written.content_digest

    with pytest.raises(ArtifactConflictError):
        read_verified_artifact(path, strict)
    with pytest.raises(ArtifactConflictError):
        reuse_or_write_artifact(path, _outcome(artifact), strict, ReplacementPolicy.PRESERVE_VALID)


def test_stripped_marking_breaks_content_digest(campaign: Campaign, tmp_path: Path) -> None:
    artifact, configuration = campaign
    written, _ = reuse_or_write_artifact(
        tmp_path / "campaign.json",
        _outcome(artifact),
        _provenance(configuration, provenance_active=False),
        ReplacementPolicy.PRESERVE_VALID,
    )
    laundered = written.model_copy(update={"source_verification": SourceVerification.STRICT})
    with pytest.raises(ArtifactValidationError):
        validate_artifact(laundered, _provenance(configuration))


@pytest.mark.parametrize("active", [True, False])
def test_verification_follows_provenance_activity(campaign: Campaign, active: bool) -> None:
    _, configuration = campaign
    runtime = configuration.runtime.model_copy(update={"provenance_active": active})
    provenance = _provenance(configuration, provenance_active=active)
    assert provenance.verification is runtime.verification
    assert provenance.verification.is_official is active


def _manifest_provenance(base: Provenance, fingerprint: SourceBaseline) -> Provenance:
    return Provenance(
        base.scientific,
        base.splitting,
        base.configuration_digest,
        fingerprint.digest,
        IMPLEMENTATION,
        ArtifactKind.DATASET_CAMPAIGN,
        DatasetName.N_BAIOT,
        campaign_stratum=StudyStratum.N_BAIOT,
        input_fingerprint=fingerprint,
    )


def test_input_mismatch_names_expected_and_observed_files(
    campaign: Campaign, tmp_path: Path
) -> None:
    artifact, configuration = campaign
    corpus = tmp_path / "raw" / "corpus.csv"
    corpus.parent.mkdir()
    corpus.write_bytes(b"first version" * 100)
    recorded = baseline_source_files((corpus,), corpus.parent, SourceVerification.STRICT)
    base = _provenance(configuration)

    written, _ = reuse_or_write_artifact(
        tmp_path / "artifact.json",
        _outcome(artifact),
        _manifest_provenance(base, recorded),
        ReplacementPolicy.PRESERVE_VALID,
    )
    assert [record.name for record in written.source_manifest] == ["corpus.csv"]

    corpus.write_bytes(b"second version" * 100)
    observed = fingerprint_source_files((corpus,), corpus.parent, SourceVerification.STRICT)
    with pytest.raises(ArtifactConflictError) as caught:
        validate_artifact(written, _manifest_provenance(base, observed))
    message = str(caught.value)
    assert "corpus.csv" in message
    assert recorded.files[0].digest in message
    assert observed.files[0].digest in message
