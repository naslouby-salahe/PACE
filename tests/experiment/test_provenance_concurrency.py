import os
import threading
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

import pace.experiment.artifacts as artifacts
import pace.experiment.implementation as implementation
from pace.config import ResolvedConfiguration
from pace.experiment.artifacts import (
    CampaignOutcome,
    EvidenceArtifact,
    Provenance,
    read_verified_artifact,
    reuse_or_write_artifact,
    validate_artifact,
    write_bytes_atomically,
)
from pace.experiment.implementation import digest_implementation, require_implementation_unchanged
from pace.experiment.sources import digest_source_files
from pace.types import (
    ArtifactConflictError,
    ArtifactKind,
    ArtifactPayload,
    ArtifactReuseState,
    DatasetName,
    DigestComponentName,
    ExecutionDevice,
    ImplementationScope,
    ReplacementPolicy,
    StudyStratum,
)
from tests.support import IMPLEMENTATION, INPUT, build_campaign_artifact, with_attributes


@pytest.fixture(scope="module")
def campaign(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[EvidenceArtifact, ResolvedConfiguration]:
    return build_campaign_artifact(tmp_path_factory.mktemp("concurrency-campaign"))


def _provenance(configuration: ResolvedConfiguration) -> Provenance:
    return Provenance(
        configuration.scientific,
        configuration.splitting,
        configuration.scientific_digest,
        INPUT,
        IMPLEMENTATION,
        ArtifactKind.DATASET_CAMPAIGN,
        DatasetName.N_BAIOT,
        ExecutionDevice.CPU,
        None,
        StudyStratum.N_BAIOT,
    )


def _outcome(artifact: EvidenceArtifact) -> CampaignOutcome:
    assert isinstance(artifact.outcome, CampaignOutcome)
    return artifact.outcome


def _sources(root: Path) -> tuple[Path, ...]:
    first = root / "corpus" / "a.csv"
    second = root / "corpus" / "b.csv"
    first.parent.mkdir(parents=True)
    first.write_bytes(b"alpha" * 1000)
    second.write_bytes(b"beta" * 1000)
    return first, second


def test_source_digest_depends_only_on_content_and_relative_names(tmp_path: Path) -> None:
    first, second = _sources(tmp_path)
    reference = digest_source_files((first, second), tmp_path)

    os.utime(first, (1, 1))
    os.utime(second, (2_000_000_000, 2_000_000_000))
    assert digest_source_files((second, first), tmp_path) == reference

    second.write_bytes(b"beta" * 999 + b"betb")
    assert digest_source_files((first, second), tmp_path) != reference
    second.write_bytes(b"beta" * 1000)
    assert digest_source_files((first, second), tmp_path) == reference

    renamed = second.rename(second.with_name("c.csv"))
    assert digest_source_files((first, renamed), tmp_path) != reference


def test_concurrent_readers_obtain_one_stable_source_digest(tmp_path: Path) -> None:
    sources = _sources(tmp_path)
    expected = digest_source_files(sources, tmp_path)

    def digest(_round: int) -> str:
        return digest_source_files(sources, tmp_path)

    with ThreadPoolExecutor(max_workers=8) as pool:
        digests = list(pool.map(digest, range(32)))
    assert set(digests) == {expected}


def test_implementation_digest_is_the_snapshot_of_the_loaded_code_and_edits_are_named(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    components = {
        DigestComponentName("source/core.py"): b"version one",
        DigestComponentName("source/decision.py"): b"stable",
    }

    def current_components(
        _scope: ImplementationScope,
    ) -> Iterator[tuple[DigestComponentName, ArtifactPayload]]:
        return iter((name, ArtifactPayload(content)) for name, content in components.items())

    monkeypatch.setattr(implementation, "_implementation_components", current_components)
    implementation._implementation_snapshot.cache_clear()  # pyright: ignore[reportPrivateUsage]
    try:
        snapshot = digest_implementation(ImplementationScope.SYNTHETIC)
        require_implementation_unchanged(ImplementationScope.SYNTHETIC)

        components[DigestComponentName("source/core.py")] = b"version two"
        components[DigestComponentName("source/new.py")] = b"added"
        with pytest.raises(ArtifactConflictError) as caught:
            require_implementation_unchanged(ImplementationScope.SYNTHETIC)
        assert "source/core.py" in str(caught.value)
        assert "source/new.py" in str(caught.value)
        assert "source/decision.py" not in str(caught.value)
        assert digest_implementation(ImplementationScope.SYNTHETIC) == snapshot
    finally:
        implementation._implementation_snapshot.cache_clear()  # pyright: ignore[reportPrivateUsage]


def test_provenance_conflicts_name_the_differing_fields(
    campaign: tuple[EvidenceArtifact, ResolvedConfiguration],
) -> None:
    artifact, configuration = campaign
    base = _provenance(configuration)
    changed = Provenance(
        base.scientific,
        base.splitting,
        base.configuration_digest,
        "5" * 64,
        "6" * 64,
        base.artifact_kind,
        base.campaign_dataset,
        ExecutionDevice.CUDA,
        None,
        base.campaign_stratum,
    )
    with pytest.raises(ArtifactConflictError) as caught:
        validate_artifact(artifact, changed)
    message = str(caught.value)
    assert "provenance does not match" in message
    assert f"input_digest (artifact {INPUT[:12]}, operation {'5' * 12})" in message
    assert (
        f"implementation_digest (artifact {IMPLEMENTATION[:12]}, operation {'6' * 12})" in message
    )
    assert "execution_device (artifact CPU, operation CUDA)" in message
    assert "configuration_digest" not in message


def test_simultaneous_identical_writers_create_once_and_all_read_the_same_artifact(
    tmp_path: Path, campaign: tuple[EvidenceArtifact, ResolvedConfiguration]
) -> None:
    artifact, configuration = campaign
    provenance = _provenance(configuration)
    destination = tmp_path / "campaigns" / "results.json"
    gate = threading.Barrier(6)

    def write(_worker: int) -> ArtifactReuseState:
        gate.wait()
        _, state = reuse_or_write_artifact(
            destination, _outcome(artifact), provenance, ReplacementPolicy.PRESERVE_VALID
        )
        return state

    with ThreadPoolExecutor(max_workers=6) as pool:
        states = list(pool.map(write, range(6)))

    assert states.count(ArtifactReuseState.CREATED) == 1
    assert states.count(ArtifactReuseState.REUSED) == 5
    assert read_verified_artifact(destination, provenance).content_digest == artifact.content_digest
    assert [
        path.name for path in destination.parent.iterdir() if not path.name.endswith(".lock")
    ] == ["results.json"]


def test_concurrent_writer_with_different_content_is_a_loud_conflict_not_a_silent_overwrite(
    tmp_path: Path,
    campaign: tuple[EvidenceArtifact, ResolvedConfiguration],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact, configuration = campaign
    provenance = _provenance(configuration)
    destination = tmp_path / "campaigns" / "results.json"
    targets = _outcome(artifact).targets
    rival_outcome = CampaignOutcome.from_targets(
        (with_attributes(targets[0], realized_false_alert_rate=0.123), *targets[1:])
    )
    rival = artifacts._build_artifact(rival_outcome, provenance)  # pyright: ignore[reportPrivateUsage]
    original_build = artifacts._build_artifact  # pyright: ignore[reportPrivateUsage]

    def build_then_lose_the_race(
        outcome: CampaignOutcome, provenance_of_run: Provenance
    ) -> EvidenceArtifact:
        built = original_build(outcome, provenance_of_run)
        write_bytes_atomically(
            destination,
            ArtifactPayload(artifacts.canonical_json(rival) + b"\n"),
            lambda _path: None,
        )
        return built

    monkeypatch.setattr(artifacts, "_build_artifact", build_then_lose_the_race)
    with pytest.raises(ArtifactConflictError, match="not deterministic"):
        reuse_or_write_artifact(
            destination, _outcome(artifact), provenance, ReplacementPolicy.PRESERVE_VALID
        )
    assert read_verified_artifact(destination, provenance).content_digest == rival.content_digest


def test_interrupted_writers_leave_only_ignorable_temporary_files(
    tmp_path: Path, campaign: tuple[EvidenceArtifact, ResolvedConfiguration]
) -> None:
    artifact, configuration = campaign
    provenance = _provenance(configuration)
    destination = tmp_path / "campaigns" / "results.json"
    destination.parent.mkdir(parents=True)
    stale = destination.parent / ".results.json.crashed"
    stale.write_bytes(b"half written")

    _, state = reuse_or_write_artifact(
        destination, _outcome(artifact), provenance, ReplacementPolicy.PRESERVE_VALID
    )

    assert state is ArtifactReuseState.CREATED
    assert not stale.exists()
    assert read_verified_artifact(destination, provenance).content_digest == artifact.content_digest


def test_the_artifact_is_complete_on_both_sides_of_the_atomic_replacement(
    tmp_path: Path,
    campaign: tuple[EvidenceArtifact, ResolvedConfiguration],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact, configuration = campaign
    provenance = _provenance(configuration)
    destination = tmp_path / "campaigns" / "results.json"
    reuse_or_write_artifact(
        destination, _outcome(artifact), provenance, ReplacementPolicy.PRESERVE_VALID
    )
    observed: list[str] = []
    original = Path.replace

    def observing_replace(source: Path, target: Path) -> Path:
        observed.append(read_verified_artifact(destination, provenance).content_digest)
        moved = original(source, target)
        observed.append(read_verified_artifact(destination, provenance).content_digest)
        return moved

    monkeypatch.setattr(Path, "replace", observing_replace)
    reuse_or_write_artifact(
        destination, _outcome(artifact), provenance, ReplacementPolicy.REPLACE_REQUESTED
    )
    assert observed == [artifact.content_digest] * 2
