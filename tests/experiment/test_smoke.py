import hashlib
import json
import logging
from pathlib import Path

import pytest

from pace.config import ResolvedConfiguration, load_configuration
from pace.experiment.artifacts import Provenance, digest_implementation, validate_artifact
from pace.experiment.smoke import run_synthetic_workflow
from pace.types import (
    ArtifactConflictError,
    ArtifactReuseState,
    ArtifactValidationError,
    ImplementationScope,
    ReplacementPolicy,
    VerificationState,
)
from tests.support import REPOSITORY_ROOT


def _configuration(repository_root: Path, output_root: Path) -> ResolvedConfiguration:
    source = REPOSITORY_ROOT / "configs" / "default.yaml"
    configuration = load_configuration(source, repository_root)
    return configuration.model_copy(
        update={"runtime": configuration.runtime.model_copy(update={"output_root": output_root})}
    )


def test_implementation_digest_includes_locked_dependency_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository_root = tmp_path / "repository"
    package_root = repository_root / "src" / "pace"
    package_directory = package_root / "experiment"
    package_directory.mkdir(parents=True)
    module_path = package_directory / "artifacts.py"
    module_path.write_text("", encoding="utf-8")
    (package_root / "core.py").write_text("value = 1\n", encoding="utf-8")
    (repository_root / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    lock_file = repository_root / "uv.lock"
    lock_file.write_text("version = 1\n", encoding="utf-8")
    monkeypatch.setattr("pace.experiment.artifacts.__file__", str(module_path))

    first_digest = digest_implementation()
    repeated_digest = digest_implementation()
    lock_file.write_text("version = 2\n", encoding="utf-8")
    updated_digest = digest_implementation()

    assert repeated_digest == first_digest
    assert updated_digest != first_digest


def test_smoke_artifact_is_verified_and_reused_without_rewrite(tmp_path: Path) -> None:
    configuration = _configuration(tmp_path, tmp_path / "outputs")
    first = run_synthetic_workflow(configuration)
    artifact_path = first.artifact_path.path
    original_content = artifact_path.read_bytes()
    persisted_artifact = json.loads(original_content)
    persisted_outcome = persisted_artifact["outcome"]
    assert persisted_outcome["fixture_kind"] == "DETERMINISTIC_SYNTHETIC"
    assert persisted_artifact["execution_device"] == "CPU"
    assert persisted_outcome["detected_count"] > 0
    assert persisted_outcome["local_alert_count"] > 0
    assert persisted_outcome["peer_alert_count"] > 0
    original_hash = hashlib.sha256(original_content).hexdigest()
    original_mtime = artifact_path.stat().st_mtime_ns
    second = run_synthetic_workflow(configuration)
    assert first.reuse_state is ArtifactReuseState.CREATED
    assert second.reuse_state is ArtifactReuseState.REUSED
    assert second.verification.state is VerificationState.VERIFIED
    assert hashlib.sha256(artifact_path.read_bytes()).hexdigest() == original_hash
    assert artifact_path.stat().st_mtime_ns == original_mtime
    assert len(tuple(artifact_path.parent.iterdir())) == 1


def test_corrupt_artifact_is_rejected_and_replaced_only_on_overwrite(tmp_path: Path) -> None:
    configuration = _configuration(tmp_path, tmp_path / "outputs")
    result = run_synthetic_workflow(configuration)
    artifact_path = result.artifact_path.path
    artifact_path.write_text("{}", encoding="utf-8")
    with pytest.raises(ArtifactValidationError):
        run_synthetic_workflow(configuration)
    log_events = tuple(
        json.loads(line)
        for line in (configuration.runtime.output_root / "logs" / "smoke.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    failure_events = tuple(event for event in log_events if event["event"] == "run.failed")
    assert len(failure_events) == 1
    assert failure_events[0]["failure_type"] == "ArtifactValidationError"
    assert failure_events[0]["configuration_digest"] == configuration.scientific_digest
    assert failure_events[0]["artifact_commit_state"] == "unknown"
    assert failure_events[0]["safe_to_resume"] is True
    overwritten = run_synthetic_workflow(configuration, ReplacementPolicy.REPLACE_REQUESTED)
    assert overwritten.reuse_state is ArtifactReuseState.CREATED
    artifact_text = artifact_path.read_text(encoding="utf-8")
    assert artifact_text.count('"verification_state":"VERIFIED"') == 1
    assert len(tuple(artifact_path.parent.iterdir())) == 1


def test_artifact_validation_checks_schema_provenance_and_content(tmp_path: Path) -> None:
    configuration = _configuration(tmp_path, tmp_path / "outputs")
    workflow = run_synthetic_workflow(configuration)
    artifact = workflow.artifact_path.path
    from pace.experiment.artifacts import EvidenceArtifact

    evidence = EvidenceArtifact.model_validate_json(artifact.read_bytes(), strict=True)
    variant = evidence.model_copy(update={"schema_version": 3})
    Provenance_input = Provenance(
        configuration.scientific,
        configuration.splitting,
        configuration.scientific_digest,
        evidence.input_digest,
        evidence.implementation_digest,
    )
    with pytest.raises(ArtifactValidationError):
        validate_artifact(
            variant,
            Provenance_input,
        )
    Provenance_input = Provenance(
        configuration.scientific,
        configuration.splitting,
        "a" * 64,
        evidence.input_digest,
        evidence.implementation_digest,
    )
    with pytest.raises(ArtifactConflictError):
        validate_artifact(
            evidence,
            Provenance_input,
        )
    different_splitting = configuration.splitting.model_copy(
        update={
            "training_fraction": configuration.splitting.reference_fraction,
            "test_fraction": configuration.splitting.training_fraction,
        }
    )
    Provenance_input = Provenance(
        configuration.scientific,
        different_splitting,
        configuration.scientific_digest,
        evidence.input_digest,
        evidence.implementation_digest,
    )
    with pytest.raises(ArtifactConflictError):
        validate_artifact(
            evidence,
            Provenance_input,
        )
    Provenance_input = Provenance(
        configuration.scientific,
        configuration.splitting,
        configuration.scientific_digest,
        evidence.input_digest,
        "b" * 64,
    )
    with pytest.raises(ArtifactConflictError):
        validate_artifact(
            evidence,
            Provenance_input,
        )
    variant = evidence.model_copy(update={"content_digest": ("a" * 64)})
    Provenance_input = Provenance(
        configuration.scientific,
        configuration.splitting,
        configuration.scientific_digest,
        evidence.input_digest,
        evidence.implementation_digest,
    )
    with pytest.raises(ArtifactValidationError):
        validate_artifact(
            variant,
            Provenance_input,
        )


def test_logging_configuration_reuses_one_handler(tmp_path: Path) -> None:
    from pace.experiment.logging_setup import configure_logging
    from pace.types import LogLevel

    log_path = tmp_path / "events.jsonl"
    configure_logging(log_path, LogLevel.INFO)
    handler_count = sum(
        getattr(handler, "log_path", None) == log_path.resolve()
        for handler in logging.getLogger().handlers
    )
    configure_logging(log_path, LogLevel.INFO)
    repeated_handler_count = sum(
        getattr(handler, "log_path", None) == log_path.resolve()
        for handler in logging.getLogger().handlers
    )
    assert handler_count == 1
    assert repeated_handler_count == 1


def test_implementation_digest_scopes_ignore_unrelated_modules(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package_root = tmp_path / "repository" / "src" / "pace"
    (package_root / "experiment").mkdir(parents=True)
    (package_root / "reporting").mkdir()
    module_path = package_root / "experiment" / "artifacts.py"
    module_path.write_text("", encoding="utf-8")
    monkeypatch.setattr("pace.experiment.artifacts.__file__", str(module_path))

    def digests() -> tuple[str, str, str]:
        return (
            digest_implementation(ImplementationScope.CAMPAIGN),
            digest_implementation(ImplementationScope.SYNTHETIC),
            digest_implementation(ImplementationScope.REPORT),
        )

    baseline = digests()
    (package_root / "cli.py").write_text("changed = 1\n", encoding="utf-8")
    assert digests() == baseline
    (package_root / "reporting" / "metrics.py").write_text("changed = 1\n", encoding="utf-8")
    campaign, synthetic, report = digests()
    assert campaign == baseline[0]
    assert synthetic == baseline[1]
    assert report != baseline[2]
    (package_root / "experiment" / "smoke.py").write_text("changed = 1\n", encoding="utf-8")
    updated = digests()
    assert updated[0] == campaign
    assert updated[1] != synthetic
    assert updated[2] == report
    (package_root / "experiment" / "runner.py").write_text("changed = 1\n", encoding="utf-8")
    assert all(new != old for new, old in zip(digests(), updated, strict=True))
