import json
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

import pace.experiment.status as status_module
from pace.config import ResolvedConfiguration, load_configuration
from pace.experiment.artifacts import (
    CampaignOutcome,
    Provenance,
    campaign_artifact_path,
    digest_files,
    digest_implementation,
    reuse_or_write_artifact,
)
from pace.experiment.runner import run_dataset_workflow
from pace.experiment.status import (
    inspect_dataset_campaign_status,
    inspect_matrix_status,
    inspect_registered_dataset_campaign_status,
)
from pace.reporting.service import report_dataset_campaign
from pace.types import (
    ArtifactConflictError,
    ArtifactKind,
    ArtifactPath,
    ArtifactReuseState,
    ArtifactValidationError,
    AttackCapture,
    AttackType,
    AttackTypeName,
    AttackTypePerformance,
    BudgetAnalysis,
    BudgetAnalysisKind,
    BudgetPlanSummary,
    BudgetSweepPerformance,
    CampaignCompletionState,
    CampaignStatus,
    ClientName,
    DatasetClientData,
    DatasetClientId,
    DatasetName,
    DatasetValidationError,
    ExecutionDevice,
    FeatureName,
    FeatureSchemaId,
    LearningSeed,
    ReplacementPolicy,
    ScientificDigest,
    StudyStratum,
    TargetPerformance,
)
from tests.support import REPOSITORY_ROOT, campaign_configuration, target_provenance


def _configuration(repository_root: Path, output_root: Path) -> ResolvedConfiguration:
    source = REPOSITORY_ROOT / "configs" / "default.yaml"
    configuration = load_configuration(source, repository_root)
    return configuration.model_copy(
        update={"runtime": configuration.runtime.model_copy(update={"output_root": output_root})}
    )


def _campaign_outcome(
    seeds: tuple[LearningSeed, ...], dataset: DatasetName = DatasetName.N_BAIOT
) -> CampaignOutcome:
    target = DatasetClientId(dataset, ClientName("device"))
    attack_type = AttackType(AttackTypeName("GAFGYT/COMBO"))
    attack_performance = AttackTypePerformance(
        attack_type=attack_type,
        test_rows=10,
        detection_rate=0.8,
        local_detection_rate=0.7,
        max_fusion_detection_rate=0.6,
        peer_union_detection_rate=0.4,
        bonferroni_detection_rate=0.5,
    )
    return CampaignOutcome.from_targets(
        tuple(_target_performance(target, seed, attack_performance) for seed in seeds)
    )


def test_campaign_target_count_deduplicates_learning_seed_repetitions() -> None:
    seeds = (0, 1, 2)
    one_target = _campaign_outcome(seeds)

    assert one_target.target_count == 1

    second_target = DatasetClientId(DatasetName.N_BAIOT, ClientName("second-device"))
    attack_type = AttackType(AttackTypeName("GAFGYT/COMBO"))
    performance = AttackTypePerformance(
        attack_type=attack_type,
        test_rows=10,
        detection_rate=0.8,
        local_detection_rate=0.7,
        max_fusion_detection_rate=0.6,
        peer_union_detection_rate=0.4,
        bonferroni_detection_rate=0.5,
    )
    two_targets = CampaignOutcome.from_targets(
        (
            *one_target.targets,
            *(_target_performance(second_target, seed, performance) for seed in seeds),
        )
    )

    assert two_targets.target_count == 2

    incomplete_seed_coverage = (
        *one_target.targets,
        _target_performance(second_target, seeds[0], performance),
        _target_performance(second_target, seeds[1], performance),
    )
    with pytest.raises(ValueError, match="coverage must match"):
        CampaignOutcome.from_targets(incomplete_seed_coverage)


def _target_performance(
    target: DatasetClientId,
    seed: LearningSeed,
    attack_performance: AttackTypePerformance,
) -> TargetPerformance:
    split_digest, peer_model_seeds, local_model_seed = target_provenance(target, seed)
    return TargetPerformance(
        target=target,
        seed=seed,
        split_digest=split_digest,
        peer_model_seeds=peer_model_seeds,
        local_model_seed=local_model_seed,
        benign_test_rows=20,
        realized_false_alert_rate=0.02,
        attack_types=(attack_performance,),
        local_false_alert_rate=0.01,
        max_fusion_false_alert_rate=0.03,
        peer_union_false_alert_rate=0.02,
        bonferroni_false_alert_rate=0.025,
        peer_count=4,
        budget_sweep=(
            BudgetSweepPerformance(multiplier=0.4, detection_rate=0.4, false_alert_rate=0.0),
            BudgetSweepPerformance(multiplier=0.75, detection_rate=0.6, false_alert_rate=0.02),
            BudgetSweepPerformance(multiplier=1.0, detection_rate=0.8, false_alert_rate=0.04),
            BudgetSweepPerformance(multiplier=1.4, detection_rate=0.9, false_alert_rate=0.1),
        ),
        study_stratum=StudyStratum[target.dataset.name],
    )


def _write_verified_campaign(
    configuration: ResolvedConfiguration, raw_root: Path
) -> tuple[Path, Path]:
    source_file = raw_root / "N-BaIoT" / "device" / "benign_traffic.csv"
    source_file.parent.mkdir(parents=True)
    source_file.write_bytes(b"immutable source fixture")
    source_files = (source_file,)
    input_digest = digest_files(source_files, raw_root)
    implementation_digest = digest_implementation()
    destination = campaign_artifact_path(
        configuration.runtime.output_root,
        DatasetName.N_BAIOT,
        configuration.scientific_digest,
    )
    artifact, reuse_state = reuse_or_write_artifact(
        destination.path,
        _campaign_outcome(configuration.scientific.learning_seeds),
        Provenance(
            configuration.scientific,
            configuration.splitting,
            configuration.scientific_digest,
            input_digest,
            implementation_digest,
            ArtifactKind.DATASET_CAMPAIGN,
            DatasetName.N_BAIOT,
        ),
        ReplacementPolicy.PRESERVE_VALID,
    )
    assert reuse_state is ArtifactReuseState.CREATED
    assert artifact.content_digest
    return source_file, destination.path


def test_dataset_workflow_persists_and_reports_by_dataset_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw_root = tmp_path / "raw"
    source_file = raw_root / "TON-IoT" / "processed.csv"
    source_file.parent.mkdir(parents=True)
    source_file.write_bytes(b"immutable TON-IoT source")
    configuration = _configuration(tmp_path, tmp_path / "outputs")
    configuration = configuration.model_copy(
        update={"runtime": configuration.runtime.model_copy(update={"raw_data_root": raw_root})}
    )
    dataset = DatasetName.TON_IOT
    client = DatasetClientData(
        DatasetClientId(dataset, ClientName("device")),
        StudyStratum.TON_IOT,
        FeatureSchemaId.TON_IOT_ZEEK_ENGINEERED_V1,
        (FeatureName("feature"),),
        np.asarray([[0.0], [1.0], [2.0], [3.0]], dtype=np.float64),
        np.arange(4, dtype=np.int64),
        (
            AttackCapture(
                AttackType(AttackTypeName("botnet")),
                source_file,
                np.asarray([[4.0], [5.0]], dtype=np.float64),
                np.arange(2, dtype=np.int64),
            ),
        ),
    )
    study_outcome = _campaign_outcome(configuration.scientific.learning_seeds, dataset)

    def return_study(
        clients: tuple[DatasetClientData, ...],
        _configuration: ResolvedConfiguration,
    ) -> tuple[TargetPerformance, ...]:
        assert len(clients) == 1
        assert clients[0].client == client.client
        return study_outcome.targets

    monkeypatch.setattr("pace.experiment.runner.run_dataset_study", return_study)
    source_files = (source_file,)
    first = run_dataset_workflow(
        clients=(client,), source_files=source_files, configuration=configuration
    )
    first_hash = sha256(first.artifact_path.path.read_bytes()).hexdigest()
    assert first.artifact_path.path.parts[-3] == "ton_iot"

    def fail_if_study_recomputes(
        _clients: tuple[DatasetClientData, ...],
        _configuration: ResolvedConfiguration,
    ) -> tuple[TargetPerformance, ...]:
        raise AssertionError("verified dataset campaign should be reused")

    monkeypatch.setattr("pace.experiment.runner.run_dataset_study", fail_if_study_recomputes)
    second = run_dataset_workflow((client,), source_files, configuration)
    assert second.reuse_state is ArtifactReuseState.REUSED
    assert sha256(second.artifact_path.path.read_bytes()).hexdigest() == first_hash
    altered_client = replace(
        client,
        benign_features=np.asarray(
            client.benign_features + 0.25,
            dtype=np.float64,
        ),
    )
    with pytest.raises(ArtifactConflictError, match="provenance"):
        run_dataset_workflow((altered_client,), source_files, configuration)
    with pytest.raises(DatasetValidationError, match="identifiers must be unique"):
        run_dataset_workflow((client, client), source_files, configuration)

    report = report_dataset_campaign(StudyStratum.TON_IOT, source_files, configuration)
    repeated_report = report_dataset_campaign(StudyStratum.TON_IOT, source_files, configuration)
    assert report.reuse_state is ArtifactReuseState.CREATED
    assert repeated_report.reuse_state is ArtifactReuseState.REUSED
    assert report.artifact_path.path.parts[-4] == "ton_iot"
    status = inspect_dataset_campaign_status(dataset, source_files, configuration)
    assert status.dataset is dataset
    assert status.state is CampaignCompletionState.COMPLETE
    assert status.target_count == 1
    cuda_configuration = configuration.model_copy(
        update={
            "runtime": configuration.runtime.model_copy(
                update={"execution_device": ExecutionDevice.CUDA}
            )
        }
    )
    with pytest.raises(ArtifactConflictError, match="provenance"):
        run_dataset_workflow((client,), source_files, cuda_configuration)
    cuda_status = inspect_dataset_campaign_status(dataset, source_files, cuda_configuration)
    assert cuda_status.state is CampaignCompletionState.STALE
    monkeypatch.setattr("pace.experiment.runner.run_dataset_study", return_study)
    replaced = run_dataset_workflow(
        (client,),
        source_files,
        cuda_configuration,
        ReplacementPolicy.REPLACE_REQUESTED,
    )
    assert replaced.reuse_state is ArtifactReuseState.CREATED
    persisted_artifact = json.loads(replaced.artifact_path.path.read_bytes())
    assert persisted_artifact["execution_device"] == ExecutionDevice.CUDA.name
    assert (
        inspect_dataset_campaign_status(dataset, source_files, cuda_configuration).state
        is CampaignCompletionState.COMPLETE
    )


def test_campaign_status_is_missing_without_treating_paths_as_complete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configuration = _configuration(tmp_path, tmp_path / "outputs")

    def unexpected_source_inspection(
        _raw_root: Path, _dataset: DatasetName, _stratum: StudyStratum
    ) -> tuple[Path, ...]:
        pytest.fail("Missing artifacts do not require source inspection")

    monkeypatch.setattr(
        "pace.experiment.status.dataset_source_files",
        unexpected_source_inspection,
    )

    status = inspect_registered_dataset_campaign_status(DatasetName.N_BAIOT, configuration)

    assert status.state is CampaignCompletionState.MISSING
    assert status.target_count is None


def test_campaign_status_verifies_provenance_before_reporting_complete(
    tmp_path: Path,
) -> None:
    configuration = _configuration(tmp_path, tmp_path / "outputs")
    raw_root = configuration.runtime.raw_data_root
    _, artifact_path = _write_verified_campaign(configuration, raw_root)
    original_content = artifact_path.read_bytes()
    original_modified = artifact_path.stat().st_mtime_ns

    status = inspect_registered_dataset_campaign_status(DatasetName.N_BAIOT, configuration)
    repeated_status = inspect_registered_dataset_campaign_status(DatasetName.N_BAIOT, configuration)

    assert status.state is CampaignCompletionState.COMPLETE
    assert status.target_count == 1
    assert status.artifact_path.path == artifact_path
    assert repeated_status == status
    assert sha256(artifact_path.read_bytes()).digest() == sha256(original_content).digest()
    assert artifact_path.stat().st_mtime_ns == original_modified


def test_campaign_status_marks_changed_input_provenance_stale(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configuration = _configuration(tmp_path, tmp_path / "outputs")
    _write_verified_campaign(configuration, configuration.runtime.raw_data_root)

    def changed_input_digest(_source_files: tuple[Path, ...], _root: Path) -> ScientificDigest:
        return "a" * 64

    monkeypatch.setattr(
        "pace.experiment.artifacts.digest_files",
        changed_input_digest,
    )

    status = inspect_registered_dataset_campaign_status(DatasetName.N_BAIOT, configuration)

    assert status.state is CampaignCompletionState.STALE
    assert status.target_count is None


def test_campaign_status_marks_corrupt_artifact_invalid(tmp_path: Path) -> None:
    configuration = _configuration(tmp_path, tmp_path / "outputs")
    _, artifact_path = _write_verified_campaign(
        configuration,
        configuration.runtime.raw_data_root,
    )
    artifact_path.write_bytes(b"{}")

    status = inspect_registered_dataset_campaign_status(DatasetName.N_BAIOT, configuration)

    assert status.state is CampaignCompletionState.INVALID
    assert status.target_count is None


def test_campaign_status_marks_unavailable_source_stale(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configuration = _configuration(tmp_path, tmp_path / "outputs")
    _, artifact_path = _write_verified_campaign(
        configuration,
        configuration.runtime.raw_data_root,
    )

    def unavailable_sources(
        raw_root: Path, _dataset: DatasetName, _stratum: StudyStratum
    ) -> tuple[Path, ...]:
        raise DatasetValidationError(f"No N-BaIoT sources at {raw_root}")

    monkeypatch.setattr("pace.experiment.status.dataset_source_files", unavailable_sources)

    status = inspect_registered_dataset_campaign_status(DatasetName.N_BAIOT, configuration)

    assert status.state is CampaignCompletionState.STALE
    assert status.artifact_path.path == artifact_path


def _runtime_configuration(tmp_path: Path) -> ResolvedConfiguration:
    configuration = campaign_configuration()
    return configuration.model_copy(
        update={
            "runtime": configuration.runtime.model_copy(
                update={"raw_data_root": tmp_path / "raw", "output_root": tmp_path / "outputs"}
            )
        }
    )


def test_status_rejects_strata_outside_the_requested_dataset(tmp_path: Path) -> None:
    runtime_configuration = _runtime_configuration(tmp_path)
    with pytest.raises(DatasetValidationError, match="does not belong"):
        inspect_dataset_campaign_status(
            DatasetName.N_BAIOT, (), runtime_configuration, StudyStratum.IOT_23
        )


def test_status_reports_missing_and_corrupt_campaigns(tmp_path: Path) -> None:
    configuration = _runtime_configuration(tmp_path)
    source = tmp_path / "raw" / "N-BaIoT" / "device" / "benign_traffic.csv"
    source.parent.mkdir(parents=True)
    source.write_text("x", encoding="utf-8")
    missing = inspect_dataset_campaign_status(DatasetName.N_BAIOT, (source,), configuration)
    assert missing.state is CampaignCompletionState.MISSING
    destination = campaign_artifact_path(
        configuration.runtime.output_root, StudyStratum.N_BAIOT, configuration.scientific_digest
    )
    destination.path.parent.mkdir(parents=True)
    destination.path.write_text("{not json", encoding="utf-8")
    corrupt = inspect_dataset_campaign_status(DatasetName.N_BAIOT, (source,), configuration)
    assert corrupt.state is CampaignCompletionState.INVALID


def test_status_treats_sources_outside_the_raw_root_as_stale(tmp_path: Path) -> None:
    configuration = _runtime_configuration(tmp_path)
    outside = tmp_path / "elsewhere.csv"
    outside.write_text("x", encoding="utf-8")
    destination = campaign_artifact_path(
        configuration.runtime.output_root, StudyStratum.N_BAIOT, configuration.scientific_digest
    )
    destination.path.parent.mkdir(parents=True)
    destination.path.write_text("{}", encoding="utf-8")
    stale = inspect_dataset_campaign_status(DatasetName.N_BAIOT, (outside,), configuration)
    assert stale.state is CampaignCompletionState.STALE


def test_registered_status_discovers_sources_from_the_raw_root(tmp_path: Path) -> None:
    configuration = _runtime_configuration(tmp_path)
    device = tmp_path / "raw" / "N-BaIoT" / "device"
    device.mkdir(parents=True)
    (device / "benign_traffic.csv").write_text("x", encoding="utf-8")
    status = inspect_registered_dataset_campaign_status(DatasetName.N_BAIOT, configuration)
    assert status.state is CampaignCompletionState.MISSING


def test_matrix_status_inspects_every_variant_and_falls_back_without_a_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configuration = campaign_configuration()
    analysis = BudgetAnalysis(
        BudgetAnalysisKind.UNIVERSAL_PRIMARY, (100, 400), (ClientName("target"),)
    )
    status = CampaignStatus(
        DatasetName.N_BAIOT,
        StudyStratum.N_BAIOT,
        ArtifactPath(tmp_path / "cell.json"),
        CampaignCompletionState.COMPLETE,
        1,
    )
    inspect = Mock(return_value=status)
    monkeypatch.setattr(status_module, "dataset_source_files", Mock(return_value=()))
    monkeypatch.setattr(status_module, "digest_source_files", Mock(return_value="1" * 64))
    monkeypatch.setattr(
        status_module,
        "read_budget_plan_summary",
        Mock(return_value=BudgetPlanSummary((analysis,), 1)),
    )
    monkeypatch.setattr(status_module, "inspect_dataset_campaign_status", inspect)

    cells = inspect_matrix_status(StudyStratum.N_BAIOT, configuration)

    assert len(cells) == inspect.call_count == 2 * len(configuration.protocol_matrix.alpha_levels)
    monkeypatch.setattr(
        status_module,
        "read_budget_plan_summary",
        Mock(side_effect=ArtifactValidationError("stale plan")),
    )
    fallback = Mock(return_value=status)
    monkeypatch.setattr(status_module, "inspect_registered_dataset_campaign_status", fallback)
    assert inspect_matrix_status(StudyStratum.N_BAIOT, configuration) == (status,)
    assert fallback.call_count == 1
