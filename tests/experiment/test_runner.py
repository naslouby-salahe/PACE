import copy
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest

from pace.config import (
    ResolvedConfiguration,
    configuration_for_protocol_variant,
)
from pace.experiment import artifacts, runner
from pace.experiment.artifacts import derive_learning_seed
from pace.experiment.runner import (
    run_dataset_matrix_workflow,
    run_dataset_study,
    run_dataset_workflow,
    run_registered_dataset_matrix_workflow,
)
from pace.experiment.splitting import PooledStandardizer
from pace.reporting.service import report_dataset_campaign
from pace.types import (
    AlphaLevel,
    ArtifactConflictError,
    ArtifactPath,
    ArtifactReuseState,
    ArtifactValidationError,
    ArtifactVerification,
    AttackType,
    AttackTypeName,
    AttackTypePerformance,
    BudgetSweepPerformance,
    CampaignWorkflowResult,
    ClientName,
    DatasetClientData,
    DatasetClientId,
    DatasetName,
    DatasetValidationError,
    FeatureSchemaId,
    ReplacementPolicy,
    SeedDomain,
    StudyStratum,
    TargetCohortRole,
    TargetPerformance,
    VerificationState,
)
from tests.support import (
    campaign_configuration,
    campaign_devices,
    protocol_matrix,
    run_single_seed_campaign,
    target_provenance,
)


def run_registered_dataset_workflow(
    dataset: DatasetName,
    configuration: ResolvedConfiguration,
    replacement_policy: ReplacementPolicy = ReplacementPolicy.PRESERVE_VALID,
) -> CampaignWorkflowResult:
    sources = runner.dataset_source_files(configuration.runtime.raw_data_root, dataset)
    return run_dataset_workflow(campaign_devices(), sources, configuration, replacement_policy)


def test_dataset_matrix_workflow_runs_every_configured_protocol_variant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configuration = campaign_configuration()
    matrix = protocol_matrix((32,))
    configuration = configuration.model_copy(
        update={
            "protocol_matrix": matrix,
            "runtime": configuration.runtime.model_copy(
                update={
                    "raw_data_root": tmp_path / "raw",
                    "output_root": tmp_path / "outputs",
                }
            ),
        }
    )
    source_file = tmp_path / "raw" / "source.csv"
    source_file.parent.mkdir(parents=True)
    source_file.write_text("stable source", encoding="utf-8")
    variants: list[ResolvedConfiguration] = []
    result = CampaignWorkflowResult(
        ArtifactPath(tmp_path / "campaign.json"),
        ArtifactVerification(VerificationState.VERIFIED, "a" * 64),
        ArtifactReuseState.REUSED,
        5,
    )

    def run_variant(
        _clients: tuple[DatasetClientData, ...],
        _source_files: tuple[Path, ...],
        variant: ResolvedConfiguration,
        _replacement_policy: ReplacementPolicy,
        _matrix_cache: object = None,
    ) -> CampaignWorkflowResult:
        variants.append(variant)
        return result

    monkeypatch.setattr("pace.experiment.runner.run_dataset_workflow", run_variant)

    results = run_dataset_matrix_workflow(
        campaign_devices(), (source_file,), configuration, ReplacementPolicy.PRESERVE_VALID
    )

    assert tuple(entry.result for entry in results) == (result, result)
    assert tuple(variant.scientific.alpha for variant in variants) == (
        AlphaLevel.ALPHA_001,
        AlphaLevel.ALPHA_005,
    )
    assert len({variant.scientific_digest for variant in variants}) == 2


def test_dataset_matrix_reuses_peer_training_and_campaign_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configuration = campaign_configuration()
    configuration = configuration.model_copy(
        update={
            "runtime": configuration.runtime.model_copy(
                update={
                    "raw_data_root": tmp_path / "raw",
                    "output_root": tmp_path / "outputs",
                }
            ),
            "protocol_matrix": protocol_matrix((32,)),
        }
    )
    clients = campaign_devices()
    raw_source = tmp_path / "raw" / "fixture.csv"
    raw_source.parent.mkdir(parents=True)
    raw_source.write_text("fixed source", encoding="utf-8")
    original_training = runner.train_peer_mlp
    training_spy = Mock(wraps=original_training)
    monkeypatch.setattr(runner, "train_peer_mlp", training_spy)
    original_local = runner.compute_local_evidence
    local_spy = Mock(wraps=original_local)
    monkeypatch.setattr(runner, "compute_local_evidence", local_spy)

    first = run_dataset_matrix_workflow(
        clients,
        (raw_source,),
        configuration,
        ReplacementPolicy.PRESERVE_VALID,
    )
    first_bytes = tuple(entry.result.artifact_path.path.read_bytes() for entry in first)
    second = run_dataset_matrix_workflow(
        clients,
        (raw_source,),
        configuration,
        ReplacementPolicy.PRESERVE_VALID,
    )

    assert len(first) == len(second) == 2
    assert all(entry.result.reuse_state is ArtifactReuseState.CREATED for entry in first)
    assert all(entry.result.reuse_state is ArtifactReuseState.REUSED for entry in second)
    assert tuple(entry.result.artifact_path.path.read_bytes() for entry in second) == first_bytes
    assert training_spy.call_count == len(clients) * len(configuration.scientific.learning_seeds)
    assert local_spy.call_count == len(clients) * len(configuration.scientific.learning_seeds)


def test_dataset_matrix_rejects_source_changes_between_variants(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configuration = campaign_configuration()
    configuration = configuration.model_copy(
        update={
            "runtime": configuration.runtime.model_copy(
                update={
                    "raw_data_root": tmp_path / "raw",
                    "output_root": tmp_path / "outputs",
                }
            ),
            "protocol_matrix": protocol_matrix((32,)),
        }
    )
    source_file = tmp_path / "raw" / "source.csv"
    source_file.parent.mkdir(parents=True)
    source_file.write_text("stable source", encoding="utf-8")
    result = CampaignWorkflowResult(
        ArtifactPath(tmp_path / "campaign.json"),
        ArtifactVerification(VerificationState.VERIFIED, "a" * 64),
        ArtifactReuseState.REUSED,
        5,
    )
    run_count = 0

    def run_variant(
        _clients: tuple[DatasetClientData, ...],
        _source_files: tuple[Path, ...],
        _configuration: ResolvedConfiguration,
        _replacement_policy: ReplacementPolicy,
        _matrix_cache: object = None,
    ) -> CampaignWorkflowResult:
        nonlocal run_count
        run_count += 1
        source_file.write_text("changed source", encoding="utf-8")
        return result

    monkeypatch.setattr("pace.experiment.runner.run_dataset_workflow", run_variant)

    campaign_devices_input = campaign_devices()
    with pytest.raises(DatasetValidationError, match="changed during matrix execution"):
        run_dataset_matrix_workflow(
            campaign_devices_input, (source_file,), configuration, ReplacementPolicy.PRESERVE_VALID
        )

    assert run_count == 1


def test_campaign_uses_held_out_types_and_excludes_target_from_peer_set() -> None:
    seed = 9
    result = run_single_seed_campaign(campaign_devices(), campaign_configuration(), seed)

    assert len(result) == 5
    assert all(target.peer_count == 4 for target in result)
    assert all(
        target.local_model_seed.seed
        == derive_learning_seed(seed, target.target, SeedDomain.LOCAL_MODEL)
        for target in result
    )
    assert all(
        provenance.seed == derive_learning_seed(seed, provenance.owner, SeedDomain.PEER_MODEL)
        for target in result
        for provenance in target.peer_model_seeds
    )
    assert all(target.benign_test_rows == 24 for target in result)
    assert all(len(target.attack_types) == 2 for target in result)
    assert all(
        tuple(point.multiplier for point in target.budget_sweep) == (0.4, 0.75, 1.0, 1.4)
        for target in result
    )
    assert all(attack.test_rows == 15 for target in result for attack in target.attack_types)
    assert all(0.0 <= target.realized_false_alert_rate <= 1.0 for target in result)


def test_campaign_uses_minimum_peer_count_for_target_eligibility() -> None:
    campaign_devices_input = campaign_devices()
    campaign_configuration_input = campaign_configuration()
    with pytest.raises(DatasetValidationError, match="No target satisfies"):
        run_single_seed_campaign(campaign_devices_input[:-1], campaign_configuration_input)


def test_outside_peer_pool_client_trains_no_expert_and_is_not_a_target() -> None:
    devices = campaign_devices()
    extra_peer = replace(
        devices[0],
        client=DatasetClientId(DatasetName.N_BAIOT, ClientName("extra-peer")),
    )
    pool = (*devices, extra_peer)
    excluded = replace(
        devices[-1],
        client=DatasetClientId(DatasetName.N_BAIOT, ClientName("excluded-client")),
        target_cohort_role=TargetCohortRole.OUTSIDE_PEER_POOL,
    )
    configuration = campaign_configuration()

    baseline = run_single_seed_campaign(pool, configuration)
    results = run_single_seed_campaign((*pool, excluded), configuration)

    assert results == baseline
    assert len(results) == 6
    assert all(result.peer_count == 5 for result in results)
    assert all(
        provenance.owner != excluded.client
        for result in results
        for provenance in result.peer_model_seeds
    )


def test_benign_only_peer_contributes_to_scaling_but_not_peer_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    devices = campaign_devices()
    benign_peer = replace(
        devices[0],
        client=DatasetClientId(DatasetName.N_BAIOT, ClientName("benign-only-peer")),
        attack_captures=(),
        target_cohort_role=TargetCohortRole.PEER_ONLY,
    )
    training_matrices: list[tuple[np.ndarray, ...]] = []
    original_standardizer = runner.fit_pooled_standardizer

    def capture_training_matrices(
        matrices: tuple[np.ndarray, ...],
    ) -> PooledStandardizer:
        training_matrices.append(matrices)
        return original_standardizer(matrices)

    monkeypatch.setattr(runner, "fit_pooled_standardizer", capture_training_matrices)
    results = run_single_seed_campaign((*devices, benign_peer), campaign_configuration())

    assert len(training_matrices) == 1
    assert len(training_matrices[0]) == len(devices) + 1
    assert len(results) == len(devices)
    assert all(result.peer_count == len(devices) - 1 for result in results)
    assert all(
        provenance.owner != benign_peer.client
        for result in results
        for provenance in result.peer_model_seeds
    )


def test_campaign_results_are_invariant_to_client_and_capture_order() -> None:
    devices = campaign_devices()
    reversed_inputs = tuple(
        replace(device, attack_captures=tuple(reversed(device.attack_captures)))
        for device in reversed(devices)
    )
    configuration = campaign_configuration()
    seed = 13

    canonical_result = run_single_seed_campaign(devices, configuration, seed)
    permuted_result = run_single_seed_campaign(reversed_inputs, configuration, seed)

    assert permuted_result == canonical_result


def test_dataset_client_rejects_repeated_attack_captures() -> None:
    device = campaign_devices()[0]

    with pytest.raises(ValueError, match="cannot repeat attack type captures"):
        replace(device, attack_captures=(*device.attack_captures, device.attack_captures[0]))


def test_generic_campaign_rejects_cross_dataset_peer_pool() -> None:
    devices = campaign_devices()
    foreign_client = replace(
        devices[-1],
        client=DatasetClientId(DatasetName.TON_IOT, ClientName("foreign")),
        study_stratum=StudyStratum.TON_IOT,
        feature_schema=FeatureSchemaId.TON_IOT_ZEEK_ENGINEERED_V1,
    )

    campaign_configuration_input = campaign_configuration()
    with pytest.raises(DatasetValidationError, match="one dataset"):
        run_single_seed_campaign((*devices[:-1], foreign_client), campaign_configuration_input)


def test_campaign_repeats_deterministically() -> None:
    devices = campaign_devices()
    configuration = campaign_configuration()

    first = run_single_seed_campaign(devices, configuration, 11)
    repeated = run_single_seed_campaign(devices, configuration, 11)
    generic = run_single_seed_campaign(devices, configuration, 11)

    assert repeated == first
    assert generic == first


def test_study_executes_each_configured_learning_seed() -> None:
    configuration = campaign_configuration()

    result = run_dataset_study(campaign_devices(), configuration)

    assert len(result) == 15
    assert {target.seed for target in result} == set(configuration.scientific.learning_seeds)
    assert len({(target.target, target.seed) for target in result}) == len(result)


def test_target_cohort_restricts_evaluated_targets_without_changing_their_results() -> None:
    devices = campaign_devices()
    base = campaign_configuration()
    cohort = (ClientName("client-1"), ClientName("client-3"))
    restricted = base.model_copy(
        update={"scientific": base.scientific.model_copy(update={"target_clients": cohort})}
    )

    everyone = run_single_seed_campaign(devices, base, 11)
    selected = run_single_seed_campaign(devices, restricted, 11)

    assert tuple(target.target.name for target in selected) == cohort
    assert selected == tuple(target for target in everyone if target.target.name in cohort)
    assert all(target.peer_count == 4 for target in selected)


def test_target_cohort_rejects_clients_outside_the_eligible_targets() -> None:
    base = campaign_configuration()
    unknown = base.model_copy(
        update={
            "scientific": base.scientific.model_copy(
                update={"target_clients": (ClientName("client-1"), ClientName("ghost"))}
            )
        }
    )

    with pytest.raises(DatasetValidationError, match="outside the eligible study targets"):
        run_single_seed_campaign(campaign_devices(), unknown, 11)


def test_target_cohort_changes_the_scientific_digest_of_the_variant() -> None:
    base = campaign_configuration()
    matrix = base.protocol_matrix
    everyone = configuration_for_protocol_variant(base, matrix.alpha_levels[0], 100)
    subset = configuration_for_protocol_variant(
        base, matrix.alpha_levels[0], 100, (ClientName("client-1"),)
    )

    assert everyone.scientific.target_clients == ()
    assert subset.scientific.target_clients == (ClientName("client-1"),)
    assert subset.scientific_digest != everyone.scientific_digest


def test_inactive_provenance_reuses_a_campaign_after_implementation_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw_root = tmp_path / "raw"
    source = raw_root / "N-BaIoT" / "capture.csv"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"stable dataset identity")
    base = campaign_configuration()
    strict = base.model_copy(
        update={
            "runtime": base.runtime.model_copy(
                update={"raw_data_root": raw_root, "output_root": tmp_path / "outputs"}
            )
        }
    )
    relaxed = strict.model_copy(
        update={"runtime": strict.runtime.model_copy(update={"provenance_active": False})}
    )
    devices = campaign_devices()

    created = run_dataset_workflow(devices, (source,), strict)
    monkeypatch.setattr(artifacts, "digest_implementation", Mock(return_value="8" * 64))

    with pytest.raises(ArtifactConflictError, match="provenance does not match"):
        run_dataset_workflow(devices, (source,), strict)
    reused = run_dataset_workflow(devices, (source,), relaxed)

    assert created.reuse_state is ArtifactReuseState.CREATED
    assert reused.reuse_state is ArtifactReuseState.REUSED
    assert reused.verification.content_digest == created.verification.content_digest
    assert reused.target_count == created.target_count


def test_campaign_artifact_is_verified_reused_and_overwritten_only_when_requested(
    tmp_path: Path,
) -> None:
    raw_root = tmp_path / "raw"
    source = raw_root / "N-BaIoT" / "capture.csv"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"stable dataset identity")
    base = campaign_configuration()
    configuration = base.model_copy(
        update={
            "runtime": base.runtime.model_copy(
                update={"raw_data_root": raw_root, "output_root": tmp_path / "outputs"}
            )
        }
    )
    target = DatasetClientId(DatasetName.N_BAIOT, ClientName("target"))
    seed = 3
    split_digest, peer_model_seeds, local_model_seed = target_provenance(target, seed)
    result = TargetPerformance(
        target=target,
        seed=seed,
        split_digest=split_digest,
        peer_model_seeds=peer_model_seeds,
        local_model_seed=local_model_seed,
        benign_test_rows=20,
        realized_false_alert_rate=0.05,
        attack_types=(
            AttackTypePerformance(
                attack_type=AttackType(AttackTypeName("GAFGYT/COMBO")),
                test_rows=15,
                detection_rate=0.8,
                local_detection_rate=0.6,
                max_fusion_detection_rate=0.5,
                peer_union_detection_rate=0.4,
                bonferroni_detection_rate=0.7,
            ),
        ),
        local_false_alert_rate=0.02,
        max_fusion_false_alert_rate=0.04,
        peer_union_false_alert_rate=0.03,
        bonferroni_false_alert_rate=0.045,
        peer_count=4,
        budget_sweep=(
            BudgetSweepPerformance(multiplier=0.4, detection_rate=0.4, false_alert_rate=0.0),
            BudgetSweepPerformance(multiplier=0.75, detection_rate=0.6, false_alert_rate=0.02),
            BudgetSweepPerformance(multiplier=1.0, detection_rate=0.8, false_alert_rate=0.04),
            BudgetSweepPerformance(multiplier=1.4, detection_rate=0.9, false_alert_rate=0.1),
        ),
        study_stratum=StudyStratum.N_BAIOT,
    )
    study_results = tuple(
        replace(
            result,
            seed=seed,
            split_digest=target_provenance(result.target, seed)[0],
            peer_model_seeds=target_provenance(result.target, seed)[1],
            local_model_seed=target_provenance(result.target, seed)[2],
        )
        for seed in configuration.scientific.learning_seeds
    )
    with (
        patch("pace.experiment.runner.run_dataset_study", return_value=study_results),
    ):
        first = run_registered_dataset_workflow(DatasetName.N_BAIOT, configuration)
        artifact_hash = sha256(first.artifact_path.path.read_bytes()).hexdigest()
        artifact_mtime = first.artifact_path.path.stat().st_mtime_ns
        assert first.reuse_state is ArtifactReuseState.CREATED

    def fail_if_campaign_recomputes(
        _clients: tuple[DatasetClientData, ...],
        _configuration: ResolvedConfiguration,
    ) -> tuple[TargetPerformance, ...]:
        raise AssertionError("verified result should be reused")

    source.write_bytes(b"changed dataset identity")
    with (
        patch(
            "pace.experiment.runner.run_dataset_study",
            fail_if_campaign_recomputes,
        ),
        pytest.raises(ArtifactConflictError),
    ):
        run_registered_dataset_workflow(DatasetName.N_BAIOT, configuration)
    source.write_bytes(b"stable dataset identity")

    with (
        patch(
            "pace.experiment.runner.run_dataset_study",
            fail_if_campaign_recomputes,
        ),
    ):
        repeated = run_registered_dataset_workflow(DatasetName.N_BAIOT, configuration)
    assert repeated.reuse_state is ArtifactReuseState.REUSED
    assert repeated.target_count == 1
    assert sha256(repeated.artifact_path.path.read_bytes()).hexdigest() == artifact_hash
    assert repeated.artifact_path.path.stat().st_mtime_ns == artifact_mtime
    assert len(tuple(first.artifact_path.path.parent.iterdir())) == 1
    first.artifact_path.path.write_text("{}", encoding="utf-8")
    with (
        pytest.raises(ArtifactValidationError),
    ):
        run_registered_dataset_workflow(DatasetName.N_BAIOT, configuration)
    with (
        patch("pace.experiment.runner.run_dataset_study", return_value=study_results),
    ):
        overwritten = run_registered_dataset_workflow(
            DatasetName.N_BAIOT,
            configuration,
            replacement_policy=ReplacementPolicy.REPLACE_REQUESTED,
        )
    assert overwritten.reuse_state is ArtifactReuseState.CREATED
    report = report_dataset_campaign(
        StudyStratum.N_BAIOT,
        runner.dataset_source_files(configuration.runtime.raw_data_root, DatasetName.N_BAIOT),
        configuration,
    )
    assert report.metrics.target_count == 1


def test_campaign_inputs_are_validated() -> None:
    configuration = campaign_configuration()
    devices = campaign_devices()
    with pytest.raises(DatasetValidationError, match="at least one client"):
        run_single_seed_campaign((), configuration)
    stratum_clone = copy.copy(devices[0])
    object.__setattr__(
        stratum_clone, "client", DatasetClientId(DatasetName.N_BAIOT, ClientName("y"))
    )
    object.__setattr__(stratum_clone, "study_stratum", StudyStratum.IOT_23)
    with pytest.raises(DatasetValidationError, match="exactly one study stratum"):
        run_single_seed_campaign((*devices, stratum_clone), configuration)
    schema_clone = copy.copy(devices[0])
    object.__setattr__(
        schema_clone, "client", DatasetClientId(DatasetName.N_BAIOT, ClientName("z"))
    )
    object.__setattr__(schema_clone, "feature_schema", FeatureSchemaId.IOT_23_ZEEK_ENGINEERED_V1)
    with pytest.raises(DatasetValidationError, match="exactly one feature schema"):
        run_single_seed_campaign((*devices, schema_clone), configuration)
    names_clone = copy.copy(devices[0])
    object.__setattr__(names_clone, "client", DatasetClientId(DatasetName.N_BAIOT, ClientName("w")))
    object.__setattr__(names_clone, "feature_names", tuple(reversed(devices[0].feature_names)))
    with pytest.raises(DatasetValidationError, match="share a feature schema"):
        run_single_seed_campaign((*devices, names_clone), configuration)


def test_peer_pool_and_reported_target_eligibility_are_enforced() -> None:
    configuration = campaign_configuration()
    devices = campaign_devices()
    outside = tuple(
        replace(device, target_cohort_role=TargetCohortRole.OUTSIDE_PEER_POOL) for device in devices
    )
    with pytest.raises(DatasetValidationError, match="no clients in its frozen peer pool"):
        run_single_seed_campaign(outside, configuration)
    reported = tuple(
        replace(device, target_cohort_role=TargetCohortRole.REPORTED_TARGET)
        for device in devices[:4]
    )
    with pytest.raises(DatasetValidationError, match="fail configured eligibility"):
        run_single_seed_campaign(reported, configuration)


def test_matrix_cache_must_match_clients_and_peer_training() -> None:
    configuration = campaign_configuration()
    devices = campaign_devices()
    cache = runner.MatrixTrainingCache(
        devices, configuration.scientific.peer_training, configuration.splitting
    )
    campaign_devices_input = campaign_devices()
    with pytest.raises(DatasetValidationError, match="different client data"):
        run_dataset_study(campaign_devices_input, configuration, cache)
    other_training = configuration.scientific.peer_training.model_copy(
        update={"batch_size": type(configuration.scientific.peer_training.batch_size)(7)}
    )
    mismatched = runner.MatrixTrainingCache(devices, other_training, configuration.splitting)
    with pytest.raises(DatasetValidationError, match="incompatible peer training"):
        run_dataset_study(devices, configuration, mismatched)


def test_workflow_rejects_mismatched_or_changing_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = campaign_configuration()
    configuration = base.model_copy(
        update={
            "runtime": base.runtime.model_copy(
                update={"raw_data_root": tmp_path / "raw", "output_root": tmp_path / "outputs"}
            )
        }
    )
    source = tmp_path / "raw" / "N-BaIoT" / "device" / "benign_traffic.csv"
    source.parent.mkdir(parents=True)
    source.write_text("original", encoding="utf-8")
    loaded = campaign_devices()
    monkeypatch.setattr(runner, "dataset_source_files", Mock(return_value=(source,)))
    monkeypatch.setattr(runner, "dataset_source_files", Mock(return_value=(source,)))

    def change_source(*_arguments: object, **_keywords: object) -> tuple[TargetPerformance, ...]:
        source.write_text("changed", encoding="utf-8")
        return ()

    monkeypatch.setattr(runner, "run_dataset_study", change_source)
    with pytest.raises(DatasetValidationError, match="changed during campaign execution"):
        run_dataset_workflow(loaded, (source,), configuration)


def test_registered_matrix_workflow_resolves_clients_then_delegates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = campaign_configuration()
    configuration = base.model_copy(
        update={
            "runtime": base.runtime.model_copy(
                update={"raw_data_root": tmp_path / "raw", "output_root": tmp_path / "outputs"}
            )
        }
    )
    source = tmp_path / "raw" / "N-BaIoT" / "device" / "benign_traffic.csv"
    source.parent.mkdir(parents=True)
    source.write_text("stable", encoding="utf-8")
    loaded = Mock()
    loaded.clients = campaign_devices()
    loader = Mock(return_value=loaded)
    monkeypatch.setattr(runner, "dataset_source_files", Mock(return_value=(source,)))
    monkeypatch.setattr(runner, "load_dataset", loader)
    delegate = Mock(return_value=())
    monkeypatch.setattr(runner, "run_dataset_matrix_workflow", delegate)

    assert run_registered_dataset_matrix_workflow(DatasetName.N_BAIOT, configuration) == ()
    assert run_registered_dataset_matrix_workflow(DatasetName.N_BAIOT, configuration) == ()

    assert delegate.call_count == 2
    assert loader.call_count == 1
