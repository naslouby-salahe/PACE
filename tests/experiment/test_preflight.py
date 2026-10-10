from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import pytest

import pace.experiment.preflight as preflight
from pace.config import (
    AnalysisSpec,
    ProtocolMatrixSettings,
    ResolvedConfiguration,
)
from pace.experiment.preflight import (
    MatrixVariant,
    assess_target_support,
    budget_plan_path,
    matrix_variants,
    persist_budget_plan,
    plan_budget_analyses,
    plan_stratum_budgets,
    preflight_registered_dataset,
    read_budget_plan_summary,
)
from pace.types import (
    AlphaLevel,
    ArtifactValidationError,
    BudgetAnalysis,
    BudgetAnalysisKind,
    ClientName,
    DatasetClientData,
    DatasetClientId,
    DatasetName,
    DatasetValidationError,
    ObservationCount,
    StratumBudgetPlan,
    StudyStratum,
    TargetBudgetSupport,
    TargetCohortRole,
    TargetEligibility,
    TargetEligibilityIssue,
)
from tests.support import (
    campaign_configuration,
    campaign_devices,
    protocol_matrix,
)

BUDGETS = (100, 400, 750, 2000, 20000)
DIGEST = "3" * 64


SPECS = (
    AnalysisSpec(kind=BudgetAnalysisKind.UNIVERSAL_PRIMARY, budgets=(100, 400)),
    AnalysisSpec(kind=BudgetAnalysisKind.UNIVERSAL_ROBUSTNESS, budgets=(100, 400, 750)),
    AnalysisSpec(kind=BudgetAnalysisKind.BROAD_SCALING, budgets=(100, 400, 2000)),
    AnalysisSpec(kind=BudgetAnalysisKind.HIGH_RESOURCE, budgets=(100, 400, 2000, 20000)),
)


def _protocol() -> ProtocolMatrixSettings:
    return protocol_matrix(BUDGETS, analyses=SPECS)


def _support(
    name: str,
    training_rows: ObservationCount,
    issues: tuple[TargetEligibilityIssue, ...] = (),
) -> TargetBudgetSupport:
    return TargetBudgetSupport(
        DatasetClientId(DatasetName.TON_IOT, ClientName(name)),
        benign_rows=training_rows * 5 // 2,
        training_rows=training_rows,
        calibration_rows=training_rows // 2,
        attack_type_count=3,
        eligibility=TargetEligibility(issues),
    )


def _cohort(*training_rows: ObservationCount) -> tuple[TargetBudgetSupport, ...]:
    return tuple(_support(f"target-{index}", rows) for index, rows in enumerate(training_rows))


def _names(*indices: int) -> tuple[ClientName, ...]:
    return tuple(ClientName(f"target-{index}") for index in indices)


def test_support_is_exact_and_never_clamped() -> None:
    target = _support("t", 20000)
    assert target.is_supported_at(20000)
    assert not _support("t", 19999).is_supported_at(20000)
    assert _support("t", 19999).is_supported_at(400)
    ineligible = _support("t", 50000, (TargetEligibilityIssue.INSUFFICIENT_PEERS,))
    assert not ineligible.is_supported_at(100)


def _cohorts(plan: StratumBudgetPlan) -> dict[BudgetAnalysisKind, tuple[ClientName, ...]]:
    return {analysis.kind: analysis.targets for analysis in plan.analyses}


def test_each_analysis_cohort_is_the_targets_supporting_its_largest_budget() -> None:
    plan = plan_budget_analyses(
        StudyStratum.TON_IOT, _cohort(80000, 30000, 2500, 900, 800), _protocol()
    )

    assert _cohorts(plan) == {
        BudgetAnalysisKind.UNIVERSAL_PRIMARY: _names(0, 1, 2, 3, 4),
        BudgetAnalysisKind.UNIVERSAL_ROBUSTNESS: _names(0, 1, 2, 3, 4),
        BudgetAnalysisKind.BROAD_SCALING: _names(0, 1, 2),
        BudgetAnalysisKind.HIGH_RESOURCE: _names(0, 1),
    }
    assert [analysis.budgets for analysis in plan.analyses] == [
        (100, 400),
        (100, 400, 750),
        (100, 400, 2000),
        (100, 400, 2000, 20000),
    ]
    assert plan.unavailable_budgets == (2000, 20000)
    assert plan.unobserved_budgets == ()
    assert plan.supported_count(20000) == 2


def test_support_is_exact_at_the_frozen_budget_boundaries() -> None:
    plan = plan_budget_analyses(
        StudyStratum.TON_IOT, _cohort(2000, 1999, 769, 750, 749, 20000, 19999), _protocol()
    )

    cohorts = _cohorts(plan)
    assert cohorts[BudgetAnalysisKind.UNIVERSAL_PRIMARY] == _names(*range(7))
    assert cohorts[BudgetAnalysisKind.UNIVERSAL_ROBUSTNESS] == _names(0, 1, 2, 3, 5, 6)
    assert cohorts[BudgetAnalysisKind.BROAD_SCALING] == _names(0, 5, 6)
    assert cohorts[BudgetAnalysisKind.HIGH_RESOURCE] == _names(5)


def test_a_stratum_where_every_target_is_large_runs_every_analysis_on_everyone() -> None:
    plan = plan_budget_analyses(
        StudyStratum.TON_IOT, _cohort(28000, 28001, 28002, 28003, 28004), _protocol()
    )

    assert set(_cohorts(plan).values()) == {_names(0, 1, 2, 3, 4)}
    assert plan.unavailable_budgets == ()
    assert plan.unobserved_budgets == ()


def test_an_analysis_nobody_supports_is_omitted_and_its_budget_is_unobserved() -> None:
    plan = plan_budget_analyses(
        StudyStratum.TON_IOT, _cohort(17371, 17455, 17390, 17482, 17431), _protocol()
    )

    assert BudgetAnalysisKind.HIGH_RESOURCE not in _cohorts(plan)
    assert _cohorts(plan)[BudgetAnalysisKind.BROAD_SCALING] == _names(0, 1, 2, 3, 4)
    assert plan.unavailable_budgets == (20000,)
    assert plan.unobserved_budgets == (20000,)


def test_a_single_supporting_target_is_kept_without_any_threshold() -> None:
    plan = plan_budget_analyses(
        StudyStratum.TON_IOT, _cohort(27363, 12688, 8573, 6480, 2695), _protocol()
    )

    assert _cohorts(plan)[BudgetAnalysisKind.HIGH_RESOURCE] == _names(0)
    assert _cohorts(plan)[BudgetAnalysisKind.BROAD_SCALING] == _names(0, 1, 2, 3, 4)


def test_ineligible_targets_never_enter_a_cohort_or_the_denominator() -> None:
    cohort = (
        *_cohort(80000, 30000, 2500, 900, 800),
        _support("weak", 90000, (TargetEligibilityIssue.NO_ATTACK_TYPES,)),
    )
    plan = plan_budget_analyses(StudyStratum.TON_IOT, cohort, _protocol())

    assert all(ClientName("weak") not in analysis.targets for analysis in plan.analyses)
    assert len(plan.eligible) == 5
    assert plan.unavailable_budgets == (2000, 20000)


def test_planning_fails_loudly_without_eligible_targets_or_any_supported_cohort() -> None:
    with pytest.raises(DatasetValidationError, match="No target satisfies"):
        plan_budget_analyses(
            StudyStratum.TON_IOT,
            (_support("weak", 10, (TargetEligibilityIssue.INSUFFICIENT_PEERS,)),),
            _protocol(),
        )
    with pytest.raises(DatasetValidationError, match="supported target cohort"):
        plan_budget_analyses(StudyStratum.TON_IOT, _cohort(50, 60, 150), _protocol())


def test_selection_never_depends_on_anything_but_row_support() -> None:
    rows = (80000, 30000, 2500, 900, 800, 2000)
    first = plan_budget_analyses(StudyStratum.TON_IOT, _cohort(*rows), _protocol())
    reordered = plan_budget_analyses(
        StudyStratum.TON_IOT, tuple(reversed(_cohort(*rows))), _protocol()
    )

    assert {kind: set(targets) for kind, targets in _cohorts(first).items()} == {
        kind: set(targets) for kind, targets in _cohorts(reordered).items()
    }
    assert [analysis.budgets for analysis in first.analyses] == [
        analysis.budgets for analysis in reordered.analyses
    ]


def test_matrix_variants_run_each_budget_once_per_frozen_cohort() -> None:
    plan = plan_budget_analyses(
        StudyStratum.TON_IOT, _cohort(80000, 30000, 2500, 900, 800), _protocol()
    )

    variants = matrix_variants(plan.analyses, (AlphaLevel.ALPHA_001, AlphaLevel.ALPHA_005))

    assert len(variants) == 2 * (3 + 3 + 4)
    assert variants[0] == MatrixVariant(
        BudgetAnalysisKind.UNIVERSAL_PRIMARY, AlphaLevel.ALPHA_001, 100, _names(0, 1, 2, 3, 4)
    )
    by_cohort = {
        cohort: {v.local_training_rows for v in variants if v.targets == cohort}
        for cohort in {v.targets for v in variants}
    }
    assert by_cohort == {
        _names(0, 1, 2, 3, 4): {100, 400, 750},
        _names(0, 1, 2): {100, 400, 2000},
        _names(0, 1): {100, 400, 2000, 20000},
    }
    assert len({(v.alpha, v.local_training_rows, v.targets) for v in variants}) == len(variants)


def test_matrix_variants_do_not_repeat_cells_when_every_cohort_coincides() -> None:
    plan = plan_budget_analyses(StudyStratum.TON_IOT, _cohort(28000, 28001, 28002), _protocol())

    variants = matrix_variants(plan.analyses, (AlphaLevel.ALPHA_001,))

    assert [v.local_training_rows for v in variants] == [100, 400, 750, 2000, 20000]
    assert {v.targets for v in variants} == {_names(0, 1, 2)}
    assert variants[0].kind is BudgetAnalysisKind.UNIVERSAL_PRIMARY


def test_analysis_rejects_unsorted_budgets_and_duplicate_targets() -> None:
    with pytest.raises(ValueError, match="unique, increasing"):
        BudgetAnalysis(BudgetAnalysisKind.UNIVERSAL_PRIMARY, (400, 100), _names(0))
    with pytest.raises(ValueError, match="unique and non-empty"):
        BudgetAnalysis(BudgetAnalysisKind.UNIVERSAL_PRIMARY, (100,), _names(0, 0))


def _configuration(tmp_path: Path, protocol: ProtocolMatrixSettings) -> ResolvedConfiguration:
    base = campaign_configuration()
    return base.model_copy(
        update={
            "protocol_matrix": protocol,
            "runtime": base.runtime.model_copy(
                update={"raw_data_root": tmp_path / "raw", "output_root": tmp_path / "outputs"}
            ),
        }
    )


def test_target_support_reports_rows_after_the_existing_split_rules() -> None:
    configuration = campaign_configuration()

    support = assess_target_support(
        campaign_devices(),
        configuration.splitting,
        configuration.scientific.minimum_peer_count,
    )

    assert len(support) == 5
    assert all(item.benign_rows == 120 for item in support)
    assert all(item.training_rows == 48 for item in support)
    assert all(item.calibration_rows == 24 for item in support)
    assert all(item.attack_type_count == 2 for item in support)
    assert all(item.eligibility.is_eligible for item in support)
    assert all(item.is_supported_at(32) and not item.is_supported_at(49) for item in support)


def test_target_support_excludes_peer_only_and_outside_pool_clients() -> None:
    configuration = campaign_configuration()
    clients = list(campaign_devices())
    clients[0] = replace(clients[0], target_cohort_role=TargetCohortRole.PEER_ONLY)
    clients[1] = replace(clients[1], target_cohort_role=TargetCohortRole.OUTSIDE_PEER_POOL)

    support = assess_target_support(
        tuple(clients), configuration.splitting, configuration.scientific.minimum_peer_count
    )

    assert {item.client.name for item in support} == {"client-2", "client-3", "client-4"}
    assert all(
        TargetEligibilityIssue.INSUFFICIENT_PEERS in item.eligibility.issues for item in support
    )


def test_stratum_plan_uses_the_configured_protocol(tmp_path: Path) -> None:
    specs = (
        AnalysisSpec(kind=BudgetAnalysisKind.UNIVERSAL_PRIMARY, budgets=(16, 32)),
        AnalysisSpec(kind=BudgetAnalysisKind.HIGH_RESOURCE, budgets=(16, 32, 64)),
    )
    configuration = _configuration(tmp_path, protocol_matrix((16, 32, 64), analyses=specs))

    plan = plan_stratum_budgets(campaign_devices(), configuration)

    assert plan.unavailable_budgets == (64,)
    assert plan.unobserved_budgets == (64,)
    assert [analysis.kind for analysis in plan.analyses] == [BudgetAnalysisKind.UNIVERSAL_PRIMARY]
    assert plan.analyses[0].targets == tuple(ClientName(f"client-{i}") for i in range(5))


def _persist(configuration: ResolvedConfiguration, plan: StratumBudgetPlan) -> None:
    persist_budget_plan(configuration.runtime.output_root, plan, DIGEST)
    assert budget_plan_path(configuration.runtime.output_root, plan.stratum).exists()


def test_persisted_plan_round_trips_and_detects_stale_inputs(tmp_path: Path) -> None:
    configuration = _configuration(tmp_path, protocol_matrix((16, 32)))
    plan = plan_stratum_budgets(campaign_devices(), configuration)
    _persist(configuration, plan)

    summary = read_budget_plan_summary(configuration, StudyStratum.N_BAIOT, DIGEST)
    assert summary.analyses == plan.analyses
    assert summary.eligible_targets == len(plan.eligible)

    with pytest.raises(ArtifactValidationError, match="stale"):
        read_budget_plan_summary(configuration, StudyStratum.N_BAIOT, "4" * 64)
    relaxed = configuration.model_copy(
        update={"runtime": configuration.runtime.model_copy(update={"provenance_active": False})}
    )
    assert read_budget_plan_summary(relaxed, StudyStratum.N_BAIOT, "4" * 64).analyses == (
        plan.analyses
    )
    budgets_changed = configuration.model_copy(
        update={"protocol_matrix": protocol_matrix((16, 33))}
    )
    with pytest.raises(ArtifactValidationError, match="stale"):
        read_budget_plan_summary(budgets_changed, StudyStratum.N_BAIOT, DIGEST)


def test_reading_a_missing_or_corrupt_plan_is_an_artifact_error(tmp_path: Path) -> None:
    configuration = _configuration(tmp_path, protocol_matrix((16, 32)))
    with pytest.raises(ArtifactValidationError, match="missing or invalid"):
        read_budget_plan_summary(configuration, StudyStratum.N_BAIOT, DIGEST)
    path = budget_plan_path(configuration.runtime.output_root, StudyStratum.N_BAIOT)
    path.parent.mkdir(parents=True)
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ArtifactValidationError, match="missing or invalid"):
        read_budget_plan_summary(configuration, StudyStratum.N_BAIOT, DIGEST)


def test_registered_preflight_loads_once_persists_and_reads_the_client_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configuration = _configuration(tmp_path, protocol_matrix((16, 32)))
    source = tmp_path / "raw" / "N-BaIoT" / "device" / "benign_traffic.csv"
    source.parent.mkdir(parents=True)
    source.write_text("stable", encoding="utf-8")
    loaded = Mock()
    clients: tuple[DatasetClientData, ...] = campaign_devices()
    loaded.clients = clients
    loader = Mock(return_value=loaded)
    monkeypatch.setattr(preflight, "dataset_source_files", Mock(return_value=(source,)))
    monkeypatch.setattr(preflight, "load_dataset", loader)

    first = preflight_registered_dataset(DatasetName.N_BAIOT, configuration)
    second = preflight_registered_dataset(DatasetName.N_BAIOT, configuration)

    assert first == second
    assert loader.call_count == 1
    assert first.stratum is StudyStratum.N_BAIOT
    assert first.analyses[0].budgets == (16, 32)
    assert budget_plan_path(configuration.runtime.output_root, StudyStratum.N_BAIOT).exists()
    assert DatasetClientId(DatasetName.N_BAIOT, ClientName("client-0")) in tuple(
        item.client for item in first.candidates
    )
