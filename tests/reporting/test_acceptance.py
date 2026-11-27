import dataclasses
from unittest.mock import Mock

import numpy as np
import pytest

from pace.config import ResolvedConfiguration
from pace.experiment.preflight import matrix_variants
from pace.reporting.acceptance import (
    acceptance_line,
    acceptance_table,
    false_alert_margin_sensitivity_table,
    summarize_acceptance,
)
from pace.reporting.figures import budget_scaling, headline_gain
from pace.reporting.fixed_fpr import (
    TargetLevel,
    cell_results,
    direction_decomposition_table,
    fixed_fpr_comparison_table,
    level_targets,
    target_level,
)
from pace.reporting.inference import (
    cell_inference,
    coverage_diagnosis_table,
    inference_sensitivity_table,
)
from pace.reporting.metrics import summarize_campaign
from pace.reporting.records import (
    RecordStore,
    ReportInputs,
    StratumBudgetEvidence,
    StratumTraining,
    VariantMetrics,
    VariantPerformances,
    records_of,
)
from pace.reporting.scaling import (
    ScalingPoint,
    common_targets,
    distinct_cohorts,
    scaling_points,
    scaling_strata_table,
    scaling_table,
)
from pace.types import (
    AcceptanceCalibrationRule,
    AcceptanceCriterion,
    AlphaLevel,
    BudgetAnalysis,
    BudgetAnalysisKind,
    ClientName,
    ColumnHeading,
    CoverageCause,
    DatasetClientId,
    DatasetName,
    FprLevel,
    InferenceMethod,
    MarginSensitivity,
    OperatingCurve,
    OperatingMethod,
    OperatingPoint,
    ScalingCohort,
    SeedRateDifference,
    StudyArm,
    StudyStratum,
    TableName,
    TargetPerformance,
    TargetTrainingRows,
)
from tests.support import (
    campaign_configuration,
    campaign_devices,
    run_single_seed_campaign,
    with_attributes,
)

ROWS = 32
BUDGETS = (0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.2)
BASE_DETECTION = (0.6, 0.7, 0.8, 0.9, 0.95, 0.97, 0.98, 0.99)


def _curve(
    method: OperatingMethod,
    attacks: int,
    shift: float,
    inflation: float,
) -> OperatingCurve:
    return OperatingCurve(
        method=method,
        points=tuple(
            OperatingPoint(
                nominal_budget=budget,
                false_alert_rate=min(1.0, budget * inflation),
                detection_rates=(min(1.0, max(0.0, detection + shift)),) * attacks,
            )
            for budget, detection in zip(BUDGETS, BASE_DETECTION, strict=True)
        ),
    )


def _crafted(
    results: tuple[TargetPerformance, ...],
    gain: float,
    pace_inflation: float,
    reference_inflation: float = 1.0,
) -> tuple[TargetPerformance, ...]:
    return tuple(
        with_attributes(
            result,
            benign_test_rows=100_000,
            curves=(
                _curve(OperatingMethod.PACE, len(result.attack_types), gain, pace_inflation),
                _curve(
                    OperatingMethod.MEAN_ENSEMBLE_MATCHED,
                    len(result.attack_types),
                    0.0,
                    reference_inflation,
                ),
            ),
        )
        for result in results
    )


def _inputs(
    configuration: ResolvedConfiguration, outcomes: tuple[TargetPerformance, ...]
) -> ReportInputs:
    alpha = configuration.scientific.alpha
    stratum = StudyStratum.N_BAIOT
    names = tuple(sorted({result.target.name for result in outcomes}))
    analysis = BudgetAnalysis(BudgetAnalysisKind.UNIVERSAL_PRIMARY, (ROWS,), names)
    metrics = summarize_campaign(outcomes, alpha, configuration.reporting)
    evidence = StratumBudgetEvidence(
        stratum,
        (analysis,),
        len(names),
        tuple(
            VariantMetrics(variant, metrics) for variant in matrix_variants((analysis,), (alpha,))
        ),
    )
    base = ReportInputs(
        configuration,
        RecordStore(records_of(outcomes, alpha, ROWS)),
        (evidence,),
        (StratumTraining(stratum, tuple(TargetTrainingRows(name, 500, 3) for name in names)),),
        (VariantPerformances(stratum, alpha, ROWS, outcomes),),
    )
    return dataclasses.replace(base, acceptance=summarize_acceptance(base))


@pytest.fixture(scope="module")
def configuration() -> ResolvedConfiguration:
    base = campaign_configuration()
    alpha = AlphaLevel.ALPHA_001
    scientific = base.scientific.model_copy(update={"alpha": alpha, "study_arms": tuple(StudyArm)})
    return base.model_copy(update={"scientific": scientific})


@pytest.fixture(scope="module")
def outcomes(configuration: ResolvedConfiguration) -> tuple[TargetPerformance, ...]:
    return tuple(
        result
        for seed in (1, 2)
        for result in run_single_seed_campaign(campaign_devices(), configuration, seed)
    )


def test_both_curves_are_read_at_the_same_fixed_false_alert_rate(
    outcomes: tuple[TargetPerformance, ...],
) -> None:
    crafted = _crafted(outcomes, 0.05, 1.0)
    grouped = [item for item in crafted if item.target == crafted[0].target]
    reading = target_level(
        grouped,
        OperatingMethod.PACE,
        OperatingMethod.MEAN_ENSEMBLE_MATCHED,
        FprLevel.ALPHA,
        AlphaLevel.ALPHA_001,
    )
    assert reading is not None
    assert np.isclose(reading.gain, 0.05)
    assert np.isclose(reading.subject_detection - reading.reference_detection, 0.05)
    assert len(reading.seed_gains) == len(grouped)


def test_a_target_whose_comparator_curve_does_not_reach_the_level_stays_unsupported(
    outcomes: tuple[TargetPerformance, ...],
) -> None:
    crafted = _crafted(outcomes, 0.05, 1.0, reference_inflation=50.0)
    grouped = [item for item in crafted if item.target == crafted[0].target]
    reading = target_level(
        grouped,
        OperatingMethod.PACE,
        OperatingMethod.MEAN_ENSEMBLE_MATCHED,
        FprLevel.ALPHA,
        AlphaLevel.ALPHA_001,
    )
    assert reading is None


def test_coverage_and_gain_are_reported_for_every_level(
    configuration: ResolvedConfiguration, outcomes: tuple[TargetPerformance, ...]
) -> None:
    inputs = _inputs(configuration, _crafted(outcomes, 0.05, 1.0))
    cell = cell_results(inputs)[0]
    for level in FprLevel:
        readings = level_targets(
            cell, OperatingMethod.PACE, OperatingMethod.MEAN_ENSEMBLE_MATCHED, level
        )
        assert len(readings) == len(cell.targets)
        assert all(np.isclose(item.gain, 0.05) for item in readings)


def test_headline_uses_official_fixed_fpr_gain_and_interval(
    configuration: ResolvedConfiguration,
    outcomes: tuple[TargetPerformance, ...],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _inputs(configuration, _crafted(outcomes, 0.05, 1.0))
    assert inputs.acceptance is not None
    cell = inputs.acceptance.cells[0]
    level = next(item for item in cell.levels if item.level is FprLevel.ALPHA)
    assert level.gain is not None
    assert level.interval is not None
    figure = Mock()
    axes = figure.add_subplot.return_value
    monkeypatch.setattr("pace.reporting.figures.new_figure", Mock(return_value=figure))
    headline_gain(inputs)
    args, keywords = axes.errorbar.call_args
    assert np.isclose(args[0], level.gain)
    assert np.allclose(
        keywords["xerr"],
        [[level.gain - level.interval.lower_bound], [level.interval.upper_bound - level.gain]],
    )
    labels = axes.set_yticks.call_args.args[1]
    assert f"{level.matched_targets}/{cell.target_count} supported" in labels[0]
    assert "same-calibration" in axes.set_title.call_args.args[0]


def test_acceptance_reads_only_the_same_calibration_comparator_never_the_block_max_curve(
    configuration: ResolvedConfiguration, outcomes: tuple[TargetPerformance, ...]
) -> None:
    crafted = tuple(
        with_attributes(
            result,
            curves=(
                *result.curves,
                _curve(OperatingMethod.MEAN_ENSEMBLE, len(result.attack_types), 0.4, 1.0),
            ),
        )
        for result in _crafted(outcomes, 0.05, 1.0)
    )
    report = _inputs(configuration, crafted).acceptance
    assert report is not None
    assert report.comparator is OperatingMethod.MEAN_ENSEMBLE_MATCHED
    primary = next(item for item in report.cells[0].levels if item.level is FprLevel.ALPHA)
    assert primary.gain is not None
    assert np.isclose(primary.gain, 0.05)
    assert report.is_accepted


def test_nominal_records_compare_against_the_block_max_arm_not_the_acceptance_comparator(
    configuration: ResolvedConfiguration, outcomes: tuple[TargetPerformance, ...]
) -> None:
    inputs = _inputs(configuration, outcomes)
    first = inputs.store.records[0]
    arm_rates = [
        np.mean(
            next(
                item for item in result.arms if item.arm is StudyArm.MEAN_ENSEMBLE_LOCAL_BRANCH
            ).detection_rates
        ).item()
        for result in outcomes
        if result.target == first.target
    ]
    assert np.isclose(first.mean_ensemble.detection, np.mean(arm_rates))


def test_a_genuine_matched_gain_with_equal_false_alerts_is_accepted(
    configuration: ResolvedConfiguration, outcomes: tuple[TargetPerformance, ...]
) -> None:
    report = _inputs(configuration, _crafted(outcomes, 0.05, 1.0)).acceptance
    assert report is not None
    cell = report.cells[0]
    assert cell.is_accepted
    assert report.is_accepted
    assert all(verdict.satisfied for verdict in cell.verdicts)
    assert "accepted" in acceptance_line(cell)


def test_extra_false_alerts_fail_the_safeguard_even_when_the_matched_gain_holds(
    configuration: ResolvedConfiguration, outcomes: tuple[TargetPerformance, ...]
) -> None:
    report = _inputs(configuration, _crafted(outcomes, 0.05, 2.5)).acceptance
    assert report is not None
    verdicts = {item.criterion: item.satisfied for item in report.cells[0].verdicts}
    assert not verdicts[AcceptanceCriterion.FALSE_ALERT_SAFEGUARD]
    assert not report.is_accepted


def test_a_missing_gain_fails_the_primary_criterion(
    configuration: ResolvedConfiguration, outcomes: tuple[TargetPerformance, ...]
) -> None:
    report = _inputs(configuration, _crafted(outcomes, 0.0, 1.0)).acceptance
    assert report is not None
    verdicts = {item.criterion: item.satisfied for item in report.cells[0].verdicts}
    assert not verdicts[AcceptanceCriterion.FIXED_FPR_GAIN]
    assert not verdicts[AcceptanceCriterion.SEED_SIGNS]


def test_unsupported_targets_are_counted_and_reduce_coverage(
    configuration: ResolvedConfiguration, outcomes: tuple[TargetPerformance, ...]
) -> None:
    report = _inputs(
        configuration, _crafted(outcomes, 0.05, 1.0, reference_inflation=50.0)
    ).acceptance
    assert report is not None
    cell = report.cells[0]
    primary = next(item for item in cell.levels if item.level is FprLevel.ALPHA)
    assert primary.matched_targets == 0
    verdicts = {item.criterion: item.satisfied for item in cell.verdicts}
    assert not verdicts[AcceptanceCriterion.COVERAGE]
    assert not verdicts[AcceptanceCriterion.FIXED_FPR_GAIN]


def test_margin_sensitivity_reports_every_margin_and_matches_the_declared_one() -> None:
    assert (
        MarginSensitivity.STANDARD.share
        == AcceptanceCalibrationRule.FALSE_ALERT_MARGIN_SHARE.threshold
    )
    assert MarginSensitivity.STRICT.share < MarginSensitivity.STANDARD.share
    assert MarginSensitivity.STANDARD.share < MarginSensitivity.LENIENT.share


def test_acceptance_tables_have_one_row_per_check_and_margin(
    configuration: ResolvedConfiguration, outcomes: tuple[TargetPerformance, ...]
) -> None:
    inputs = _inputs(configuration, _crafted(outcomes, 0.05, 1.0))
    table = acceptance_table(inputs)
    assert len(table.rows) == 10
    assert all(len(row) == len(table.columns) for row in table.rows)
    margins = false_alert_margin_sensitivity_table(inputs)
    assert len(margins.rows) == len(MarginSensitivity)


def test_direction_decomposition_matches_every_target_when_the_operating_ranges_overlap(
    configuration: ResolvedConfiguration, outcomes: tuple[TargetPerformance, ...]
) -> None:
    inputs = _inputs(configuration, _crafted(outcomes, 0.05, 1.0))
    table = direction_decomposition_table(inputs)
    assert len(table.rows) == 1
    row = dict(zip(table.columns, table.rows[0], strict=True))
    targets = str(len({item.target for item in outcomes}))
    assert row[ColumnHeading.TARGETS] == targets
    assert row[ColumnHeading.FORWARD_TARGETS] == targets
    assert row[ColumnHeading.REVERSE_TARGETS] == targets
    assert row[ColumnHeading.COMMON_TARGETS] == targets
    assert np.isclose(float(row[ColumnHeading.FORWARD_COMMON_GAIN]), 0.05)
    assert np.isclose(float(row[ColumnHeading.REVERSE_COMMON_GAIN]), 0.05)


def test_direction_decomposition_shows_that_a_higher_pace_operating_range_leaves_targets_unmatched(
    configuration: ResolvedConfiguration, outcomes: tuple[TargetPerformance, ...]
) -> None:
    inputs = _inputs(configuration, _crafted(outcomes, 0.05, 15.0))
    table = direction_decomposition_table(inputs)
    row = dict(zip(table.columns, table.rows[0], strict=True))
    assert row[ColumnHeading.FORWARD_TARGETS] == "0"
    assert int(row[ColumnHeading.REVERSE_TARGETS]) > 0


def test_fixed_fpr_table_reports_one_row_per_level_with_coverage(
    configuration: ResolvedConfiguration, outcomes: tuple[TargetPerformance, ...]
) -> None:
    inputs = _inputs(configuration, _crafted(outcomes, 0.05, 1.0))
    table = fixed_fpr_comparison_table(inputs)
    assert len(table.rows) == len(FprLevel)
    coverage = {row[table.columns.index(ColumnHeading.COVERAGE)] for row in table.rows}
    assert coverage == {"1.000000"}


def test_inference_sensitivity_reports_every_estimator_for_a_supported_cell(
    configuration: ResolvedConfiguration, outcomes: tuple[TargetPerformance, ...]
) -> None:
    inputs = _inputs(configuration, _crafted(outcomes, 0.05, 1.0))
    table = inference_sensitivity_table(inputs)
    assert table.name is TableName.INFERENCE_SENSITIVITY
    assert all(len(row) == len(table.columns) for row in table.rows)
    methods = {row[table.columns.index(ColumnHeading.INFERENCE_METHOD)] for row in table.rows}
    assert InferenceMethod.STRATIFIED_TARGETS.display_label in methods
    assert InferenceMethod.DATASET_CLUSTERS.display_label in methods
    assert InferenceMethod.UNSUPPORTED_AS_ZERO.display_label in methods
    estimate_column = table.columns.index(ColumnHeading.ESTIMATE)
    pooled = [
        float(row[estimate_column])
        for row in table.rows
        if row[table.columns.index(ColumnHeading.INFERENCE_METHOD)]
        == InferenceMethod.STRATIFIED_TARGETS.display_label
    ]
    assert len(pooled) == 1
    assert np.isclose(pooled[0], 0.05)


def test_unsupported_targets_pull_the_zero_filled_estimate_towards_zero(
    configuration: ResolvedConfiguration, outcomes: tuple[TargetPerformance, ...]
) -> None:
    inputs = _inputs(configuration, _crafted(outcomes, 0.05, 1.0))
    cell = cell_results(inputs)[0]
    readings = cell_inference(
        cell, OperatingMethod.MEAN_ENSEMBLE_MATCHED, inputs.configuration.reporting
    )
    zero_filled = next(
        item for item in readings if item.method is InferenceMethod.UNSUPPORTED_AS_ZERO
    )
    assert np.isclose(zero_filled.estimate, 0.05)
    assert zero_filled.target_count == len(cell.targets)


def test_coverage_diagnosis_counts_every_target_exactly_once(
    configuration: ResolvedConfiguration, outcomes: tuple[TargetPerformance, ...]
) -> None:
    supported = coverage_diagnosis_table(_inputs(configuration, _crafted(outcomes, 0.05, 1.0)))
    unsupported = coverage_diagnosis_table(
        _inputs(configuration, _crafted(outcomes, 0.05, 1.0, reference_inflation=50.0))
    )
    counts = supported.columns.index(ColumnHeading.TARGETS)
    targets = len({item.target for item in outcomes})
    assert sum(int(row[counts]) for row in supported.rows) == targets
    assert sum(int(row[counts]) for row in unsupported.rows) == targets
    by_cause = {row[2]: int(row[counts]) for row in unsupported.rows}
    assert by_cause[CoverageCause.SUPPORTED.display_label] == 0
    assert by_cause[CoverageCause.GRID_CEILING.display_label] == 0
    assert (
        by_cause[CoverageCause.SHIFT_FLOOR.display_label]
        + by_cause[CoverageCause.RESOLUTION_LIMITED.display_label]
        == targets
    )
    assert {row[2]: int(row[counts]) for row in supported.rows}[
        CoverageCause.SUPPORTED.display_label
    ] == targets


def test_scaling_points_use_matched_gains_with_available_and_common_cohorts(
    configuration: ResolvedConfiguration, outcomes: tuple[TargetPerformance, ...]
) -> None:
    points = scaling_points(_inputs(configuration, _crafted(outcomes, 0.05, 1.0)))
    available = next(point for point in points if point.cohort is ScalingCohort.AVAILABLE)
    common = next(point for point in points if point.cohort is ScalingCohort.COMMON)
    assert available.gain is not None
    assert np.isclose(available.gain, 0.05)
    assert available.supported == common.supported == available.target_count
    assert available.interval is not None
    assert available.harmed == 0
    assert available.pace_detection is not None
    assert available.comparator_detection is not None
    assert np.isclose(available.pace_detection - available.comparator_detection, 0.05)
    assert available.pace_false_alert is not None
    assert available.comparator_false_alert is not None
    assert sum(item.targets for item in available.strata) == available.supported
    assert all(np.isclose(item.difference, 0.05) for item in available.seed_gains)
    assert available.gain_without_excluded is not None


def test_scaling_tables_and_figure_expose_matched_fpr_evidence(
    configuration: ResolvedConfiguration, outcomes: tuple[TargetPerformance, ...]
) -> None:
    inputs = _inputs(configuration, _crafted(outcomes, 0.05, 1.0))
    table = scaling_table(inputs)
    assert ColumnHeading.SMALLEST_SEED_GAIN in table.columns
    assert len(table.rows) == len(scaling_points(inputs))
    strata = scaling_strata_table(inputs)
    assert len(strata.rows) == sum(len(point.strata) for point in scaling_points(inputs))
    figure = budget_scaling(inputs)
    assert len(figure.axes) == len(inputs.alphas())
    styles = {
        (line.get_linestyle(), line.get_marker())
        for axes in figure.axes
        for line in axes.get_lines()
    }
    assert ("-", "o") in styles


def _level(name: str) -> TargetLevel:
    return TargetLevel(
        DatasetClientId(DatasetName.TON_IOT, ClientName(name)),
        StudyStratum.TON_IOT,
        (SeedRateDifference(1, 0.1),),
        0.9,
        0.8,
    )


def test_a_target_unsupported_at_one_budget_leaves_the_common_cohort() -> None:
    first, second = _level("a"), _level("b")
    common = common_targets(
        {100: {first.target: first, second.target: second}, 400: {first.target: first}}
    )
    assert common == frozenset({first.target})
    assert common_targets({}) == frozenset()


def _point(cohort: ScalingCohort, *readings: TargetLevel) -> ScalingPoint:
    return ScalingPoint(
        BudgetAnalysisKind.UNIVERSAL_PRIMARY,
        AlphaLevel.ALPHA_001,
        100,
        cohort,
        2,
        readings,
        None,
        None,
        None,
    )


def test_cohort_rows_merge_only_when_every_point_agrees() -> None:
    first, second = _level("a"), _level("b")
    same = (
        _point(ScalingCohort.AVAILABLE, first, second),
        _point(ScalingCohort.COMMON, first, second),
    )
    assert distinct_cohorts(same) == (ScalingCohort.AVAILABLE,)
    different = (
        _point(ScalingCohort.AVAILABLE, first, second),
        _point(ScalingCohort.COMMON, first),
    )
    assert distinct_cohorts(different) == tuple(ScalingCohort)
