from dataclasses import replace
from math import isclose, sqrt
from statistics import NormalDist

import pytest

from pace.config import ReportingSettings, ResolvedConfiguration
from pace.reporting.metrics import (
    matched_fpr_detection_rate,
    paired_minimum_detectable_effect,
    paired_power_at_effect,
    stratified_paired_interval,
    summarize_budget_power,
    summarize_campaign,
)
from pace.types import (
    AlphaLevel,
    ArmPerformance,
    AttackType,
    AttackTypeName,
    AttackTypePerformance,
    BudgetAnalysis,
    BudgetAnalysisKind,
    BudgetSweepPerformance,
    ClientName,
    DatasetClientId,
    DatasetName,
    StratumBudgetPlan,
    StudyArm,
    StudyStratum,
    TargetPairedDifference,
    TargetPerformance,
)
from tests.support import (
    campaign_results,
    protocol_matrix,
    target_provenance,
    two_seed_configuration,
    with_attributes,
)

PROTOCOL = protocol_matrix((100, 400)).power
SPEC_KINDS = (BudgetAnalysisKind.UNIVERSAL_PRIMARY, BudgetAnalysisKind.HIGH_RESOURCE)


def _target_result(
    target_name: str,
    seed: int,
    method_rates: tuple[float, float],
    local_rates: tuple[float, float],
    false_alert_rate: float,
) -> TargetPerformance:
    attack_types = (
        AttackType(AttackTypeName("GAFGYT/COMBO")),
        AttackType(AttackTypeName("MIRAI/ACK")),
    )
    performances = tuple(
        AttackTypePerformance(
            attack_type=attack_type,
            test_rows=10,
            detection_rate=method_rate,
            local_detection_rate=local_rate,
            max_fusion_detection_rate=max(0.0, method_rate - 0.1),
            peer_union_detection_rate=max(0.0, method_rate - 0.25),
            bonferroni_detection_rate=max(0.0, method_rate - 0.05),
        )
        for attack_type, method_rate, local_rate in zip(
            attack_types, method_rates, local_rates, strict=True
        )
    )
    target = DatasetClientId(DatasetName.N_BAIOT, ClientName(target_name))
    learning_seed = seed
    split_digest, peer_model_seeds, local_model_seed = target_provenance(target, learning_seed)
    return TargetPerformance(
        target=target,
        seed=learning_seed,
        split_digest=split_digest,
        peer_model_seeds=peer_model_seeds,
        local_model_seed=local_model_seed,
        benign_test_rows=20,
        realized_false_alert_rate=false_alert_rate,
        attack_types=performances,
        local_false_alert_rate=0.01,
        max_fusion_false_alert_rate=false_alert_rate * 0.8,
        peer_union_false_alert_rate=false_alert_rate * 0.5,
        bonferroni_false_alert_rate=false_alert_rate * 0.9,
        peer_count=4,
        budget_sweep=tuple(
            BudgetSweepPerformance(
                multiplier=multiplier,
                detection_rate=rate,
                false_alert_rate=fpr,
            )
            for multiplier, rate, fpr in (
                (0.4, 0.2, 0.0),
                (0.75, 0.4, 0.02),
                (1.0, 0.6, 0.05),
                (1.4, 0.8, 0.2),
            )
        ),
        study_stratum=StudyStratum.N_BAIOT,
    )


def _reporting_settings() -> ReportingSettings:
    return ReportingSettings(
        bootstrap_replicates=1000,
        bootstrap_seed=314159,
        confidence_level=0.95,
    )


def test_campaign_reporting_averages_attack_types_then_seeds_by_target() -> None:
    results = (
        _target_result("one", 0, (0.6, 0.4), (0.5, 0.3), 0.02),
        _target_result("one", 1, (0.8, 0.4), (0.6, 0.4), 0.02),
        _target_result("two", 0, (0.1, 0.3), (0.5, 0.5), 0.1),
        _target_result("two", 1, (0.1, 0.3), (0.5, 0.5), 0.1),
    )

    summary = summarize_campaign(results, AlphaLevel.ALPHA_005, _reporting_settings())

    assert summary.target_count == 2
    assert abs(summary.macro_detection_rate - 0.375) < 1e-12
    assert summary.local_macro_detection_rate == 0.475
    assert summary.max_fusion_macro_detection_rate == 0.275
    assert summary.bonferroni_macro_detection_rate == 0.325
    assert abs(summary.macro_detection_difference + 0.1) < 1e-12
    assert abs(summary.max_fusion_macro_detection_difference + 0.2) < 1e-12
    assert abs(summary.mean_method_max_fusion_difference - 0.1) < 1e-12
    assert abs(summary.worst_method_max_fusion_difference - 0.1) < 1e-12
    assert summary.improved_target_share == 1.0
    assert abs(summary.worst_detection_difference + 0.3) < 1e-12
    assert abs(summary.tenth_percentile_detection_difference + 0.26) < 1e-12
    assert summary.catastrophic_cell_count == 2
    assert summary.harmed_target_count == 1
    assert abs(summary.mean_false_alert_rate - 0.06) < 1e-12
    assert abs(summary.median_false_alert_rate - 0.06) < 1e-12
    assert abs(summary.ninetieth_percentile_false_alert_rate - 0.092) < 1e-12
    assert summary.worst_false_alert_rate == 0.1
    assert summary.exceedance_share == 0.5
    assert summary.local_mean_false_alert_rate == 0.01
    assert abs(summary.max_fusion_mean_false_alert_rate - 0.048) < 1e-12
    assert abs(summary.max_fusion_worst_false_alert_rate - 0.08) < 1e-12
    assert abs(summary.bonferroni_mean_false_alert_rate - 0.054) < 1e-12
    assert summary.targets[0].local_false_alert_rate == 0.01
    assert summary.method_local_confidence_interval.confidence_level == 0.95
    assert summary.matched_fpr_target_count == 2
    assert summary.matched_fpr_macro_difference is not None
    assert summary.matched_fpr_confidence_interval is not None
    assert (
        summary.method_local_confidence_interval.lower_bound
        <= summary.macro_detection_difference
        <= summary.method_local_confidence_interval.upper_bound
    )


def test_matched_fpr_interpolates_the_monotone_envelope_and_does_not_extrapolate() -> None:
    sweep = (
        BudgetSweepPerformance(multiplier=0.4, detection_rate=0.2, false_alert_rate=0.0),
        BudgetSweepPerformance(multiplier=0.75, detection_rate=0.6, false_alert_rate=0.04),
        BudgetSweepPerformance(multiplier=1.0, detection_rate=0.5, false_alert_rate=0.04),
        BudgetSweepPerformance(multiplier=1.4, detection_rate=0.8, false_alert_rate=0.2),
    )

    matched = matched_fpr_detection_rate(sweep, 0.02)
    outside = matched_fpr_detection_rate(sweep, 0.21)

    assert matched is not None
    assert abs(matched - 0.4) < 1e-12
    assert outside is None


def test_campaign_excludes_targets_outside_the_observed_matched_fpr_sweep() -> None:
    outside = replace(
        _target_result("outside", 0, (0.6, 0.4), (0.5, 0.3), 0.02),
        max_fusion_false_alert_rate=0.5,
    )

    summary = summarize_campaign((outside,), AlphaLevel.ALPHA_005, _reporting_settings())

    assert summary.target_count == 1
    assert summary.matched_fpr_target_count == 0
    assert summary.targets[0].matched_fpr_difference is None
    assert summary.matched_fpr_macro_difference is None
    assert summary.matched_fpr_confidence_interval is None


def test_campaign_reporting_rejects_inconsistent_attack_coverage() -> None:
    first = _target_result("one", 0, (0.6, 0.4), (0.5, 0.3), 0.02)
    mismatched = _target_result("one", 1, (0.6, 0.4), (0.5, 0.3), 0.02)
    changed_type = AttackTypePerformance(
        attack_type=AttackType(AttackTypeName("MIRAI/SYN")),
        test_rows=10,
        detection_rate=0.4,
        local_detection_rate=0.3,
        max_fusion_detection_rate=0.2,
        peer_union_detection_rate=0.15,
        bonferroni_detection_rate=0.35,
    )
    second = replace(
        mismatched,
        attack_types=(mismatched.attack_types[0], changed_type),
    )

    reporting_settings = _reporting_settings()
    with pytest.raises(ValueError, match="Attack type coverage"):
        summarize_campaign((first, second), AlphaLevel.ALPHA_005, reporting_settings)


def _settings() -> ReportingSettings:
    return ReportingSettings(
        bootstrap_replicates=2000,
        bootstrap_seed=41,
        confidence_level=0.95,
    )


def _difference(dataset: DatasetName, client: str, difference: float) -> TargetPairedDifference:
    return TargetPairedDifference(
        DatasetClientId(dataset, ClientName(client)),
        difference,
    )


def test_dataset_stratified_resampling_preserves_stratum_sizes() -> None:
    differences = (
        _difference(DatasetName.N_BAIOT, "target-a", 0.5),
        _difference(DatasetName.N_BAIOT, "target-b", 0.5),
        _difference(DatasetName.IOT_23, "target-c", -0.5),
    )

    interval = stratified_paired_interval(differences, _settings())

    expected = 1.0 / 6.0
    assert abs(interval.lower_bound - expected) < 1e-12
    assert abs(interval.upper_bound - expected) < 1e-12
    assert interval.confidence_level == 0.95


def test_bootstrap_is_deterministic_and_target_order_invariant() -> None:
    differences = (
        _difference(DatasetName.N_BAIOT, "target-b", 0.9),
        _difference(DatasetName.IOT_23, "target-c", -0.3),
        _difference(DatasetName.N_BAIOT, "target-a", 0.2),
    )

    first = stratified_paired_interval(differences, _settings())
    repeated = stratified_paired_interval(differences, _settings())
    reordered = stratified_paired_interval(tuple(reversed(differences)), _settings())

    assert first == repeated == reordered


def test_bootstrap_rejects_empty_or_repeated_target_inputs() -> None:
    settings = _settings()
    with pytest.raises(ValueError, match="requires target-level"):
        stratified_paired_interval((), settings)
    repeated_target = _difference(DatasetName.N_BAIOT, "target-a", 0.2)
    settings = _settings()
    with pytest.raises(ValueError, match="each target exactly once"):
        stratified_paired_interval((repeated_target, repeated_target), settings)


def _alpha_and_reporting() -> tuple[AlphaLevel, ResolvedConfiguration]:
    configuration = two_seed_configuration()
    return configuration.scientific.alpha, configuration


def test_campaign_summary_rejects_inconsistentcampaign_results() -> None:
    alpha, configuration = _alpha_and_reporting()
    results = campaign_results()
    with pytest.raises(ValueError, match="requires target results"):
        summarize_campaign((), alpha, configuration.reporting)
    with pytest.raises(ValueError, match="repeated target seeds"):
        summarize_campaign((*results, results[0]), alpha, configuration.reporting)
    shifted = tuple(
        BudgetSweepPerformance(
            multiplier=point.multiplier + 0.01,
            detection_rate=point.detection_rate,
            false_alert_rate=point.false_alert_rate,
        )
        for point in results[0].budget_sweep
    )
    variant = with_attributes(results[0], budget_sweep=shifted)
    with pytest.raises(ValueError, match="sweep coverage must match across targets"):
        summarize_campaign(
            (variant, *results[1:]),
            alpha,
            configuration.reporting,
        )
    dropped = tuple(result for result in results if result.target != results[0].target)
    uneven = (
        *dropped,
        *(r for r in results if r.target == results[0].target and r.seed == 0),
    )
    with pytest.raises(ValueError, match="coverage must match across seeds"):
        summarize_campaign(uneven, alpha, configuration.reporting)


def test_matched_fpr_interpolation_edge_cases() -> None:
    def point(multiplier: float, tpr: float, fpr: float) -> BudgetSweepPerformance:
        return BudgetSweepPerformance(
            multiplier=multiplier, detection_rate=tpr, false_alert_rate=fpr
        )

    point_input = point(1.0, 0.5, 0.1)
    with pytest.raises(ValueError, match="at least two"):
        matched_fpr_detection_rate((point_input,), 0.1)
    point_input = point(1.0, 0.5, 0.1)
    point_input = point(1.0, 0.6, 0.2)
    with pytest.raises(ValueError, match="unique and increasing"):
        matched_fpr_detection_rate((point_input, point_input), 0.1)
    flat = (point(0.5, 0.5, 0.1), point(1.0, 0.6, 0.1))
    assert matched_fpr_detection_rate(flat, 0.1) is None
    rising = (point(0.5, 0.4, 0.05), point(1.0, 0.8, 0.1))
    assert matched_fpr_detection_rate(rising, 0.5) is None
    matched = matched_fpr_detection_rate(rising, 0.075)
    assert matched is not None
    assert abs(matched - 0.6) < 1e-9


def _with_legacy_arm(
    result: TargetPerformance, detection_rate: float, false_alert_rate: float
) -> TargetPerformance:
    arm = ArmPerformance(
        arm=StudyArm.MEAN_ENSEMBLE_LOCAL_BRANCH,
        false_alert_rate=false_alert_rate,
        detection_rates=(detection_rate, detection_rate),
    )
    return replace(result, arms=(arm,))


@pytest.mark.parametrize(
    ("arm_false_alert_rate", "expected_matched_rate"),
    [(0.0, 0.2), (0.035, 0.5), (0.1, 0.6 + 0.2 / 3.0), (0.5, 0.8)],
)
def test_arm_matched_difference_clamps_outside_the_sweep_and_interpolates_inside(
    arm_false_alert_rate: float, expected_matched_rate: float
) -> None:
    results = tuple(
        _with_legacy_arm(
            _target_result("a", seed, (0.8, 0.6), (0.7, 0.5), 0.05), 0.4, arm_false_alert_rate
        )
        for seed in (1, 2)
    )
    summary = summarize_campaign(results, AlphaLevel.ALPHA_005, _reporting_settings())
    (arm,) = summary.arms
    (target,) = arm.targets
    assert isclose(target.matched_difference, expected_matched_rate - 0.4, abs_tol=1e-9)
    assert isclose(target.nominal_difference, 0.7 - 0.4, abs_tol=1e-9)
    assert [item.seed for item in target.seed_false_alert_rates] == [1, 2]


def test_minimum_detectable_effect_matches_the_closed_form() -> None:
    normal = NormalDist()
    z = normal.inv_cdf(0.975) + normal.inv_cdf(0.8)

    assert isclose(
        paired_minimum_detectable_effect(47, 0.1, PROTOCOL), z * 0.1 / sqrt(47), rel_tol=1e-12
    )
    assert paired_minimum_detectable_effect(47, 0.1, PROTOCOL) > paired_minimum_detectable_effect(
        58, 0.1, PROTOCOL
    )
    assert paired_minimum_detectable_effect(47, 0.2, PROTOCOL) > paired_minimum_detectable_effect(
        47, 0.1, PROTOCOL
    )


def test_power_at_the_meaningful_effect_is_consistent_with_the_detectable_effect() -> None:
    count, sd = 47, 0.05
    detectable = paired_minimum_detectable_effect(count, sd, PROTOCOL)
    widened = PROTOCOL.model_copy(update={"minimum_meaningful_effect": detectable})

    assert isclose(paired_power_at_effect(count, sd, widened), 0.8, abs_tol=1e-3)
    assert (
        0.0
        < paired_power_at_effect(count, 0.2, PROTOCOL)
        < paired_power_at_effect(count, 0.05, PROTOCOL)
    )
    assert paired_power_at_effect(10_000, 0.05, PROTOCOL) < 1.0


def _plan(stratum: StudyStratum, specs: dict[BudgetAnalysisKind, int]) -> StratumBudgetPlan:
    names = tuple(ClientName(f"t{i}") for i in range(max(specs.values())))
    return StratumBudgetPlan(
        stratum,
        (100, 400),
        (),
        tuple(BudgetAnalysis(kind, (100, 400), names[:count]) for kind, count in specs.items()),
        (),
        (),
    )


def test_power_summary_pools_each_analysis_across_strata_without_using_outcomes() -> None:
    plans = (
        _plan(
            StudyStratum.TON_IOT,
            {BudgetAnalysisKind.UNIVERSAL_PRIMARY: 9, BudgetAnalysisKind.HIGH_RESOURCE: 1},
        ),
        _plan(StudyStratum.UNSW_IOT_ATTACK_FLOWS_G1, {BudgetAnalysisKind.UNIVERSAL_PRIMARY: 5}),
    )

    power = summarize_budget_power(plans, protocol_matrix((100, 400)))

    assert [(item.kind, item.target_count) for item in power] == [
        (BudgetAnalysisKind.UNIVERSAL_PRIMARY, 14)
    ]
    assert [point.difference_sd for point in power[0].points] == [0.05, 0.1]
    assert (
        power[0].points[0].minimum_detectable_effect < power[0].points[1].minimum_detectable_effect
    )
