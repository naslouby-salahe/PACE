from math import isclose

import pytest

from pace.experiment.preflight import matrix_variants
from pace.reporting.records import StratumBudgetEvidence, VariantMetrics, summarize_budget_effects
from pace.types import (
    AlphaLevel,
    BudgetAnalysis,
    BudgetAnalysisKind,
    BudgetAnalysisScope,
    BudgetEffectAnalysis,
    BudgetOutcome,
    ClientName,
    DatasetClientId,
    DatasetName,
    DatasetValidationError,
    LocalTrainingRowCount,
    StudyStratum,
    TargetMetricSummary,
)
from tests.support import campaign_configuration, minimal_campaign_metrics

REPORTING = campaign_configuration().reporting
KINDS = (
    BudgetAnalysisKind.UNIVERSAL_PRIMARY,
    BudgetAnalysisKind.UNIVERSAL_ROBUSTNESS,
    BudgetAnalysisKind.BROAD_SCALING,
    BudgetAnalysisKind.HIGH_RESOURCE,
)
BUDGETS = {
    BudgetAnalysisKind.UNIVERSAL_PRIMARY: (100, 400),
    BudgetAnalysisKind.UNIVERSAL_ROBUSTNESS: (100, 400, 750),
    BudgetAnalysisKind.BROAD_SCALING: (100, 400, 2000),
    BudgetAnalysisKind.HIGH_RESOURCE: (100, 400, 2000, 20000),
}
GAIN = {100: 0.0, 400: 0.1, 750: 0.15, 2000: 0.25, 20000: 0.3}
ALPHAS = (AlphaLevel.ALPHA_001,)


def _names(*names: str) -> tuple[ClientName, ...]:
    return tuple(ClientName(name) for name in names)


SCALE: dict[str, float] = {}


def _summary(
    dataset: DatasetName, name: str, rows: LocalTrainingRowCount, base: float
) -> TargetMetricSummary:
    template = minimal_campaign_metrics().targets[0]
    return TargetMetricSummary(
        target=DatasetClientId(dataset, ClientName(name)),
        method_detection_rate=base + SCALE.get(name, 1.0) * GAIN[rows],
        local_detection_rate=base,
        max_fusion_detection_rate=base + 0.02,
        bonferroni_detection_rate=template.bonferroni_detection_rate,
        detection_difference=GAIN[rows],
        max_fusion_detection_difference=0.02,
        method_max_fusion_difference=GAIN[rows] - 0.02,
        realized_false_alert_rate=0.01 + GAIN[rows] / 10,
        local_false_alert_rate=template.local_false_alert_rate,
        max_fusion_false_alert_rate=template.max_fusion_false_alert_rate,
        bonferroni_false_alert_rate=template.bonferroni_false_alert_rate,
        matched_fpr_difference=template.matched_fpr_difference,
    )


def _evidence(
    stratum: StudyStratum,
    dataset: DatasetName,
    eligible: tuple[ClientName, ...],
    cohorts: dict[BudgetAnalysisKind, tuple[ClientName, ...]],
    alphas: tuple[AlphaLevel, ...] = ALPHAS,
) -> StratumBudgetEvidence:
    analyses = tuple(
        BudgetAnalysis(kind, BUDGETS[kind], cohort) for kind, cohort in cohorts.items() if cohort
    )
    variants = tuple(
        VariantMetrics(
            variant,
            minimal_campaign_metrics(
                tuple(
                    _summary(dataset, name, variant.local_training_rows, 0.4 + 0.1 * index)
                    for index, name in enumerate(variant.targets)
                )
            ),
        )
        for variant in matrix_variants(analyses, alphas)
    )
    return StratumBudgetEvidence(stratum, analyses, len(eligible), variants)


def _ton(alphas: tuple[AlphaLevel, ...] = ALPHAS) -> StratumBudgetEvidence:
    everyone = _names("a", "b", "c")
    return _evidence(
        StudyStratum.TON_IOT,
        DatasetName.TON_IOT,
        everyone,
        {
            BudgetAnalysisKind.UNIVERSAL_PRIMARY: everyone,
            BudgetAnalysisKind.UNIVERSAL_ROBUSTNESS: everyone,
            BudgetAnalysisKind.BROAD_SCALING: _names("a", "b"),
            BudgetAnalysisKind.HIGH_RESOURCE: _names("a"),
        },
        alphas,
    )


def _unsw(alphas: tuple[AlphaLevel, ...] = ALPHAS) -> StratumBudgetEvidence:
    everyone = _names("x", "y")
    return _evidence(
        StudyStratum.UNSW_IOT_ATTACK_FLOWS_G1,
        DatasetName.UNSW_IOT_ATTACK_FLOWS,
        everyone,
        dict.fromkeys(KINDS, everyone),
        alphas,
    )


def _report(
    *evidence: StratumBudgetEvidence,
) -> dict[tuple[BudgetAnalysisKind, BudgetAnalysisScope], BudgetEffectAnalysis]:
    report = summarize_budget_effects(evidence, KINDS, REPORTING)
    return {(item.kind, item.scope): item for item in report.analyses}


def _pairs(analysis: BudgetEffectAnalysis) -> list[tuple[int, int]]:
    return [(item.from_rows, item.to_rows) for item in analysis.comparisons]


def test_universal_analyses_pool_every_eligible_target_with_exact_pairs() -> None:
    analyses = _report(_ton(), _unsw())

    primary = analyses[(BudgetAnalysisKind.UNIVERSAL_PRIMARY, BudgetAnalysisScope.POOLED)]
    assert (primary.target_count, primary.budgets) == (5, (100, 400))
    assert _pairs(primary) == [(100, 400)]
    pace = next(i for i in primary.comparisons[0].pooled if i.outcome is BudgetOutcome.PACE_TPR)
    assert isclose(pace.mean_difference, 0.1, abs_tol=1e-9)
    assert pace.interval is not None
    assert pace.interval.lower_bound <= pace.mean_difference <= pace.interval.upper_bound
    local = next(i for i in primary.comparisons[0].pooled if i.outcome is BudgetOutcome.LOCAL_TPR)
    assert isclose(local.mean_difference, 0.0, abs_tol=1e-9)

    robustness = analyses[(BudgetAnalysisKind.UNIVERSAL_ROBUSTNESS, BudgetAnalysisScope.POOLED)]
    assert robustness.target_count == 5
    assert _pairs(robustness) == [(100, 400), (400, 750), (100, 750)]


def test_broad_scaling_is_paired_inside_its_own_frozen_cohort() -> None:
    analyses = _report(_ton(), _unsw())

    broad = analyses[(BudgetAnalysisKind.BROAD_SCALING, BudgetAnalysisScope.POOLED)]
    assert broad.target_count == 4
    assert broad.budgets == (100, 400, 2000)
    assert _pairs(broad) == [(100, 400), (400, 2000), (100, 2000)]
    assert {item.target_count for item in broad.comparisons} == {4}
    last = broad.comparisons[-1]
    pace = next(i for i in last.pooled if i.outcome is BudgetOutcome.PACE_TPR)
    assert isclose(pace.mean_difference, 0.25, abs_tol=1e-9)
    assert {effect.stratum: effect.target_count for effect in last.strata} == {
        StudyStratum.TON_IOT: 2,
        StudyStratum.UNSW_IOT_ATTACK_FLOWS_G1: 2,
    }
    assert all(i.interval is None for e in last.strata for i in e.differences)


def test_high_resource_pairs_the_same_targets_across_every_budget() -> None:
    analyses = _report(_ton(), _unsw())

    high = analyses[(BudgetAnalysisKind.HIGH_RESOURCE, BudgetAnalysisScope.POOLED)]
    assert high.target_count == 3
    assert high.budgets == (100, 400, 2000, 20000)
    assert _pairs(high) == [(100, 400), (400, 2000), (2000, 20000), (100, 20000)]
    assert {item.target_count for item in high.comparisons} == {3}


def test_only_complete_strata_get_a_separate_within_stratum_result() -> None:
    analyses = _report(_ton(), _unsw())

    complete = [key for key in analyses if key[1] is BudgetAnalysisScope.COMPLETE_STRATUM]
    assert complete == [(BudgetAnalysisKind.HIGH_RESOURCE, BudgetAnalysisScope.COMPLETE_STRATUM)]
    only = analyses[complete[0]]
    assert only.strata == (StudyStratum.UNSW_IOT_ATTACK_FLOWS_G1,)
    assert only.target_count == 2
    assert len(only.comparisons) == 4


def test_feasibility_counts_each_frozen_cohort_per_stratum() -> None:
    report = summarize_budget_effects((_ton(), _unsw()), KINDS, REPORTING)

    assert report.kinds == KINDS
    counts = {
        item.stratum: (item.eligible_targets, item.cohort_sizes) for item in report.feasibility
    }
    assert counts == {
        StudyStratum.TON_IOT: (3, (3, 3, 2, 1)),
        StudyStratum.UNSW_IOT_ATTACK_FLOWS_G1: (2, (2, 2, 2, 2)),
    }


def test_strata_without_a_cohort_for_an_analysis_simply_do_not_contribute() -> None:
    everyone = _names("p", "q")
    g2 = _evidence(
        StudyStratum.UNSW_IOT_ATTACK_FLOWS_G2,
        DatasetName.UNSW_IOT_ATTACK_FLOWS,
        everyone,
        {
            BudgetAnalysisKind.UNIVERSAL_PRIMARY: everyone,
            BudgetAnalysisKind.UNIVERSAL_ROBUSTNESS: everyone,
            BudgetAnalysisKind.BROAD_SCALING: everyone,
            BudgetAnalysisKind.HIGH_RESOURCE: (),
        },
    )

    analyses = _report(_ton(), g2)

    assert (
        analyses[(BudgetAnalysisKind.HIGH_RESOURCE, BudgetAnalysisScope.POOLED)].target_count == 1
    )
    assert (
        analyses[(BudgetAnalysisKind.UNIVERSAL_PRIMARY, BudgetAnalysisScope.POOLED)].target_count
        == 5
    )
    assert (BudgetAnalysisKind.HIGH_RESOURCE, BudgetAnalysisScope.COMPLETE_STRATUM) not in analyses


def test_every_alpha_level_gets_its_own_paired_comparison() -> None:
    alphas = (AlphaLevel.ALPHA_001, AlphaLevel.ALPHA_005)

    analyses = _report(_ton(alphas), _unsw(alphas))

    primary = analyses[(BudgetAnalysisKind.UNIVERSAL_PRIMARY, BudgetAnalysisScope.POOLED)]
    assert [item.alpha for item in primary.comparisons] == list(alphas)


def test_a_missing_variant_is_an_error_not_a_silent_gap() -> None:
    ton = _ton()
    broken = StratumBudgetEvidence(
        ton.stratum,
        ton.analyses,
        ton.eligible_targets,
        tuple(entry for entry in ton.variants if entry.variant.local_training_rows != 400),
    )

    with pytest.raises(DatasetValidationError, match="No campaign variant covers"):
        summarize_budget_effects((broken, _unsw()), KINDS, REPORTING)


def test_pooled_analyses_require_identical_budgets_in_every_stratum() -> None:
    ton = _ton()
    shifted = StratumBudgetEvidence(
        StudyStratum.UNSW_IOT_ATTACK_FLOWS_G1,
        (BudgetAnalysis(BudgetAnalysisKind.UNIVERSAL_PRIMARY, (100, 750), _names("x")),),
        1,
        (),
    )

    with pytest.raises(DatasetValidationError, match="identical budgets"):
        summarize_budget_effects((ton, shifted), KINDS, REPORTING)


def test_budget_effects_need_at_least_one_stratum() -> None:
    with pytest.raises(DatasetValidationError, match="at least one study stratum"):
        summarize_budget_effects((), KINDS, REPORTING)
