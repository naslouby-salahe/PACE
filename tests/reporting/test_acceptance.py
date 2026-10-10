from dataclasses import replace
from math import isclose

import pytest

from pace.experiment.preflight import matrix_variants
from pace.reporting.acceptance import summarize_acceptance
from pace.reporting.records import StratumBudgetEvidence, VariantMetrics
from pace.types import (
    AcceptanceCriterion,
    AcceptanceDesign,
    AlphaLevel,
    ArmMetricSummary,
    ArmTargetSummary,
    BudgetAnalysis,
    BudgetAnalysisKind,
    ClientName,
    CriterionVerdict,
    DatasetClientId,
    DatasetName,
    DatasetValidationError,
    SeedFalseAlertRates,
    SeedRateDifference,
    StudyStratum,
)
from tests.support import campaign_configuration, minimal_campaign_metrics

REPORTING = campaign_configuration().reporting
SEEDS = (6, 7, 8)


def _target(
    dataset: DatasetName,
    name: str,
    gain: float,
    method_fpr: float = 0.03,
    arm_fpr: float = 0.03,
) -> ArmTargetSummary:
    return ArmTargetSummary(
        target=DatasetClientId(dataset, ClientName(name)),
        nominal_difference=gain,
        matched_difference=gain,
        method_false_alert_rate=method_fpr,
        arm_false_alert_rate=arm_fpr,
        seed_differences=tuple(SeedRateDifference(seed, gain) for seed in SEEDS),
        seed_false_alert_rates=tuple(
            SeedFalseAlertRates(seed, method_fpr, arm_fpr) for seed in SEEDS
        ),
    )


def _evidence(
    stratum: StudyStratum,
    targets: tuple[ArmTargetSummary, ...],
    alpha: AlphaLevel = AlphaLevel.ALPHA_001,
    with_comparator: bool = True,
) -> StratumBudgetEvidence:
    names = tuple(item.target.name for item in targets)
    analyses = (BudgetAnalysis(BudgetAnalysisKind.UNIVERSAL_PRIMARY, (100, 400), names),)
    arm = ArmMetricSummary(
        arm=AcceptanceDesign.PROTOCOL.comparator,
        target_count=len(targets),
        macro_detection_rate=0.5,
        macro_detection_difference=0.0,
        method_arm_difference=0.0,
        method_arm_confidence_interval=minimal_campaign_metrics().method_local_confidence_interval,
        worst_detection_difference=0.0,
        catastrophic_cell_count=0,
        harmed_target_count=0,
        mean_false_alert_rate=0.03,
        worst_false_alert_rate=0.03,
        targets=targets,
    )
    metrics = replace(minimal_campaign_metrics(), arms=(arm,) if with_comparator else ())
    variants = tuple(
        VariantMetrics(variant, metrics) for variant in matrix_variants(analyses, (alpha,))
    )
    return StratumBudgetEvidence(stratum, analyses, len(targets), variants)


def _targets(
    dataset: DatasetName, gains: tuple[float, ...], **rates: float
) -> tuple[ArmTargetSummary, ...]:
    return tuple(
        _target(dataset, f"{dataset.name}-{index}", gain, **rates)
        for index, gain in enumerate(gains)
    )


def _evidence_pair(
    ton_gains: tuple[float, ...],
    other_gains: tuple[float, ...],
    alpha: AlphaLevel = AlphaLevel.ALPHA_001,
    **rates: float,
) -> tuple[StratumBudgetEvidence, ...]:
    return (
        _evidence(StudyStratum.TON_IOT, _targets(DatasetName.TON_IOT, ton_gains, **rates), alpha),
        _evidence(
            StudyStratum.UNSW_IOT_ATTACK_FLOWS_G1,
            _targets(DatasetName.UNSW_IOT_ATTACK_FLOWS, other_gains, **rates),
            alpha,
        ),
    )


def _failed(verdicts: tuple[CriterionVerdict, ...]) -> set[AcceptanceCriterion]:
    return {verdict.criterion for verdict in verdicts if not verdict.satisfied}


def test_consistent_gains_pass_every_criterion() -> None:
    report = summarize_acceptance(
        _evidence_pair((0.1, 0.12, 0.11, 0.09), (0.02, 0.03, 0.025, 0.02)), REPORTING
    )
    assert report.is_accepted
    assert {cell.local_training_rows for cell in report.cells} == {100, 400}
    assert all(cell.is_gating and not _failed(cell.verdicts) for cell in report.cells)


def test_one_collapsed_target_breaks_non_inferiority() -> None:
    report = summarize_acceptance(
        _evidence_pair((0.1, 0.12, 0.11, 0.09), (0.02, 0.03, 0.025, -0.33)), REPORTING
    )
    assert not report.is_accepted
    for cell in report.cells:
        assert cell.severe_target_count == 1
        assert isclose(cell.worst_target_difference, -0.33)
        assert AcceptanceCriterion.NON_INFERIORITY in _failed(cell.verdicts)


def test_gain_only_in_the_excluded_dataset_is_rejected() -> None:
    report = summarize_acceptance(
        _evidence_pair((0.2, 0.22, 0.21, 0.19), (0.0, 0.0, 0.0, 0.0)), REPORTING
    )
    assert not report.is_accepted
    for cell in report.cells:
        assert cell.gain_without_excluded is not None
        assert isclose(cell.gain_without_excluded, 0.0, abs_tol=1e-12)
        assert AcceptanceCriterion.GAIN_WITHOUT_EXCLUDED_DATASET in _failed(cell.verdicts)
        assert AcceptanceCriterion.STRATUM_ROBUSTNESS in _failed(cell.verdicts)


def test_false_alert_inflation_is_rejected_per_alpha() -> None:
    gains = ((0.1, 0.12, 0.11, 0.09), (0.02, 0.03, 0.025, 0.02))
    inflated = summarize_acceptance(
        _evidence_pair(*gains, method_fpr=0.06, arm_fpr=0.03), REPORTING
    )
    for cell in inflated.cells:
        assert isclose(cell.false_alert_increase, 0.03)
        assert AcceptanceCriterion.FALSE_ALERT_RATE in _failed(cell.verdicts)
    slight = summarize_acceptance(
        _evidence_pair(*gains, AlphaLevel.ALPHA_005, method_fpr=0.0315, arm_fpr=0.03), REPORTING
    )
    assert all(
        AcceptanceCriterion.FALSE_ALERT_RATE not in _failed(cell.verdicts) for cell in slight.cells
    )


def test_a_negative_seed_breaks_seed_signs() -> None:
    evidence = _evidence_pair((0.1, 0.12), (0.02, 0.03))
    flipped = tuple(
        replace(
            target,
            seed_differences=(*target.seed_differences[:2], SeedRateDifference(8, -0.5)),
        )
        for target in evidence[0].variants[0].metrics.arms[0].targets
    )
    arm = replace(evidence[0].variants[0].metrics.arms[0], targets=flipped)
    metrics = replace(evidence[0].variants[0].metrics, arms=(arm,))
    broken = StratumBudgetEvidence(
        evidence[0].stratum,
        evidence[0].analyses,
        evidence[0].eligible_targets,
        tuple(VariantMetrics(entry.variant, metrics) for entry in evidence[0].variants),
    )
    report = summarize_acceptance((broken, evidence[1]), REPORTING)
    assert all(AcceptanceCriterion.SEED_SIGNS in _failed(cell.verdicts) for cell in report.cells)


def test_a_campaign_without_the_comparator_arm_is_rejected() -> None:
    targets = _targets(DatasetName.TON_IOT, (0.1, 0.1))
    with pytest.raises(DatasetValidationError, match="MEAN_ENSEMBLE_LOCAL_BRANCH"):
        summarize_acceptance(
            (_evidence(StudyStratum.TON_IOT, targets, with_comparator=False),), REPORTING
        )


def test_scaling_analyses_report_only_their_cohort_complete_budget() -> None:
    names = tuple(item.target.name for item in _targets(DatasetName.TON_IOT, (0.1, 0.1)))
    analyses = (BudgetAnalysis(BudgetAnalysisKind.BROAD_SCALING, (100, 400, 2000), names),)
    base = _evidence(StudyStratum.TON_IOT, _targets(DatasetName.TON_IOT, (0.1, 0.1)))
    variants = tuple(
        VariantMetrics(variant, base.variants[0].metrics)
        for variant in matrix_variants(analyses, (AlphaLevel.ALPHA_001,))
    )
    evidence = StratumBudgetEvidence(StudyStratum.TON_IOT, analyses, 2, variants)
    report = summarize_acceptance((evidence,), REPORTING)
    assert [(cell.kind, cell.local_training_rows) for cell in report.cells] == [
        (BudgetAnalysisKind.BROAD_SCALING, 2000)
    ]
    assert not report.is_accepted
