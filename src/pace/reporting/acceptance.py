from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, replace

import numpy as np

from pace.config import ReportingSettings
from pace.experiment.preflight import MatrixVariant
from pace.reporting.metrics import stratified_paired_interval
from pace.reporting.records import (
    ReportInputs,
    ResultTable,
    StratumBudgetEvidence,
    alpha_text,
    analysis_text,
    count,
    number,
    verdict_text,
)
from pace.types import (
    AcceptanceCell,
    AcceptanceCriterion,
    AcceptanceDesign,
    AcceptanceMeasure,
    AcceptanceReport,
    AcceptanceRule,
    AlphaLevel,
    ArmTargetSummary,
    BudgetAnalysis,
    BudgetAnalysisKind,
    ColumnHeading,
    CriterionSatisfied,
    CriterionVerdict,
    DatasetValidationError,
    DisplayText,
    FalseAlertRate,
    LearningSeed,
    LocalTrainingRowCount,
    RateConfidenceInterval,
    RateDifference,
    SeedFalseAlertRates,
    SeedRateDifference,
    StudyStratum,
    TableName,
    TargetPairedDifference,
    UnavailableReading,
)


@dataclass(frozen=True, slots=True)
class _CellKey:
    kind: BudgetAnalysisKind
    alpha: AlphaLevel
    rows: LocalTrainingRowCount


@dataclass(frozen=True, slots=True)
class _Pooled:
    stratum: StudyStratum
    summary: ArmTargetSummary


def _cohort_complete_budgets(
    variant: MatrixVariant, analyses: tuple[BudgetAnalysis, ...]
) -> tuple[LocalTrainingRowCount, ...]:
    budgets = next(item for item in analyses if item.kind is variant.kind).budgets
    if variant.kind is BudgetAnalysisKind.UNIVERSAL_PRIMARY:
        return budgets
    return (max(budgets),)


def _pooled_by_cell(evidence: tuple[StratumBudgetEvidence, ...]) -> dict[_CellKey, list[_Pooled]]:
    pooled: dict[_CellKey, list[_Pooled]] = defaultdict(list)
    for stratum_evidence in evidence:
        for entry in stratum_evidence.variants:
            complete = _cohort_complete_budgets(entry.variant, stratum_evidence.analyses)
            if entry.variant.local_training_rows not in complete:
                continue
            comparator = AcceptanceDesign.PROTOCOL.comparator
            arm = next((item for item in entry.metrics.arms if item.arm is comparator), None)
            if arm is None:
                raise DatasetValidationError(
                    f"{stratum_evidence.stratum.name} campaign has no {comparator.name} arm"
                )
            key = _CellKey(
                entry.variant.kind, entry.variant.alpha, entry.variant.local_training_rows
            )
            pooled[key].extend(_Pooled(stratum_evidence.stratum, item) for item in arm.targets)
    return pooled


def _interval(
    members: list[_Pooled],
    value: Callable[[ArmTargetSummary], RateDifference],
    settings: ReportingSettings,
) -> tuple[RateDifference, RateConfidenceInterval] | None:
    differences = tuple(
        TargetPairedDifference(member.summary.target, value(member.summary)) for member in members
    )
    if not differences:
        return None
    return (
        np.mean([item.difference for item in differences]).item(),
        stratified_paired_interval(differences, settings),
    )


def _seed_gains(members: list[_Pooled]) -> tuple[SeedRateDifference, ...]:
    by_seed: dict[LearningSeed, list[RateDifference]] = defaultdict(list)
    for member in members:
        for item in member.summary.seed_differences:
            by_seed[item.seed].append(item.difference)
    return tuple(
        SeedRateDifference(seed, np.mean(by_seed[seed]).item()) for seed in sorted(by_seed)
    )


def _stratum_left_out_minimum(members: list[_Pooled]) -> RateDifference:
    strata = {member.stratum for member in members}
    remaining = [
        [
            member.summary.nominal_difference
            for member in members
            if len(strata) < 2 or member.stratum is not left_out
        ]
        for left_out in strata
    ]
    return min(np.mean(values).item() for values in remaining)


def _exceedance_share(
    members: list[_Pooled],
    alpha: AlphaLevel,
    rate: Callable[[SeedFalseAlertRates], FalseAlertRate],
) -> RateDifference:
    limit = AcceptanceRule.EXCEEDANCE_MULTIPLE.threshold * alpha.fraction
    return np.mean(
        [rate(cell) > limit for member in members for cell in member.summary.seed_false_alert_rates]
    ).item()


def _verdicts(cell: AcceptanceCell) -> tuple[CriterionVerdict, ...]:
    floor = AcceptanceRule.GAIN_LOWER_BOUND_FLOOR.threshold
    without = cell.gain_without_excluded_interval
    checks = {
        AcceptanceCriterion.NON_INFERIORITY: cell.severe_target_count == 0
        and cell.harmed_target_count / cell.target_count
        <= AcceptanceRule.HARMED_TARGET_SHARE.threshold,
        AcceptanceCriterion.NOMINAL_GAIN: cell.nominal_interval.lower_bound > 0.0,
        AcceptanceCriterion.MATCHED_GAIN: cell.matched_interval.lower_bound > 0.0,
        AcceptanceCriterion.FALSE_ALERT_RATE: cell.false_alert_increase
        <= cell.alpha.acceptance_false_alert_increase
        and cell.exceedance_increase <= AcceptanceRule.EXCEEDANCE_INCREASE.threshold,
        AcceptanceCriterion.GAIN_WITHOUT_EXCLUDED_DATASET: cell.gain_without_excluded is not None
        and without is not None
        and cell.gain_without_excluded >= cell.alpha.acceptance_gain_without_excluded_dataset
        and without.lower_bound >= floor,
        AcceptanceCriterion.SEED_SIGNS: len(cell.seed_gains) > 0
        and min(item.difference for item in cell.seed_gains) > 0.0,
        AcceptanceCriterion.STRATUM_ROBUSTNESS: cell.minimum_stratum_left_out_gain > 0.0,
    }
    return tuple(CriterionVerdict(criterion, satisfied) for criterion, satisfied in checks.items())


def _cell(key: _CellKey, members: list[_Pooled], settings: ReportingSettings) -> AcceptanceCell:
    nominal = _interval(members, lambda item: item.nominal_difference, settings)
    if nominal is None:
        raise DatasetValidationError("Acceptance analysis requires target-level differences")
    matched = _interval(members, lambda item: item.matched_difference, settings)
    if matched is None:
        raise DatasetValidationError("Acceptance analysis requires target-level differences")
    without = _interval(
        [
            member
            for member in members
            if member.summary.target.dataset is not AcceptanceDesign.PROTOCOL.excluded_dataset
        ],
        lambda item: item.nominal_difference,
        settings,
    )
    losses = np.asarray([member.summary.nominal_difference for member in members])
    unverified = AcceptanceCell(
        kind=key.kind,
        alpha=key.alpha,
        local_training_rows=key.rows,
        target_count=len(members),
        nominal_gain=nominal[0],
        nominal_interval=nominal[1],
        matched_gain=matched[0],
        matched_interval=matched[1],
        gain_without_excluded=None if without is None else without[0],
        gain_without_excluded_interval=None if without is None else without[1],
        harmed_target_count=np.sum(losses < -AcceptanceRule.HARMED_TARGET_LOSS.threshold).item(),
        severe_target_count=np.sum(losses < -AcceptanceRule.SEVERE_TARGET_LOSS.threshold).item(),
        worst_target_difference=np.min(losses).item(),
        false_alert_increase=np.mean(
            [
                member.summary.method_false_alert_rate - member.summary.arm_false_alert_rate
                for member in members
            ]
        ).item(),
        exceedance_increase=_exceedance_share(members, key.alpha, lambda cell: cell.method)
        - _exceedance_share(members, key.alpha, lambda cell: cell.arm),
        seed_gains=_seed_gains(members),
        minimum_stratum_left_out_gain=_stratum_left_out_minimum(members),
        verdicts=(),
    )
    return replace(unverified, verdicts=_verdicts(unverified))


def summarize_acceptance(
    evidence: tuple[StratumBudgetEvidence, ...], settings: ReportingSettings
) -> AcceptanceReport:
    pooled = _pooled_by_cell(evidence)
    return AcceptanceReport(
        AcceptanceDesign.PROTOCOL.comparator,
        tuple(
            _cell(key, pooled[key], settings)
            for key in sorted(
                pooled,
                key=lambda item: (
                    tuple(BudgetAnalysisKind).index(item.kind),
                    item.alpha.fraction,
                    item.rows,
                ),
            )
        ),
    )


@dataclass(frozen=True, slots=True)
class ObservedCheck:
    criterion: AcceptanceCriterion
    measure: AcceptanceMeasure
    measured: RateDifference | None
    threshold: RateDifference
    satisfied: CriterionSatisfied


def _inferiority_checks(cell: AcceptanceCell) -> tuple[ObservedCheck, ...]:
    share = cell.harmed_target_count / cell.target_count
    limit = AcceptanceRule.HARMED_TARGET_SHARE.threshold
    return (
        ObservedCheck(
            AcceptanceCriterion.NON_INFERIORITY,
            AcceptanceMeasure.SEVERE_TARGETS,
            cell.severe_target_count,
            0.0,
            cell.severe_target_count == 0,
        ),
        ObservedCheck(
            AcceptanceCriterion.NON_INFERIORITY,
            AcceptanceMeasure.HARMED_TARGET_SHARE,
            share,
            limit,
            share <= limit,
        ),
    )


def _gain_checks(cell: AcceptanceCell) -> tuple[ObservedCheck, ...]:
    floor = AcceptanceRule.GAIN_LOWER_BOUND_FLOOR.threshold
    without = cell.gain_without_excluded_interval
    minimum = cell.alpha.acceptance_gain_without_excluded_dataset
    return (
        ObservedCheck(
            AcceptanceCriterion.NOMINAL_GAIN,
            AcceptanceMeasure.NOMINAL_GAIN_LOWER_BOUND,
            cell.nominal_interval.lower_bound,
            0.0,
            cell.nominal_interval.lower_bound > 0.0,
        ),
        ObservedCheck(
            AcceptanceCriterion.MATCHED_GAIN,
            AcceptanceMeasure.MATCHED_GAIN_LOWER_BOUND,
            cell.matched_interval.lower_bound,
            0.0,
            cell.matched_interval.lower_bound > 0.0,
        ),
        ObservedCheck(
            AcceptanceCriterion.GAIN_WITHOUT_EXCLUDED_DATASET,
            AcceptanceMeasure.GAIN_WITHOUT_EXCLUDED,
            cell.gain_without_excluded,
            minimum,
            cell.gain_without_excluded is not None and cell.gain_without_excluded >= minimum,
        ),
        ObservedCheck(
            AcceptanceCriterion.GAIN_WITHOUT_EXCLUDED_DATASET,
            AcceptanceMeasure.LOWER_BOUND_WITHOUT_EXCLUDED,
            None if without is None else without.lower_bound,
            floor,
            without is not None and without.lower_bound >= floor,
        ),
    )


def _stability_checks(cell: AcceptanceCell) -> tuple[ObservedCheck, ...]:
    increase = cell.alpha.acceptance_false_alert_increase
    exceedance = AcceptanceRule.EXCEEDANCE_INCREASE.threshold
    smallest_seed = min((item.difference for item in cell.seed_gains), default=0.0)
    return (
        ObservedCheck(
            AcceptanceCriterion.FALSE_ALERT_RATE,
            AcceptanceMeasure.FALSE_ALERT_RATE_INCREASE,
            cell.false_alert_increase,
            increase,
            cell.false_alert_increase <= increase,
        ),
        ObservedCheck(
            AcceptanceCriterion.FALSE_ALERT_RATE,
            AcceptanceMeasure.EXCEEDANCE_SHARE_INCREASE,
            cell.exceedance_increase,
            exceedance,
            cell.exceedance_increase <= exceedance,
        ),
        ObservedCheck(
            AcceptanceCriterion.SEED_SIGNS,
            AcceptanceMeasure.SMALLEST_SEED_GAIN,
            smallest_seed,
            0.0,
            len(cell.seed_gains) > 0 and smallest_seed > 0.0,
        ),
        ObservedCheck(
            AcceptanceCriterion.STRATUM_ROBUSTNESS,
            AcceptanceMeasure.SMALLEST_LEAVE_ONE_STRATUM_GAIN,
            cell.minimum_stratum_left_out_gain,
            0.0,
            cell.minimum_stratum_left_out_gain > 0.0,
        ),
    )


def _measured_text(value: RateDifference | None) -> DisplayText:
    if value is None:
        return DisplayText(UnavailableReading.NOT_AVAILABLE)
    return number(value)


def _criterion_rows(cell: AcceptanceCell) -> tuple[tuple[DisplayText, ...], ...]:
    checks = (*_inferiority_checks(cell), *_gain_checks(cell), *_stability_checks(cell))
    return tuple(
        (
            analysis_text(cell.kind),
            alpha_text(cell.alpha),
            count(cell.local_training_rows),
            DisplayText(check.criterion.name.lower()),
            check.measure.label,
            _measured_text(check.measured),
            number(check.threshold),
            verdict_text(check.satisfied),
        )
        for check in checks
    )


def acceptance_table(inputs: ReportInputs) -> ResultTable:
    return ResultTable(
        TableName.ACCEPTANCE,
        (
            ColumnHeading.ANALYSIS,
            ColumnHeading.ALPHA,
            ColumnHeading.LOCAL_ROWS,
            ColumnHeading.CRITERION,
            ColumnHeading.MEASURE,
            ColumnHeading.VALUE,
            ColumnHeading.THRESHOLD,
            ColumnHeading.SATISFIED,
        ),
        tuple(row for cell in inputs.acceptance.cells for row in _criterion_rows(cell)),
    )
