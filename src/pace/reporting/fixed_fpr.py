from dataclasses import dataclass

import numpy as np

from pace.config import ReportingSettings
from pace.reporting.metrics import stratified_paired_interval
from pace.reporting.operating import (
    TargetComparison,
    comparison_references,
    curve_of,
    primary_comparisons,
    read_curve,
    unique_results,
)
from pace.reporting.records import ReportInputs, ResultTable, count, number, optional_number
from pace.types import (
    AlphaLevel,
    BudgetAnalysisKind,
    ColumnHeading,
    DatasetClientId,
    DetectionRate,
    DisplayText,
    FalseAlertRate,
    FprLevel,
    LearningSeed,
    LocalTrainingRowCount,
    MatchDirection,
    OperatingMethod,
    RateConfidenceInterval,
    RateDifference,
    SeedRateDifference,
    StudyStratum,
    TableName,
    TargetPairedDifference,
    TargetPerformance,
)


@dataclass(frozen=True, slots=True)
class TargetLevel:
    target: DatasetClientId
    stratum: StudyStratum
    seed_gains: tuple[SeedRateDifference, ...]
    subject_detection: DetectionRate
    reference_detection: DetectionRate

    @property
    def gain(self) -> RateDifference:
        return np.mean([item.difference for item in self.seed_gains]).item()


@dataclass(frozen=True, slots=True)
class CellResults:
    kind: BudgetAnalysisKind
    alpha: AlphaLevel
    rows: LocalTrainingRowCount
    targets: dict[DatasetClientId, list[TargetPerformance]]


def cell_results(inputs: ReportInputs) -> tuple[CellResults, ...]:
    grouped = unique_results(inputs.performances)
    cells: list[CellResults] = []
    for kind in BudgetAnalysisKind:
        budgets = inputs.budgets(kind)
        if not budgets:
            continue
        admitted = budgets if kind is BudgetAnalysisKind.UNIVERSAL_PRIMARY else (max(budgets),)
        cohort = inputs.cohort(kind)
        for (alpha, rows), targets in sorted(
            grouped.items(), key=lambda item: (item[0][0].fraction, item[0][1])
        ):
            selected = {
                target: results
                for target, results in sorted(targets.items(), key=lambda item: item[0].name)
                if target in cohort
            }
            if rows in admitted and selected:
                cells.append(CellResults(kind, alpha, rows, selected))
    return tuple(cells)


def _seed_reading(
    result: TargetPerformance,
    subject: OperatingMethod,
    reference: OperatingMethod,
    rate: FalseAlertRate,
) -> tuple[LearningSeed, DetectionRate, DetectionRate] | None:
    subject_curve, reference_curve = curve_of(result, subject), curve_of(result, reference)
    if subject_curve is None or reference_curve is None:
        return None
    tolerance = 0.5 / result.benign_test_rows
    read_subject = read_curve(subject_curve, rate, tolerance)
    read_reference = read_curve(reference_curve, rate, tolerance)
    if read_subject.detection is None or read_reference.detection is None:
        return None
    return result.seed, read_subject.detection, read_reference.detection


def target_level(
    results: list[TargetPerformance],
    subject: OperatingMethod,
    reference: OperatingMethod,
    level: FprLevel,
    alpha: AlphaLevel,
) -> TargetLevel | None:
    rate = level.rate(alpha)
    readings = [_seed_reading(result, subject, reference, rate) for result in results]
    supported = [reading for reading in readings if reading is not None]
    if not supported or len(supported) != len(readings):
        return None
    return TargetLevel(
        results[0].target,
        results[0].study_stratum,
        tuple(SeedRateDifference(seed, shown - base) for seed, shown, base in supported),
        np.mean([shown for _, shown, _ in supported]).item(),
        np.mean([base for _, _, base in supported]).item(),
    )


def level_targets(
    cell: CellResults,
    subject: OperatingMethod,
    reference: OperatingMethod,
    level: FprLevel,
) -> tuple[TargetLevel, ...]:
    return tuple(
        reading
        for results in cell.targets.values()
        if (reading := target_level(results, subject, reference, level, cell.alpha)) is not None
    )


def has_curves(cell: CellResults, subject: OperatingMethod, reference: OperatingMethod) -> bool:
    return any(
        curve_of(result, subject) is not None and curve_of(result, reference) is not None
        for results in cell.targets.values()
        for result in results
    )


def level_interval(
    readings: tuple[TargetLevel, ...], settings: ReportingSettings
) -> RateConfidenceInterval | None:
    if not readings:
        return None
    return stratified_paired_interval(
        tuple(TargetPairedDifference(item.target, item.gain) for item in readings), settings
    )


def _interval_texts(interval: RateConfidenceInterval | None) -> tuple[DisplayText, DisplayText]:
    if interval is None:
        return DisplayText(""), DisplayText("")
    return number(interval.lower_bound), number(interval.upper_bound)


def _mean_or_none(values: list[RateDifference]) -> DisplayText:
    return optional_number(np.mean(values).item() if values else None)


def _level_row(
    cell: CellResults,
    reference: OperatingMethod,
    level: FprLevel,
    settings: ReportingSettings,
) -> tuple[DisplayText, ...]:
    readings = level_targets(cell, OperatingMethod.PACE, reference, level)
    lower, upper = _interval_texts(level_interval(readings, settings))
    return (
        DisplayText(f"{cell.alpha.fraction}"),
        count(cell.rows),
        reference.display_label,
        level.display_label,
        count(len(cell.targets)),
        count(len(readings)),
        number(len(readings) / len(cell.targets)),
        _mean_or_none([item.subject_detection for item in readings]),
        _mean_or_none([item.reference_detection for item in readings]),
        _mean_or_none([item.gain for item in readings]),
        lower,
        upper,
    )


def _comparison_rows(inputs: ReportInputs) -> tuple[tuple[DisplayText, ...], ...]:
    settings = inputs.configuration.reporting
    return tuple(
        _level_row(cell, reference, level, settings)
        for reference in comparison_references(inputs)
        for cell in cell_results(inputs)
        if cell.kind is BudgetAnalysisKind.UNIVERSAL_PRIMARY
        and has_curves(cell, OperatingMethod.PACE, reference)
        for level in FprLevel
    )


def fixed_fpr_comparison_table(inputs: ReportInputs) -> ResultTable:
    return ResultTable(
        TableName.FIXED_FPR_COMPARISON,
        (
            ColumnHeading.ALPHA,
            ColumnHeading.LOCAL_ROWS,
            ColumnHeading.COMPARATOR,
            ColumnHeading.FPR_LEVEL,
            ColumnHeading.TARGETS,
            ColumnHeading.MATCHED_TARGETS,
            ColumnHeading.COVERAGE,
            ColumnHeading.SUBJECT_DETECTION,
            ColumnHeading.REFERENCE_DETECTION,
            ColumnHeading.MATCHED_DETECTION_DIFFERENCE,
            ColumnHeading.MATCHED_LOWER_BOUND,
            ColumnHeading.MATCHED_UPPER_BOUND,
        ),
        _comparison_rows(inputs),
    )


def _mean_of(values: list[RateDifference]) -> DisplayText:
    return optional_number(np.mean(values).item() if values else None)


def _decomposition_row(
    reference: OperatingMethod,
    alpha: AlphaLevel,
    rows: LocalTrainingRowCount,
    comparisons: tuple[TargetComparison, ...],
) -> tuple[DisplayText, ...]:
    forward = {
        item.target: gain
        for item in comparisons
        if (gain := item.matched_difference(MatchDirection.SUBJECT_AT_REFERENCE)) is not None
    }
    reverse = {
        item.target: gain
        for item in comparisons
        if (gain := item.matched_difference(MatchDirection.REFERENCE_AT_SUBJECT)) is not None
    }
    common = [target for target in forward if target in reverse]
    unmatched = [target for target in reverse if target not in forward]
    return (
        DisplayText(f"{alpha.fraction}"),
        count(rows),
        reference.display_label,
        count(len(comparisons)),
        count(len(forward)),
        count(len(reverse)),
        count(len(common)),
        _mean_of([forward[target] for target in common]),
        _mean_of([reverse[target] for target in common]),
        _mean_of([reverse[target] for target in unmatched]),
        _mean_of([item.reference_false_alert() for item in comparisons if item.target in forward]),
        _mean_of(
            [item.reference_false_alert() for item in comparisons if item.target not in forward]
        ),
    )


def direction_decomposition_table(inputs: ReportInputs) -> ResultTable:
    rows = tuple(
        _decomposition_row(reference, cell.alpha, cell.rows, cell.comparisons)
        for reference in comparison_references(inputs)
        for cell in primary_comparisons(inputs, reference)
    )
    return ResultTable(
        TableName.DIRECTION_DECOMPOSITION,
        (
            ColumnHeading.ALPHA,
            ColumnHeading.LOCAL_ROWS,
            ColumnHeading.COMPARATOR,
            ColumnHeading.TARGETS,
            ColumnHeading.FORWARD_TARGETS,
            ColumnHeading.REVERSE_TARGETS,
            ColumnHeading.COMMON_TARGETS,
            ColumnHeading.FORWARD_COMMON_GAIN,
            ColumnHeading.REVERSE_COMMON_GAIN,
            ColumnHeading.REVERSE_UNMATCHED_GAIN,
            ColumnHeading.MATCHED_REFERENCE_FALSE_ALERT,
            ColumnHeading.UNMATCHED_REFERENCE_FALSE_ALERT,
        ),
        rows,
    )
