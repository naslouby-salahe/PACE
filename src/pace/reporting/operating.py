from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from pace.config import ReportingSettings
from pace.reporting.metrics import stratified_paired_interval
from pace.reporting.records import (
    ReportInputs,
    ResultTable,
    VariantPerformances,
    count,
    number,
    optional_number,
)
from pace.types import (
    AlphaLevel,
    BudgetAnalysisKind,
    ColumnHeading,
    DatasetClientId,
    DetectionRate,
    DisplayText,
    ExternalBaseline,
    FalseAlarmBudget,
    FalseAlertRate,
    LearningSeed,
    LocalTrainingRowCount,
    MatchClass,
    MatchDirection,
    ObservationCount,
    OperatingCurve,
    OperatingMethod,
    OperatingPoint,
    RateConfidenceInterval,
    RateDifference,
    StudyStratum,
    TableName,
    TargetPairedDifference,
    TargetPerformance,
)


@dataclass(frozen=True, slots=True)
class CurveReading:
    match: MatchClass
    detection: DetectionRate | None
    error: FalseAlertRate | None


def point_detection(point: OperatingPoint) -> DetectionRate:
    return np.mean(point.detection_rates).item()


def point_at(curve: OperatingCurve, budget: FalseAlarmBudget) -> OperatingPoint:
    return next(point for point in curve.points if np.isclose(point.nominal_budget, budget))


def read_curve(
    curve: OperatingCurve, false_alert_rate: FalseAlertRate, tolerance: FalseAlertRate
) -> CurveReading:
    ordered = sorted((point.false_alert_rate, point_detection(point)) for point in curve.points)
    rates = np.asarray([rate for rate, _ in ordered], dtype=np.float64)
    envelope = np.maximum.accumulate(np.asarray([value for _, value in ordered], dtype=np.float64))
    nearest = np.argmin(np.abs(rates - false_alert_rate)).item()
    gap = abs(rates[nearest] - false_alert_rate).item()
    if gap <= tolerance:
        return CurveReading(MatchClass.EXACT, envelope[nearest].item(), gap)
    if rates[0] < false_alert_rate < rates[-1]:
        return CurveReading(
            MatchClass.INTERPOLATED, np.interp(false_alert_rate, rates, envelope).item(), 0.0
        )
    return CurveReading(MatchClass.UNAVAILABLE, None, None)


def curve_of(result: TargetPerformance, method: OperatingMethod) -> OperatingCurve | None:
    return next((curve for curve in result.curves if curve.method is method), None)


@dataclass(frozen=True, slots=True)
class PairReading:
    subject_detection: DetectionRate
    subject_false_alert: FalseAlertRate
    reference_detection: DetectionRate
    reference_false_alert: FalseAlertRate
    subject_at_reference: CurveReading
    reference_at_subject: CurveReading

    def matched_difference(self, direction: MatchDirection) -> RateDifference | None:
        match direction:
            case MatchDirection.SUBJECT_AT_REFERENCE:
                reading = self.subject_at_reference
                return (
                    None
                    if reading.detection is None
                    else reading.detection - self.reference_detection
                )
            case MatchDirection.REFERENCE_AT_SUBJECT:
                reading = self.reference_at_subject
                return (
                    None
                    if reading.detection is None
                    else self.subject_detection - reading.detection
                )

    def reading(self, direction: MatchDirection) -> CurveReading:
        match direction:
            case MatchDirection.SUBJECT_AT_REFERENCE:
                return self.subject_at_reference
            case MatchDirection.REFERENCE_AT_SUBJECT:
                return self.reference_at_subject


def pair_reading(
    result: TargetPerformance,
    subject: OperatingMethod,
    reference: OperatingMethod,
    budget: FalseAlarmBudget,
) -> PairReading | None:
    subject_curve, reference_curve = curve_of(result, subject), curve_of(result, reference)
    if subject_curve is None or reference_curve is None:
        return None
    subject_point, reference_point = (
        point_at(subject_curve, budget),
        point_at(reference_curve, budget),
    )
    tolerance = 0.5 / result.benign_test_rows
    return PairReading(
        point_detection(subject_point),
        subject_point.false_alert_rate,
        point_detection(reference_point),
        reference_point.false_alert_rate,
        read_curve(subject_curve, reference_point.false_alert_rate, tolerance),
        read_curve(reference_curve, subject_point.false_alert_rate, tolerance),
    )


@dataclass(frozen=True, slots=True)
class TargetComparison:
    target: DatasetClientId
    stratum: StudyStratum
    readings: tuple[PairReading, ...]

    def nominal_difference(self) -> RateDifference:
        return np.mean(
            [item.subject_detection - item.reference_detection for item in self.readings]
        ).item()

    def false_alert_difference(self) -> RateDifference:
        return np.mean(
            [item.subject_false_alert - item.reference_false_alert for item in self.readings]
        ).item()

    def classes(self, direction: MatchDirection) -> tuple[MatchClass, ...]:
        return tuple(item.reading(direction).match for item in self.readings)

    def matched_difference(self, direction: MatchDirection) -> RateDifference | None:
        values = [item.matched_difference(direction) for item in self.readings]
        if any(value is None for value in values):
            return None
        return np.mean([value for value in values if value is not None]).item()

    def matching_error(self, direction: MatchDirection) -> FalseAlertRate | None:
        errors = [item.reading(direction).error for item in self.readings]
        if any(error is None for error in errors):
            return None
        return np.mean([error for error in errors if error is not None]).item()

    def reference_false_alert(self) -> FalseAlertRate:
        return np.mean([item.reference_false_alert for item in self.readings]).item()


@dataclass(frozen=True, slots=True)
class ComparisonCell:
    alpha: AlphaLevel
    rows: LocalTrainingRowCount
    reference: OperatingMethod
    comparisons: tuple[TargetComparison, ...]

    def stratum(self, stratum: StudyStratum) -> tuple[TargetComparison, ...]:
        return tuple(item for item in self.comparisons if item.stratum is stratum)


def unique_results(
    performances: tuple[VariantPerformances, ...],
) -> dict[tuple[AlphaLevel, LocalTrainingRowCount], dict[DatasetClientId, list[TargetPerformance]]]:
    grouped: dict[
        tuple[AlphaLevel, LocalTrainingRowCount],
        dict[DatasetClientId, dict[LearningSeed, TargetPerformance]],
    ] = defaultdict(lambda: defaultdict(dict))
    for variant in performances:
        for result in variant.results:
            grouped[(variant.alpha, variant.rows)][result.target].setdefault(result.seed, result)
    return {
        key: {target: list(by_seed.values()) for target, by_seed in targets.items()}
        for key, targets in grouped.items()
    }


def primary_comparisons(
    inputs: ReportInputs,
    reference: OperatingMethod,
    subject: OperatingMethod = OperatingMethod.PACE,
) -> tuple[ComparisonCell, ...]:
    cohort = inputs.cohort(BudgetAnalysisKind.UNIVERSAL_PRIMARY)
    budgets = inputs.budgets(BudgetAnalysisKind.UNIVERSAL_PRIMARY)
    cells: list[ComparisonCell] = []
    for (alpha, rows), targets in sorted(
        unique_results(inputs.performances).items(),
        key=lambda item: (item[0][1], item[0][0].fraction),
    ):
        if rows not in budgets:
            continue
        comparisons: list[TargetComparison] = []
        for target, results in sorted(targets.items(), key=lambda item: item[0].name):
            if target not in cohort:
                continue
            readings = [
                pair_reading(result, subject, reference, alpha.fraction) for result in results
            ]
            if all(reading is not None for reading in readings):
                comparisons.append(
                    TargetComparison(
                        target,
                        results[0].study_stratum,
                        tuple(reading for reading in readings if reading is not None),
                    )
                )
        if comparisons:
            cells.append(ComparisonCell(alpha, rows, reference, tuple(comparisons)))
    return tuple(cells)


def comparison_references(inputs: ReportInputs) -> tuple[OperatingMethod, ...]:
    baselines = tuple(
        baseline.operating_method for baseline in inputs.configuration.scientific.baselines.methods
    )
    variants = tuple(
        variant.operating_method for variant in inputs.configuration.scientific.curve_variants
    )
    return (
        OperatingMethod.MEAN_ENSEMBLE,
        OperatingMethod.MEAN_ENSEMBLE_MATCHED,
        *variants,
        *baselines,
    )


def _interval(
    differences: tuple[TargetPairedDifference, ...], settings: ReportingSettings
) -> RateConfidenceInterval | None:
    return stratified_paired_interval(differences, settings) if differences else None


def _interval_cells(
    interval: RateConfidenceInterval | None,
) -> tuple[DisplayText, DisplayText]:
    if interval is None:
        return DisplayText(""), DisplayText("")
    return number(interval.lower_bound), number(interval.upper_bound)


def _mean_text(values: tuple[RateDifference, ...]) -> DisplayText:
    return optional_number(np.mean(values).item() if values else None)


def matched_comparison_table(inputs: ReportInputs) -> ResultTable:
    settings = inputs.configuration.reporting
    rows: list[tuple[DisplayText, ...]] = []
    for reference in comparison_references(inputs):
        for cell in primary_comparisons(inputs, reference):
            nominal = tuple(
                TargetPairedDifference(target=item.target, difference=item.nominal_difference())
                for item in cell.comparisons
            )
            nominal_interval = _interval(nominal, settings)
            for direction in MatchDirection:
                matched = tuple(
                    TargetPairedDifference(target=item.target, difference=difference)
                    for item in cell.comparisons
                    if (difference := item.matched_difference(direction)) is not None
                )
                interval = _interval(matched, settings)
                errors = tuple(
                    error
                    for item in cell.comparisons
                    if (error := item.matching_error(direction)) is not None
                )
                rows.append(
                    (
                        DisplayText(f"{cell.alpha.fraction}"),
                        count(cell.rows),
                        reference.display_label,
                        direction.display_label,
                        count(len(cell.comparisons)),
                        number(
                            np.mean([item.nominal_difference() for item in cell.comparisons]).item()
                        ),
                        *_interval_cells(nominal_interval),
                        number(
                            np.mean(
                                [item.false_alert_difference() for item in cell.comparisons]
                            ).item()
                        ),
                        count(len(matched)),
                        _mean_text(tuple(item.difference for item in matched)),
                        *_interval_cells(interval),
                        _mean_text(errors),
                    )
                )
    return ResultTable(
        TableName.MATCHED_COMPARISON,
        (
            ColumnHeading.ALPHA,
            ColumnHeading.LOCAL_ROWS,
            ColumnHeading.COMPARATOR,
            ColumnHeading.DIRECTION,
            ColumnHeading.TARGETS,
            ColumnHeading.DETECTION_DIFFERENCE,
            ColumnHeading.NOMINAL_LOWER,
            ColumnHeading.NOMINAL_UPPER,
            ColumnHeading.FALSE_ALERT_DIFFERENCE,
            ColumnHeading.MATCHED_TARGETS,
            ColumnHeading.MATCHED_DETECTION_DIFFERENCE,
            ColumnHeading.MATCHED_LOWER_BOUND,
            ColumnHeading.MATCHED_UPPER_BOUND,
            ColumnHeading.MEAN_MATCH_ERROR,
        ),
        tuple(rows),
    )


def _class_count(
    items: tuple[TargetComparison, ...], direction: MatchDirection, kind: MatchClass
) -> ObservationCount:
    return sum(item.classes(direction).count(kind) for item in items)


def operating_coverage_table(inputs: ReportInputs) -> ResultTable:
    rows: list[tuple[DisplayText, ...]] = []
    for reference in comparison_references(inputs):
        for cell in primary_comparisons(inputs, reference):
            for direction in MatchDirection:
                for stratum in StudyStratum:
                    items = cell.stratum(stratum)
                    if not items:
                        continue
                    matched = tuple(
                        item for item in items if item.matched_difference(direction) is not None
                    )
                    unmatched = tuple(item for item in items if item not in matched)
                    rows.append(
                        (
                            DisplayText(f"{cell.alpha.fraction}"),
                            count(cell.rows),
                            reference.display_label,
                            direction.display_label,
                            stratum.display_label,
                            count(_class_count(items, direction, MatchClass.EXACT)),
                            count(_class_count(items, direction, MatchClass.INTERPOLATED)),
                            count(_class_count(items, direction, MatchClass.UNAVAILABLE)),
                            count(len(matched)),
                            count(len(unmatched)),
                            number(len(matched) / len(items)),
                            _mean_text(tuple(item.reference_false_alert() for item in matched)),
                            _mean_text(tuple(item.reference_false_alert() for item in unmatched)),
                        )
                    )
    return ResultTable(
        TableName.OPERATING_COVERAGE,
        (
            ColumnHeading.ALPHA,
            ColumnHeading.LOCAL_ROWS,
            ColumnHeading.COMPARATOR,
            ColumnHeading.DIRECTION,
            ColumnHeading.STRATUM,
            ColumnHeading.EXACT_CELLS,
            ColumnHeading.INTERPOLATED_CELLS,
            ColumnHeading.UNAVAILABLE_CELLS,
            ColumnHeading.MATCHED_TARGETS,
            ColumnHeading.UNMATCHED_TARGETS,
            ColumnHeading.COVERAGE,
            ColumnHeading.MATCHED_REFERENCE_FALSE_ALERT,
            ColumnHeading.UNMATCHED_REFERENCE_FALSE_ALERT,
        ),
        tuple(rows),
    )


@dataclass(frozen=True, slots=True)
class MeanOperatingPoint:
    rows: LocalTrainingRowCount
    method: OperatingMethod
    budget: FalseAlarmBudget
    targets: ObservationCount
    false_alert_rate: FalseAlertRate
    detection_rate: DetectionRate


def _gather_operating_result(
    gathered: dict[
        tuple[LocalTrainingRowCount, OperatingMethod, FalseAlarmBudget],
        list[tuple[DatasetClientId, FalseAlertRate, DetectionRate]],
    ],
    rows: LocalTrainingRowCount,
    target: DatasetClientId,
    result: TargetPerformance,
) -> None:
    for curve in result.curves:
        for point in curve.points:
            gathered[(rows, curve.method, point.nominal_budget)].append(
                (target, point.false_alert_rate, point_detection(point))
            )


def mean_operating_points(inputs: ReportInputs) -> tuple[MeanOperatingPoint, ...]:
    cohort = inputs.cohort(BudgetAnalysisKind.UNIVERSAL_PRIMARY)
    budgets = inputs.budgets(BudgetAnalysisKind.UNIVERSAL_PRIMARY)
    gathered: dict[
        tuple[LocalTrainingRowCount, OperatingMethod, FalseAlarmBudget],
        list[tuple[DatasetClientId, FalseAlertRate, DetectionRate]],
    ] = defaultdict(list)
    for (alpha, rows), targets in unique_results(inputs.performances).items():
        if rows not in budgets or alpha is not AlphaLevel.ALPHA_005:
            continue
        for target, seeds in targets.items():
            if target not in cohort:
                continue
            for result in seeds:
                _gather_operating_result(gathered, rows, target, result)
    return tuple(
        MeanOperatingPoint(
            rows,
            method,
            budget,
            len({target for target, _, _ in values}),
            np.mean([rate for _, rate, _ in values]).item(),
            np.mean([detection for _, _, detection in values]).item(),
        )
        for (rows, method, budget), values in sorted(
            gathered.items(), key=lambda item: (item[0][0], item[0][1].name, item[0][2])
        )
    )


def operating_points_table(inputs: ReportInputs) -> ResultTable:
    rows_out = tuple(
        (
            count(point.rows),
            point.method.display_label,
            DisplayText(f"{point.budget}"),
            count(point.targets),
            number(point.false_alert_rate),
            number(point.detection_rate),
        )
        for point in mean_operating_points(inputs)
    )
    return ResultTable(
        TableName.OPERATING_POINTS,
        (
            ColumnHeading.LOCAL_ROWS,
            ColumnHeading.METHOD,
            ColumnHeading.NOMINAL_BUDGET,
            ColumnHeading.TARGETS,
            ColumnHeading.FALSE_ALERT_RATE,
            ColumnHeading.DETECTION_RATE,
        ),
        rows_out,
    )


def baselines_table(inputs: ReportInputs) -> ResultTable:
    strata = tuple(item.stratum for item in inputs.evidence)
    rows = tuple(
        (
            baseline.display_label,
            stratum.display_label,
            baseline.fidelity(stratum).display_label,
            baseline.deviation_note,
        )
        for baseline in ExternalBaseline
        if baseline in inputs.configuration.scientific.baselines.methods
        for stratum in strata
    )
    return ResultTable(
        TableName.BASELINES,
        (
            ColumnHeading.COMPARATOR,
            ColumnHeading.STRATUM,
            ColumnHeading.FIDELITY,
            ColumnHeading.DEVIATION,
        ),
        rows,
    )
