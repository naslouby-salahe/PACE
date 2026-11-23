from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from pace.config import ReportingSettings
from pace.reporting.fixed_fpr import CellResults, TargetLevel, level_interval, level_targets
from pace.reporting.operating import curve_of, point_at, unique_results
from pace.reporting.records import (
    ReportInputs,
    ResultTable,
    alpha_text,
    analysis_text,
    count,
    mean_of,
    number,
    optional_number,
)
from pace.reporting.text import slash_list
from pace.types import (
    AcceptanceDesign,
    AcceptanceRule,
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
    ObservationCount,
    OperatingMethod,
    RateConfidenceInterval,
    RateDifference,
    ScalingCohort,
    SeedRateDifference,
    StudyStratum,
    TableName,
    TargetPerformance,
)


@dataclass(frozen=True, slots=True)
class StratumContribution:
    stratum: StudyStratum
    targets: ObservationCount
    gain: RateDifference


@dataclass(frozen=True, slots=True)
class ScalingPoint:
    kind: BudgetAnalysisKind
    alpha: AlphaLevel
    rows: LocalTrainingRowCount
    cohort: ScalingCohort
    target_count: ObservationCount
    readings: tuple[TargetLevel, ...]
    pace_false_alert: FalseAlertRate | None
    comparator_false_alert: FalseAlertRate | None
    interval: RateConfidenceInterval | None

    @property
    def supported(self) -> ObservationCount:
        return len(self.readings)

    @property
    def gain(self) -> RateDifference | None:
        return np.mean([item.gain for item in self.readings]).item() if self.readings else None

    @property
    def pace_detection(self) -> DetectionRate | None:
        if not self.readings:
            return None
        return np.mean([item.subject_detection for item in self.readings]).item()

    @property
    def comparator_detection(self) -> DetectionRate | None:
        if not self.readings:
            return None
        return np.mean([item.reference_detection for item in self.readings]).item()

    @property
    def harmed(self) -> ObservationCount:
        limit = AcceptanceRule.HARMED_TARGET_LOSS.threshold
        return sum(item.gain < -limit for item in self.readings)

    @property
    def seed_gains(self) -> tuple[SeedRateDifference, ...]:
        by_seed: dict[LearningSeed, list[RateDifference]] = defaultdict(list)
        for reading in self.readings:
            for item in reading.seed_gains:
                by_seed[item.seed].append(item.difference)
        return tuple(
            SeedRateDifference(seed, np.mean(by_seed[seed]).item()) for seed in sorted(by_seed)
        )

    @property
    def gain_without_excluded(self) -> RateDifference | None:
        kept = [
            item.gain
            for item in self.readings
            if item.target.dataset is not AcceptanceDesign.PROTOCOL.excluded_dataset
        ]
        return np.mean(kept).item() if kept else None

    @property
    def strata(self) -> tuple[StratumContribution, ...]:
        by_stratum: dict[StudyStratum, list[RateDifference]] = defaultdict(list)
        for reading in self.readings:
            by_stratum[reading.stratum].append(reading.gain)
        return tuple(
            StratumContribution(stratum, len(gains), np.mean(gains).item())
            for stratum, gains in sorted(by_stratum.items(), key=lambda item: item[0].name)
        )


def _nominal_rate(
    results: list[TargetPerformance], method: OperatingMethod, alpha: AlphaLevel
) -> FalseAlertRate:
    rates: list[FalseAlertRate] = []
    for result in results:
        curve = curve_of(result, method)
        if curve is None:
            raise ValueError(f"A result has no {method.name} curve")
        rates.append(point_at(curve, alpha.fraction).false_alert_rate)
    return np.mean(rates).item()


def _point(
    cell: CellResults,
    readings: tuple[TargetLevel, ...],
    cohort: ScalingCohort,
    settings: ReportingSettings,
) -> ScalingPoint:
    members = [cell.targets[reading.target] for reading in readings]
    pace = [_nominal_rate(results, OperatingMethod.PACE, cell.alpha) for results in members]
    reference = [
        _nominal_rate(results, OperatingMethod.MEAN_ENSEMBLE_MATCHED, cell.alpha)
        for results in members
    ]
    return ScalingPoint(
        cell.kind,
        cell.alpha,
        cell.rows,
        cohort,
        len(cell.targets),
        readings,
        mean_of(tuple(pace)) if pace else None,
        mean_of(tuple(reference)) if reference else None,
        level_interval(readings, settings),
    )


def _supported(cell: CellResults) -> dict[DatasetClientId, TargetLevel]:
    return {
        reading.target: reading
        for reading in level_targets(
            cell, OperatingMethod.PACE, OperatingMethod.MEAN_ENSEMBLE_MATCHED, FprLevel.ALPHA
        )
    }


def common_targets(
    supported: Mapping[LocalTrainingRowCount, Mapping[DatasetClientId, TargetLevel]],
) -> frozenset[DatasetClientId]:
    shared: frozenset[DatasetClientId] | None = None
    for item in supported.values():
        keys = frozenset(item)
        shared = keys if shared is None else shared & keys
    return frozenset() if shared is None else shared


def scaling_points(inputs: ReportInputs) -> tuple[ScalingPoint, ...]:
    grouped = unique_results(inputs.performances)
    settings = inputs.configuration.reporting
    points: list[ScalingPoint] = []
    for kind in BudgetAnalysisKind:
        cohort = inputs.cohort(kind)
        for alpha in inputs.alphas():
            cells = {
                rows: CellResults(
                    kind,
                    alpha,
                    rows,
                    {
                        target: results
                        for target, results in sorted(
                            grouped[(alpha, rows)].items(), key=lambda item: item[0].name
                        )
                        if target in cohort
                    },
                )
                for rows in inputs.budgets(kind)
                if (alpha, rows) in grouped
            }
            cells = {rows: cell for rows, cell in cells.items() if cell.targets}
            supported = {rows: _supported(cell) for rows, cell in cells.items()}
            common = common_targets(supported)
            for rows, cell in cells.items():
                available = tuple(supported[rows].values())
                points.append(_point(cell, available, ScalingCohort.AVAILABLE, settings))
                shared = tuple(item for item in available if item.target in common)
                points.append(_point(cell, shared, ScalingCohort.COMMON, settings))
    return tuple(points)


def _seed_text(gains: tuple[SeedRateDifference, ...]) -> DisplayText:
    return slash_list(DisplayText(f"{item.seed}: {item.difference:+.4f}") for item in gains)


def scaling_table(inputs: ReportInputs) -> ResultTable:
    return ResultTable(
        TableName.SCALING,
        (
            ColumnHeading.ANALYSIS,
            ColumnHeading.ALPHA,
            ColumnHeading.LOCAL_ROWS,
            ColumnHeading.COHORT,
            ColumnHeading.TARGETS,
            ColumnHeading.SUPPORTED_TARGETS,
            ColumnHeading.PACE_DETECTION,
            ColumnHeading.MEAN_ENSEMBLE_DETECTION,
            ColumnHeading.PACE_FALSE_ALERT,
            ColumnHeading.MEAN_ENSEMBLE_FALSE_ALERT,
            ColumnHeading.GAIN,
            ColumnHeading.LOWER,
            ColumnHeading.UPPER,
            ColumnHeading.GAIN_WITHOUT_EXCLUDED,
            ColumnHeading.HARMED_TARGETS,
            ColumnHeading.SMALLEST_SEED_GAIN,
            ColumnHeading.SEED_GAINS,
        ),
        tuple(
            (
                analysis_text(point.kind),
                alpha_text(point.alpha),
                count(point.rows),
                point.cohort.label,
                count(point.target_count),
                count(point.supported),
                optional_number(point.pace_detection),
                optional_number(point.comparator_detection),
                optional_number(point.pace_false_alert),
                optional_number(point.comparator_false_alert),
                optional_number(point.gain),
                optional_number(None if point.interval is None else point.interval.lower_bound),
                optional_number(None if point.interval is None else point.interval.upper_bound),
                optional_number(point.gain_without_excluded),
                count(point.harmed),
                optional_number(min((item.difference for item in point.seed_gains), default=None)),
                _seed_text(point.seed_gains),
            )
            for point in scaling_points(inputs)
        ),
    )


def scaling_strata_table(inputs: ReportInputs) -> ResultTable:
    return ResultTable(
        TableName.SCALING_STRATA,
        (
            ColumnHeading.ANALYSIS,
            ColumnHeading.ALPHA,
            ColumnHeading.LOCAL_ROWS,
            ColumnHeading.COHORT,
            ColumnHeading.STRATUM,
            ColumnHeading.SUPPORTED_TARGETS,
            ColumnHeading.GAIN,
        ),
        tuple(
            (
                analysis_text(point.kind),
                alpha_text(point.alpha),
                count(point.rows),
                point.cohort.label,
                DisplayText(contribution.stratum.name),
                count(contribution.targets),
                number(contribution.gain),
            )
            for point in scaling_points(inputs)
            for contribution in point.strata
        ),
    )
