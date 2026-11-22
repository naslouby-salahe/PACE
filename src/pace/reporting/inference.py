from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from pace.config import ReportingSettings
from pace.reporting.fixed_fpr import CellResults, TargetLevel, cell_results, level_targets
from pace.reporting.metrics import stratified_paired_interval
from pace.reporting.operating import curve_of
from pace.reporting.records import ReportInputs, ResultTable, alpha_text, count, number
from pace.types import (
    AlphaLevel,
    BootstrapPValue,
    BudgetAnalysisKind,
    ColumnHeading,
    CoverageCause,
    CriterionSatisfied,
    DatasetName,
    DisplayText,
    FalseAlertRate,
    FloatArray,
    FloatMatrix,
    FprLevel,
    InferenceMethod,
    ObservationCount,
    OperatingCurve,
    OperatingMethod,
    RateConfidenceInterval,
    RateDifference,
    RateReach,
    TableName,
    TargetPairedDifference,
    TargetPerformance,
)


@dataclass(frozen=True, slots=True)
class InferenceReading:
    scope: DisplayText
    method: InferenceMethod
    target_count: ObservationCount
    estimate: RateDifference
    interval: RateConfidenceInterval
    adjusted_p_value: BootstrapPValue | None = None


def dataset_groups(
    differences: tuple[TargetPairedDifference, ...],
) -> dict[DatasetName, FloatArray]:
    grouped: dict[DatasetName, list[RateDifference]] = defaultdict(list)
    for item in sorted(differences, key=lambda entry: entry.target.name):
        grouped[item.target.dataset].append(item.difference)
    return {
        dataset: np.asarray(values, dtype=np.float64)
        for dataset, values in sorted(grouped.items(), key=lambda entry: entry[0].name)
    }


def _within_means(
    groups: tuple[FloatArray, ...], settings: ReportingSettings, generator: np.random.Generator
) -> FloatArray:
    return np.stack(
        [
            np.mean(
                values[
                    generator.integers(
                        0, values.size, size=(settings.bootstrap_replicates, values.size)
                    )
                ],
                axis=1,
            )
            for values in groups
        ]
    )


def _slot_means(
    values: FloatArray,
    settings: ReportingSettings,
    slots: ObservationCount,
    generator: np.random.Generator,
) -> FloatArray:
    return np.asarray(
        np.mean(
            values[
                generator.integers(
                    0, values.size, size=(settings.bootstrap_replicates, slots, values.size)
                )
            ],
            axis=2,
        ),
        dtype=np.float64,
    )


def cluster_bootstrap_estimates(
    groups: tuple[FloatArray, ...], settings: ReportingSettings, method: InferenceMethod
) -> FloatArray:
    generator = np.random.default_rng(settings.bootstrap_seed)
    slots = len(groups)
    means = np.stack([_slot_means(values, settings, slots, generator) for values in groups])
    sizes = np.asarray([values.size for values in groups], dtype=np.float64)
    drawn = generator.integers(0, slots, size=(settings.bootstrap_replicates, slots))
    picked_means: FloatMatrix = np.take_along_axis(means, drawn[np.newaxis], axis=0)[0]
    if method is InferenceMethod.DATASET_MEANS:
        return np.asarray(np.mean(picked_means, axis=1), dtype=np.float64)
    weights = sizes[drawn]
    return np.asarray(
        np.sum(picked_means * weights, axis=1) / np.sum(weights, axis=1), dtype=np.float64
    )


def within_dataset_estimates(values: FloatArray, settings: ReportingSettings) -> FloatArray:
    generator = np.random.default_rng(settings.bootstrap_seed)
    return np.asarray(_within_means((values,), settings, generator)[0], dtype=np.float64)


def estimates_interval(
    estimates: FloatArray, settings: ReportingSettings
) -> RateConfidenceInterval:
    tail = (1.0 - settings.confidence_level) / 2.0
    lower, upper = np.quantile(estimates, (tail, 1.0 - tail), method="linear").tolist()
    return RateConfidenceInterval(
        lower_bound=lower, upper_bound=upper, confidence_level=settings.confidence_level
    )


def bootstrap_p_value(estimates: FloatArray) -> BootstrapPValue:
    replicates = estimates.size
    below = np.sum(estimates <= 0.0).item()
    above = np.sum(estimates >= 0.0).item()
    return min(1.0, 2.0 * (min(below, above) + 1) / (replicates + 1))


def holm_adjusted(p_values: tuple[BootstrapPValue, ...]) -> tuple[BootstrapPValue, ...]:
    order = sorted(range(len(p_values)), key=lambda index: p_values[index])
    adjusted = [0.0] * len(p_values)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(p_values) - rank) * p_values[index]))
        adjusted[index] = running
    return tuple(adjusted)


def _differences(readings: tuple[TargetLevel, ...]) -> tuple[TargetPairedDifference, ...]:
    return tuple(TargetPairedDifference(item.target, item.gain) for item in readings)


def _pooled_readings(
    differences: tuple[TargetPairedDifference, ...],
    settings: ReportingSettings,
    scope: DisplayText,
) -> tuple[InferenceReading, ...]:
    groups = tuple(dataset_groups(differences).values())
    return (
        InferenceReading(
            scope,
            InferenceMethod.STRATIFIED_TARGETS,
            len(differences),
            np.mean([item.difference for item in differences]).item(),
            stratified_paired_interval(differences, settings),
        ),
        InferenceReading(
            scope,
            InferenceMethod.DATASET_CLUSTERS,
            len(differences),
            np.mean([item.difference for item in differences]).item(),
            estimates_interval(
                cluster_bootstrap_estimates(groups, settings, InferenceMethod.DATASET_CLUSTERS),
                settings,
            ),
        ),
        InferenceReading(
            scope,
            InferenceMethod.DATASET_MEANS,
            len(differences),
            np.mean([np.mean(values).item() for values in groups]).item(),
            estimates_interval(
                cluster_bootstrap_estimates(groups, settings, InferenceMethod.DATASET_MEANS),
                settings,
            ),
        ),
    )


def _zero_filled(
    cell: CellResults, readings: tuple[TargetLevel, ...]
) -> tuple[TargetPairedDifference, ...]:
    supported = {item.target: item.gain for item in readings}
    return tuple(
        TargetPairedDifference(target, supported.get(target, 0.0)) for target in cell.targets
    )


def _per_dataset_readings(
    differences: tuple[TargetPairedDifference, ...], settings: ReportingSettings
) -> tuple[InferenceReading, ...]:
    groups = dataset_groups(differences)
    replicates = {
        dataset: within_dataset_estimates(values, settings) for dataset, values in groups.items()
    }
    adjusted = holm_adjusted(tuple(bootstrap_p_value(item) for item in replicates.values()))
    return tuple(
        InferenceReading(
            DisplayText(dataset.name),
            InferenceMethod.WITHIN_DATASET,
            groups[dataset].size,
            np.mean(groups[dataset]).item(),
            estimates_interval(replicates[dataset], settings),
            p_value,
        )
        for dataset, p_value in zip(groups, adjusted, strict=True)
    )


def _left_out_readings(
    differences: tuple[TargetPairedDifference, ...], settings: ReportingSettings
) -> tuple[InferenceReading, ...]:
    datasets = tuple(dataset_groups(differences))
    if len(datasets) < 2:
        return ()
    return tuple(
        InferenceReading(
            DisplayText(f"without {left_out.name}"),
            InferenceMethod.LEAVE_ONE_DATASET_OUT,
            len(kept),
            np.mean([item.difference for item in kept]).item(),
            stratified_paired_interval(kept, settings),
        )
        for left_out in datasets
        if (kept := tuple(item for item in differences if item.target.dataset is not left_out))
    )


def cell_inference(
    cell: CellResults, reference: OperatingMethod, settings: ReportingSettings
) -> tuple[InferenceReading, ...]:
    readings = level_targets(cell, OperatingMethod.PACE, reference, FprLevel.ALPHA)
    if not readings:
        return ()
    differences = _differences(readings)
    padded = _zero_filled(cell, readings)
    return (
        *_pooled_readings(differences, settings, DisplayText("all supported targets")),
        InferenceReading(
            DisplayText("all targets"),
            InferenceMethod.UNSUPPORTED_AS_ZERO,
            len(padded),
            np.mean([item.difference for item in padded]).item(),
            stratified_paired_interval(padded, settings),
        ),
        *_per_dataset_readings(differences, settings),
        *_left_out_readings(differences, settings),
    )


def _p_text(value: BootstrapPValue | None) -> DisplayText:
    return DisplayText("" if value is None else f"{value:.4f}")


def inference_sensitivity_table(inputs: ReportInputs) -> ResultTable:
    settings = inputs.configuration.reporting
    reference = OperatingMethod.MEAN_ENSEMBLE_MATCHED
    rows: list[tuple[DisplayText, ...]] = []
    for cell in cell_results(inputs):
        if cell.kind is not BudgetAnalysisKind.UNIVERSAL_PRIMARY or not _has_reference(
            cell, reference
        ):
            continue
        for reading in cell_inference(cell, reference, settings):
            rows.append(
                (
                    alpha_text(cell.alpha),
                    count(cell.rows),
                    reading.scope,
                    reading.method.display_label,
                    count(reading.target_count),
                    number(reading.estimate),
                    number(reading.interval.lower_bound),
                    number(reading.interval.upper_bound),
                    _p_text(reading.adjusted_p_value),
                )
            )
    return ResultTable(
        TableName.INFERENCE_SENSITIVITY,
        (
            ColumnHeading.ALPHA,
            ColumnHeading.LOCAL_ROWS,
            ColumnHeading.SCOPE,
            ColumnHeading.INFERENCE_METHOD,
            ColumnHeading.TARGETS,
            ColumnHeading.ESTIMATE,
            ColumnHeading.LOWER,
            ColumnHeading.UPPER,
            ColumnHeading.ADJUSTED_P_VALUE,
        ),
        tuple(rows),
    )


def _has_reference(cell: CellResults, reference: OperatingMethod) -> bool:
    return all(
        curve_of(result, reference) is not None
        for results in cell.targets.values()
        for result in results
    )


def _curve_reaches(
    curve: OperatingCurve, rate: FalseAlertRate, tolerance: FalseAlertRate
) -> RateReach:
    rates = [point.false_alert_rate for point in curve.points]
    if min(abs(item - rate) for item in rates) <= tolerance or min(rates) < rate < max(rates):
        return RateReach.WITHIN
    return RateReach.ABOVE if min(rates) > rate else RateReach.BELOW


def coverage_cause(
    results: list[TargetPerformance],
    reference: OperatingMethod,
    alpha: AlphaLevel,
    supported: CriterionSatisfied,
) -> CoverageCause:
    if supported:
        return CoverageCause.SUPPORTED
    rate = alpha.fraction
    floored = False
    resolution_limited = False
    for result in results:
        tolerance = 0.5 / result.benign_test_rows
        for method in (OperatingMethod.PACE, reference):
            curve = curve_of(result, method)
            reach = RateReach.WITHIN if curve is None else _curve_reaches(curve, rate, tolerance)
            floored = floored or reach is RateReach.ABOVE
            resolution_limited = resolution_limited or (
                reach is RateReach.ABOVE and _resolution_floor(result) >= rate
            )
    if resolution_limited:
        return CoverageCause.RESOLUTION_LIMITED
    return CoverageCause.SHIFT_FLOOR if floored else CoverageCause.GRID_CEILING


def _resolution_floor(result: TargetPerformance) -> FalseAlertRate:
    allocation = result.allocation
    if allocation is None:
        return 0.0
    experts = allocation.peer_expert_count + allocation.bank_expert_count + 1
    return min(1.0, experts / (allocation.calibration_rows + 1))


def coverage_diagnosis_table(inputs: ReportInputs) -> ResultTable:
    reference = OperatingMethod.MEAN_ENSEMBLE_MATCHED
    rows: list[tuple[DisplayText, ...]] = []
    for cell in cell_results(inputs):
        if cell.kind is not BudgetAnalysisKind.UNIVERSAL_PRIMARY or not _has_reference(
            cell, reference
        ):
            continue
        supported = {
            item.target
            for item in level_targets(cell, OperatingMethod.PACE, reference, FprLevel.ALPHA)
        }
        causes = [
            coverage_cause(results, reference, cell.alpha, target in supported)
            for target, results in cell.targets.items()
        ]
        for cause in CoverageCause:
            rows.append(
                (
                    alpha_text(cell.alpha),
                    count(cell.rows),
                    cause.display_label,
                    count(sum(item is cause for item in causes)),
                    number(sum(item is cause for item in causes) / len(causes)),
                )
            )
    return ResultTable(
        TableName.COVERAGE_DIAGNOSIS,
        (
            ColumnHeading.ALPHA,
            ColumnHeading.LOCAL_ROWS,
            ColumnHeading.CAUSE,
            ColumnHeading.TARGETS,
            ColumnHeading.COVERAGE,
        ),
        tuple(rows),
    )
