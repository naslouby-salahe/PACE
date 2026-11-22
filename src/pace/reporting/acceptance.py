from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from pace.config import ReportingSettings
from pace.reporting.fixed_fpr import (
    CellResults,
    TargetLevel,
    cell_results,
    direction_decomposition_table,
    fixed_fpr_comparison_table,
    level_interval,
    level_targets,
)
from pace.reporting.inference import coverage_diagnosis_table, inference_sensitivity_table
from pace.reporting.metrics import stratified_paired_interval
from pace.reporting.operating import curve_of, point_at
from pace.reporting.records import (
    ReportInputs,
    ResultTable,
    alpha_text,
    analysis_text,
    count,
    number,
    verdict_text,
)
from pace.types import (
    AcceptanceCalibrationRule,
    AcceptanceCell,
    AcceptanceCriterion,
    AcceptanceDesign,
    AcceptanceMeasure,
    AcceptanceReport,
    AcceptanceRule,
    AlphaLevel,
    ColumnHeading,
    CriterionSatisfied,
    CriterionVerdict,
    DisplayText,
    FalseAlertRate,
    FprLevel,
    LearningSeed,
    LevelGain,
    MarginSensitivity,
    ObservationCount,
    OperatingMethod,
    RateConfidenceInterval,
    RateDifference,
    SeedRateDifference,
    TableName,
    TargetPairedDifference,
    TargetPerformance,
    UnavailableReading,
)


@dataclass(frozen=True, slots=True)
class _FalseAlert:
    difference: RateDifference
    interval: RateConfidenceInterval
    exceedance: RateDifference


def _nominal_rate(
    result: TargetPerformance, method: OperatingMethod, alpha: AlphaLevel
) -> FalseAlertRate:
    curve = curve_of(result, method)
    if curve is None:
        raise ValueError(f"A result has no {method.name} curve")
    return point_at(curve, alpha.fraction).false_alert_rate


def _false_alert(
    cell: CellResults,
    reference: OperatingMethod,
    settings: ReportingSettings,
) -> _FalseAlert:
    limit = AcceptanceRule.EXCEEDANCE_MULTIPLE.threshold * cell.alpha.fraction
    differences: list[TargetPairedDifference] = []
    shown_exceed: ObservationCount = 0
    base_exceed: ObservationCount = 0
    cells: ObservationCount = 0
    for target, results in cell.targets.items():
        shown = [_nominal_rate(item, OperatingMethod.PACE, cell.alpha) for item in results]
        base = [_nominal_rate(item, reference, cell.alpha) for item in results]
        differences.append(
            TargetPairedDifference(target, np.mean(shown).item() - np.mean(base).item())
        )
        shown_exceed += sum(rate > limit for rate in shown)
        base_exceed += sum(rate > limit for rate in base)
        cells += len(shown)
    return _FalseAlert(
        np.mean([item.difference for item in differences]).item(),
        stratified_paired_interval(tuple(differences), settings),
        (shown_exceed - base_exceed) / cells,
    )


def _level_gain(
    level: FprLevel, readings: tuple[TargetLevel, ...], settings: ReportingSettings
) -> LevelGain:
    return LevelGain(
        level=level,
        matched_targets=len(readings),
        gain=np.mean([item.gain for item in readings]).item() if readings else None,
        interval=level_interval(readings, settings),
        subject_detection=(
            np.mean([item.subject_detection for item in readings]).item() if readings else None
        ),
        reference_detection=(
            np.mean([item.reference_detection for item in readings]).item() if readings else None
        ),
    )


def _seed_gains(readings: tuple[TargetLevel, ...]) -> tuple[SeedRateDifference, ...]:
    by_seed: dict[LearningSeed, list[RateDifference]] = defaultdict(list)
    for reading in readings:
        for item in reading.seed_gains:
            by_seed[item.seed].append(item.difference)
    return tuple(
        SeedRateDifference(seed, np.mean(by_seed[seed]).item()) for seed in sorted(by_seed)
    )


def _stratum_left_out_minimum(readings: tuple[TargetLevel, ...]) -> RateDifference | None:
    strata = {item.stratum for item in readings}
    if not strata:
        return None
    remaining = [
        [item.gain for item in readings if len(strata) < 2 or item.stratum is not left_out]
        for left_out in strata
    ]
    return min(np.mean(values).item() for values in remaining)


def _without_excluded(
    readings: tuple[TargetLevel, ...], settings: ReportingSettings
) -> RateConfidenceInterval | None:
    return level_interval(
        tuple(
            item
            for item in readings
            if item.target.dataset is not AcceptanceDesign.PROTOCOL.excluded_dataset
        ),
        settings,
    )


def _verdicts(cell: AcceptanceCell) -> tuple[CriterionVerdict, ...]:
    primary = next(item for item in cell.levels if item.level is FprLevel.ALPHA)
    harmed_share = (
        cell.harmed_target_count / primary.matched_targets if primary.matched_targets else 1.0
    )
    without = cell.gain_without_excluded_interval
    checks = {
        AcceptanceCriterion.FIXED_FPR_GAIN: primary.interval is not None
        and primary.interval.lower_bound > 0.0,
        AcceptanceCriterion.COVERAGE: primary.matched_targets / cell.target_count
        >= AcceptanceCalibrationRule.COVERAGE_FLOOR.threshold,
        AcceptanceCriterion.LOW_FPR_CONSISTENCY: all(
            item.gain is not None and item.gain > 0.0 for item in cell.levels
        ),
        AcceptanceCriterion.GENERALISATION: without is not None and without.lower_bound > 0.0,
        AcceptanceCriterion.STRATUM_ROBUSTNESS: cell.minimum_stratum_left_out_gain is not None
        and cell.minimum_stratum_left_out_gain > 0.0,
        AcceptanceCriterion.SEED_SIGNS: len(cell.seed_gains) > 0
        and min(item.difference for item in cell.seed_gains) > 0.0,
        AcceptanceCriterion.NON_INFERIORITY: cell.severe_target_count == 0
        and harmed_share <= AcceptanceRule.HARMED_TARGET_SHARE.threshold,
        AcceptanceCriterion.FALSE_ALERT_SAFEGUARD: cell.false_alert_interval.upper_bound
        <= AcceptanceCalibrationRule.FALSE_ALERT_MARGIN_SHARE.threshold * cell.alpha.fraction
        and cell.exceedance_increase <= AcceptanceRule.EXCEEDANCE_INCREASE.threshold,
    }
    return tuple(CriterionVerdict(criterion, satisfied) for criterion, satisfied in checks.items())


def _acceptance_cell(
    cell: CellResults, reference: OperatingMethod, settings: ReportingSettings
) -> AcceptanceCell:
    by_level = {
        level: level_targets(cell, OperatingMethod.PACE, reference, level) for level in FprLevel
    }
    primary = by_level[FprLevel.ALPHA]
    gains = np.asarray([item.gain for item in primary], dtype=np.float64)
    false_alert = _false_alert(cell, reference, settings)
    unverified = AcceptanceCell(
        kind=cell.kind,
        alpha=cell.alpha,
        local_training_rows=cell.rows,
        target_count=len(cell.targets),
        levels=tuple(
            _level_gain(level, readings, settings) for level, readings in by_level.items()
        ),
        gain_without_excluded_interval=_without_excluded(primary, settings),
        harmed_target_count=np.sum(gains < -AcceptanceRule.HARMED_TARGET_LOSS.threshold).item(),
        severe_target_count=np.sum(gains < -AcceptanceRule.SEVERE_TARGET_LOSS.threshold).item(),
        worst_target_difference=np.min(gains).item() if gains.size else None,
        false_alert_difference=false_alert.difference,
        false_alert_interval=false_alert.interval,
        exceedance_increase=false_alert.exceedance,
        seed_gains=_seed_gains(primary),
        minimum_stratum_left_out_gain=_stratum_left_out_minimum(primary),
        verdicts=(),
    )
    return AcceptanceCell(
        kind=unverified.kind,
        alpha=unverified.alpha,
        local_training_rows=unverified.local_training_rows,
        target_count=unverified.target_count,
        levels=unverified.levels,
        gain_without_excluded_interval=unverified.gain_without_excluded_interval,
        harmed_target_count=unverified.harmed_target_count,
        severe_target_count=unverified.severe_target_count,
        worst_target_difference=unverified.worst_target_difference,
        false_alert_difference=unverified.false_alert_difference,
        false_alert_interval=unverified.false_alert_interval,
        exceedance_increase=unverified.exceedance_increase,
        seed_gains=unverified.seed_gains,
        minimum_stratum_left_out_gain=unverified.minimum_stratum_left_out_gain,
        verdicts=_verdicts(unverified),
    )


def summarize_acceptance(inputs: ReportInputs) -> AcceptanceReport:
    reference = OperatingMethod.MEAN_ENSEMBLE_MATCHED
    settings = inputs.configuration.reporting
    return AcceptanceReport(
        reference,
        tuple(
            _acceptance_cell(cell, reference, settings)
            for cell in cell_results(inputs)
            if all(
                curve_of(result, reference) is not None
                for results in cell.targets.values()
                for result in results
            )
        ),
    )


@dataclass(frozen=True, slots=True)
class _Check:
    criterion: AcceptanceCriterion
    measure: AcceptanceMeasure
    measured: RateDifference | None
    threshold: RateDifference
    satisfied: CriterionSatisfied


def _verdict_of(cell: AcceptanceCell, criterion: AcceptanceCriterion) -> CriterionSatisfied:
    return next(item.satisfied for item in cell.verdicts if item.criterion is criterion)


def _gain_checks(cell: AcceptanceCell) -> tuple[_Check, ...]:
    primary = next(item for item in cell.levels if item.level is FprLevel.ALPHA)
    level_gains = [item.gain for item in cell.levels]
    smallest_level = (
        None
        if any(gain is None for gain in level_gains)
        else min(gain for gain in level_gains if gain is not None)
    )
    without = cell.gain_without_excluded_interval
    seed_minimum = min((item.difference for item in cell.seed_gains), default=None)
    return (
        _Check(
            AcceptanceCriterion.FIXED_FPR_GAIN,
            AcceptanceMeasure.FIXED_FPR_GAIN_LOWER_BOUND,
            None if primary.interval is None else primary.interval.lower_bound,
            0.0,
            _verdict_of(cell, AcceptanceCriterion.FIXED_FPR_GAIN),
        ),
        _Check(
            AcceptanceCriterion.COVERAGE,
            AcceptanceMeasure.COVERAGE_SHARE,
            primary.matched_targets / cell.target_count,
            AcceptanceCalibrationRule.COVERAGE_FLOOR.threshold,
            _verdict_of(cell, AcceptanceCriterion.COVERAGE),
        ),
        _Check(
            AcceptanceCriterion.LOW_FPR_CONSISTENCY,
            AcceptanceMeasure.SMALLEST_LEVEL_GAIN,
            smallest_level,
            0.0,
            _verdict_of(cell, AcceptanceCriterion.LOW_FPR_CONSISTENCY),
        ),
        _Check(
            AcceptanceCriterion.GENERALISATION,
            AcceptanceMeasure.LOWER_BOUND_WITHOUT_EXCLUDED,
            None if without is None else without.lower_bound,
            0.0,
            _verdict_of(cell, AcceptanceCriterion.GENERALISATION),
        ),
        _Check(
            AcceptanceCriterion.STRATUM_ROBUSTNESS,
            AcceptanceMeasure.SMALLEST_LEAVE_ONE_STRATUM_GAIN,
            cell.minimum_stratum_left_out_gain,
            0.0,
            _verdict_of(cell, AcceptanceCriterion.STRATUM_ROBUSTNESS),
        ),
        _Check(
            AcceptanceCriterion.SEED_SIGNS,
            AcceptanceMeasure.SMALLEST_SEED_GAIN,
            seed_minimum,
            0.0,
            _verdict_of(cell, AcceptanceCriterion.SEED_SIGNS),
        ),
    )


def _safeguard_checks(cell: AcceptanceCell) -> tuple[_Check, ...]:
    primary = next(item for item in cell.levels if item.level is FprLevel.ALPHA)
    harmed_share = (
        cell.harmed_target_count / primary.matched_targets if primary.matched_targets else None
    )
    margin = AcceptanceCalibrationRule.FALSE_ALERT_MARGIN_SHARE.threshold * cell.alpha.fraction
    limit = AcceptanceRule.HARMED_TARGET_SHARE.threshold
    return (
        _Check(
            AcceptanceCriterion.NON_INFERIORITY,
            AcceptanceMeasure.SEVERE_TARGETS,
            cell.severe_target_count,
            0.0,
            cell.severe_target_count == 0,
        ),
        _Check(
            AcceptanceCriterion.NON_INFERIORITY,
            AcceptanceMeasure.HARMED_TARGET_SHARE,
            harmed_share,
            limit,
            harmed_share is not None and harmed_share <= limit,
        ),
        _Check(
            AcceptanceCriterion.FALSE_ALERT_SAFEGUARD,
            AcceptanceMeasure.FALSE_ALERT_UPPER_BOUND,
            cell.false_alert_interval.upper_bound,
            margin,
            cell.false_alert_interval.upper_bound <= margin,
        ),
        _Check(
            AcceptanceCriterion.FALSE_ALERT_SAFEGUARD,
            AcceptanceMeasure.EXCEEDANCE_SHARE_INCREASE,
            cell.exceedance_increase,
            AcceptanceRule.EXCEEDANCE_INCREASE.threshold,
            cell.exceedance_increase <= AcceptanceRule.EXCEEDANCE_INCREASE.threshold,
        ),
    )


def _checks(cell: AcceptanceCell) -> tuple[_Check, ...]:
    return (*_gain_checks(cell), *_safeguard_checks(cell))


def _measured_text(value: RateDifference | None) -> DisplayText:
    if value is None:
        return DisplayText(UnavailableReading.NOT_AVAILABLE)
    return number(value)


def acceptance_table(inputs: ReportInputs) -> ResultTable:
    report = inputs.acceptance
    if report is None:
        raise ValueError("The acceptance report is missing")
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
        tuple(
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
            for cell in report.cells
            for check in _checks(cell)
        ),
    )


def false_alert_margin_sensitivity_table(inputs: ReportInputs) -> ResultTable:
    report = inputs.acceptance
    if report is None:
        raise ValueError("The acceptance report is missing")
    return ResultTable(
        TableName.FALSE_ALERT_MARGIN_SENSITIVITY,
        (
            ColumnHeading.ANALYSIS,
            ColumnHeading.ALPHA,
            ColumnHeading.LOCAL_ROWS,
            ColumnHeading.MARGIN_SHARE,
            ColumnHeading.MARGIN,
            ColumnHeading.UPPER_BOUND,
            ColumnHeading.SATISFIED,
        ),
        tuple(
            (
                analysis_text(cell.kind),
                alpha_text(cell.alpha),
                count(cell.local_training_rows),
                number(sensitivity.share),
                number(sensitivity.share * cell.alpha.fraction),
                number(cell.false_alert_interval.upper_bound),
                verdict_text(
                    cell.false_alert_interval.upper_bound <= sensitivity.share * cell.alpha.fraction
                ),
            )
            for cell in report.cells
            for sensitivity in MarginSensitivity
        ),
    )


def _level_text(item: LevelGain) -> DisplayText:
    if item.gain is None or item.interval is None:
        return DisplayText(f"{item.level.display_label} {UnavailableReading.NOT_AVAILABLE}")
    bounds = f"[{item.interval.lower_bound:.4f}, {item.interval.upper_bound:.4f}]"
    return DisplayText(
        f"{item.level.display_label} {item.gain:+.4f} {bounds} n={item.matched_targets}"
    )


def acceptance_line(cell: AcceptanceCell) -> DisplayText:
    failed = " / ".join(
        verdict.criterion.name.lower() for verdict in cell.verdicts if not verdict.satisfied
    )
    outcome = "accepted" if cell.is_accepted else f"failed [{failed}]"
    gating = "gating " if cell.is_gating else ""
    levels = " ; ".join(_level_text(item) for item in cell.levels)
    return DisplayText(
        f"{cell.kind.name.lower()} alpha={cell.alpha.fraction:.2f} rows={cell.local_training_rows} "
        f"targets={cell.target_count}: fixed-FPR gain {levels}; "
        f"harmed {cell.harmed_target_count}, severe {cell.severe_target_count}, "
        f"FPR difference {cell.false_alert_difference:+.4f} "
        f"(upper {cell.false_alert_interval.upper_bound:+.4f}), {gating}{outcome}"
    )


def acceptance_tables(inputs: ReportInputs) -> tuple[ResultTable, ...]:
    return (
        fixed_fpr_comparison_table(inputs),
        direction_decomposition_table(inputs),
        acceptance_table(inputs),
        false_alert_margin_sensitivity_table(inputs),
        inference_sensitivity_table(inputs),
        coverage_diagnosis_table(inputs),
    )
