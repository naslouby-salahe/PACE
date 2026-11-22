from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from pace.reporting.operating import (
    ComparisonCell,
    curve_of,
    point_at,
    primary_comparisons,
    unique_results,
)
from pace.reporting.records import ReportInputs, ResultTable, count, number, optional_number
from pace.types import (
    AlphaLevel,
    AttackType,
    BudgetAnalysisKind,
    ColumnHeading,
    DatasetClientId,
    DetectionRate,
    DisplayText,
    FalseAlertRate,
    LocalTrainingRowCount,
    MatchDirection,
    ObservationCount,
    OperatingMethod,
    RateDifference,
    StudyArm,
    StudyStratum,
    TableName,
    TargetPerformance,
)


def _primary_results(
    inputs: ReportInputs,
) -> tuple[
    tuple[AlphaLevel, LocalTrainingRowCount, dict[DatasetClientId, list[TargetPerformance]]], ...
]:
    cohort = inputs.cohort(BudgetAnalysisKind.UNIVERSAL_PRIMARY)
    budgets = inputs.budgets(BudgetAnalysisKind.UNIVERSAL_PRIMARY)
    return tuple(
        (
            alpha,
            rows,
            {target: seeds for target, seeds in targets.items() if target in cohort},
        )
        for (alpha, rows), targets in sorted(
            unique_results(inputs.performances).items(),
            key=lambda item: (item[0][1], item[0][0].fraction),
        )
        if rows in budgets
    )


def _curve_false_alert(
    result: TargetPerformance, method: OperatingMethod, alpha: AlphaLevel
) -> FalseAlertRate | None:
    curve = curve_of(result, method)
    return None if curve is None else point_at(curve, alpha.fraction).false_alert_rate


def _mean_optional(values: list[FalseAlertRate | None]) -> DisplayText:
    present = [value for value in values if value is not None]
    return optional_number(np.mean(present).item() if present else None)


def _method_false_alert(
    cells: list[TargetPerformance], method: OperatingMethod, alpha: AlphaLevel
) -> DisplayText:
    return _mean_optional([_curve_false_alert(result, method, alpha) for result in cells])


def calibration_audit_table(inputs: ReportInputs) -> ResultTable:
    rows: list[tuple[DisplayText, ...]] = []
    for alpha, budget, targets in _primary_results(inputs):
        for stratum in StudyStratum:
            cells = [
                result
                for seeds in targets.values()
                for result in seeds
                if result.study_stratum is stratum and result.allocation is not None
            ]
            if not cells:
                continue
            audits = [result.allocation for result in cells if result.allocation is not None]
            realized = [result.realized_false_alert_rate for result in cells]
            rows.append(
                (
                    DisplayText(f"{alpha.fraction}"),
                    count(budget),
                    stratum.display_label,
                    count(len({result.target for result in cells})),
                    number(np.mean([audit.calibration_rows for audit in audits]).item()),
                    number(np.mean([audit.resolution_floor for audit in audits]).item()),
                    number(np.mean([audit.effective_nominal_budget for audit in audits]).item()),
                    number(np.mean(realized).item()),
                    _method_false_alert(cells, OperatingMethod.LEGACY_PACE, alpha),
                    _method_false_alert(cells, OperatingMethod.CROSS_FITTED_PACE, alpha),
                    number(np.mean([value > 2.0 * alpha.fraction for value in realized]).item()),
                )
            )
    return ResultTable(
        TableName.CALIBRATION_AUDIT,
        (
            ColumnHeading.ALPHA,
            ColumnHeading.LOCAL_ROWS,
            ColumnHeading.STRATUM,
            ColumnHeading.TARGETS,
            ColumnHeading.CALIBRATION_ROWS,
            ColumnHeading.RESOLUTION_FLOOR,
            ColumnHeading.EFFECTIVE_BUDGET,
            ColumnHeading.PACE_FALSE_ALERT,
            ColumnHeading.ORIGINAL_FALSE_ALERT,
            ColumnHeading.CROSS_FITTED_FALSE_ALERT,
            ColumnHeading.SHARE_ABOVE_TWICE_ALPHA,
        ),
        tuple(rows),
    )


def high_false_alert_table(inputs: ReportInputs) -> ResultTable:
    rows: list[tuple[DisplayText, ...]] = []
    for alpha, budget, targets in _primary_results(inputs):
        ranked = sorted(
            targets.items(),
            key=lambda item: -np.mean([r.realized_false_alert_rate for r in item[1]]),
        )[: inputs.configuration.reporting.high_false_alert_targets]
        for target, seeds in ranked:
            audits = [result.allocation for result in seeds if result.allocation is not None]
            for method in OperatingMethod:
                values = [_curve_false_alert(result, method, alpha) for result in seeds]
                if all(value is None for value in values):
                    continue
                rows.append(
                    (
                        DisplayText(f"{alpha.fraction}"),
                        count(budget),
                        seeds[0].study_stratum.display_label,
                        DisplayText(target.name),
                        count(seeds[0].benign_test_rows),
                        optional_number(
                            np.mean([audit.calibration_rows for audit in audits]).item()
                            if audits
                            else None
                        ),
                        method.display_label,
                        _mean_optional(values),
                    )
                )
    return ResultTable(
        TableName.HIGH_FALSE_ALERT_TARGETS,
        (
            ColumnHeading.ALPHA,
            ColumnHeading.LOCAL_ROWS,
            ColumnHeading.STRATUM,
            ColumnHeading.TARGET,
            ColumnHeading.BENIGN_TEST_ROWS,
            ColumnHeading.CALIBRATION_ROWS,
            ColumnHeading.METHOD,
            ColumnHeading.FALSE_ALERT_RATE,
        ),
        tuple(rows),
    )


@dataclass(frozen=True, slots=True)
class AttackRates:
    pace: DetectionRate
    mean_ensemble: DetectionRate
    local: DetectionRate
    peer_union: DetectionRate
    seeds_with_gain: ObservationCount


def _attack_rates(seeds: list[TargetPerformance]) -> dict[AttackType, AttackRates]:
    reference = StudyArm.MEAN_ENSEMBLE_LOCAL_BRANCH
    collected: dict[AttackType, list[tuple[DetectionRate, ...]]] = defaultdict(list)
    for result in seeds:
        arm = next((item for item in result.arms if item.arm is reference), None)
        if arm is None:
            continue
        for index, item in enumerate(result.attack_types):
            collected[item.attack_type].append(
                (
                    item.detection_rate,
                    arm.detection_rates[index],
                    item.local_detection_rate,
                    item.peer_union_detection_rate,
                )
            )
    return {
        attack: AttackRates(
            np.mean([row[0] for row in rows]).item(),
            np.mean([row[1] for row in rows]).item(),
            np.mean([row[2] for row in rows]).item(),
            np.mean([row[3] for row in rows]).item(),
            sum(row[0] > row[1] for row in rows),
        )
        for attack, rows in collected.items()
    }


def ton_iot_attack_table(inputs: ReportInputs) -> ResultTable:
    rows: list[tuple[DisplayText, ...]] = []
    for alpha, budget, targets in _primary_results(inputs):
        for target, seeds in sorted(targets.items(), key=lambda item: item[0].name):
            if seeds[0].study_stratum is not StudyStratum.TON_IOT:
                continue
            for attack, rates in _attack_rates(seeds).items():
                rows.append(
                    (
                        DisplayText(f"{alpha.fraction}"),
                        count(budget),
                        DisplayText(target.name),
                        DisplayText(attack.identifier),
                        number(rates.pace),
                        number(rates.mean_ensemble),
                        number(rates.pace - rates.mean_ensemble),
                        number(rates.local),
                        number(rates.peer_union),
                        count(rates.seeds_with_gain),
                    )
                )
    return ResultTable(
        TableName.TON_IOT_ATTACKS,
        (
            ColumnHeading.ALPHA,
            ColumnHeading.LOCAL_ROWS,
            ColumnHeading.TARGET,
            ColumnHeading.ATTACK_TYPE,
            ColumnHeading.PACE_DETECTION,
            ColumnHeading.MEAN_ENSEMBLE_DETECTION,
            ColumnHeading.GAIN,
            ColumnHeading.LOCAL_DETECTION,
            ColumnHeading.PEER_UNION_DETECTION,
            ColumnHeading.SEEDS_POSITIVE,
        ),
        tuple(rows),
    )


def _stratum_gains(cell: ComparisonCell, stratum: StudyStratum) -> tuple[RateDifference, ...]:
    return tuple(item.nominal_difference() for item in cell.stratum(stratum))


def gain_concentration_table(inputs: ReportInputs) -> ResultTable:
    rows: list[tuple[DisplayText, ...]] = []
    for cell in primary_comparisons(inputs, OperatingMethod.MEAN_ENSEMBLE):
        total = sum(sum(_stratum_gains(cell, stratum)) for stratum in StudyStratum)
        for stratum in StudyStratum:
            items = cell.stratum(stratum)
            if not items:
                continue
            gains = _stratum_gains(cell, stratum)
            matched = [
                value
                for item in items
                if (value := item.matched_difference(MatchDirection.SUBJECT_AT_REFERENCE))
                is not None
            ]
            rows.append(
                (
                    DisplayText(f"{cell.alpha.fraction}"),
                    count(cell.rows),
                    stratum.display_label,
                    count(len(items)),
                    number(np.mean(gains).item()),
                    optional_number(sum(gains) / total if total else None),
                    count(len(matched)),
                    optional_number(np.mean(matched).item() if matched else None),
                )
            )
    return ResultTable(
        TableName.GAIN_CONCENTRATION,
        (
            ColumnHeading.ALPHA,
            ColumnHeading.LOCAL_ROWS,
            ColumnHeading.STRATUM,
            ColumnHeading.TARGETS,
            ColumnHeading.GAIN,
            ColumnHeading.GAIN_SHARE,
            ColumnHeading.MATCHED_TARGETS,
            ColumnHeading.MATCHED_GAIN,
        ),
        tuple(rows),
    )
