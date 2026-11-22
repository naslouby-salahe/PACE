from collections import defaultdict

import numpy as np
from matplotlib.figure import Figure

from pace.reporting.figure_canvas import new_figure
from pace.reporting.operating import (
    curve_of,
    point_at,
    point_detection,
    unique_results,
)
from pace.reporting.records import ReportInputs, ResultTable, count, number
from pace.types import (
    AlphaLevel,
    BudgetAnalysisKind,
    CalibrationMode,
    CalibrationScenario,
    ColumnHeading,
    DetectionRate,
    DetectorCount,
    DisplayText,
    FalseAlarmBudget,
    FalseAlertRate,
    LocalTrainingRowCount,
    OperatingMethod,
    PlotInk,
    PlotLegendAnchor,
    PlotLineStyle,
    PlotMarker,
    SensitivityRecord,
    TableName,
)


def calibration_sensitivity_table(records: tuple[SensitivityRecord, ...]) -> ResultTable:
    rows = tuple(
        (
            item.scenario.display_label,
            DisplayText(item.mode.name),
            count(item.rows),
            count(item.experts),
            DisplayText(f"{item.tier_budget}"),
            count(item.replicates),
            number(item.mean_false_alert),
            number(item.budget_ratio),
            number(item.mean_detection),
        )
        for item in records
    )
    return ResultTable(
        TableName.CALIBRATION_SENSITIVITY,
        (
            ColumnHeading.SCENARIO,
            ColumnHeading.METHOD,
            ColumnHeading.CALIBRATION_ROWS,
            ColumnHeading.EXPERT_COUNT,
            ColumnHeading.NOMINAL_BUDGET,
            ColumnHeading.REPLICATES,
            ColumnHeading.FALSE_ALERT_RATE,
            ColumnHeading.BUDGET_RATIO,
            ColumnHeading.DETECTION_RATE,
        ),
        rows,
    )


def calibration_sensitivity(inputs: ReportInputs) -> Figure:
    return calibration_sensitivity_figure(inputs.sensitivity)


def _sensitivity_series(
    records: tuple[SensitivityRecord, ...],
    scenario: CalibrationScenario,
    mode: CalibrationMode,
    budget: FalseAlarmBudget,
    experts: DetectorCount,
) -> list[SensitivityRecord]:
    return sorted(
        (
            item
            for item in records
            if item.scenario is scenario
            and item.mode is mode
            and item.tier_budget == budget
            and item.experts == experts
        ),
        key=lambda item: item.rows,
    )


def calibration_sensitivity_figure(records: tuple[SensitivityRecord, ...]) -> Figure:
    figure = new_figure(11.0, 7.5)
    figure.suptitle("Synthetic calibration study")
    if not records:
        figure.add_subplot(1, 1, 1).set_title("No calibration study was recorded")
        return figure
    budget = min(item.tier_budget for item in records)
    experts = max(item.experts for item in records)
    for index, scenario in enumerate(CalibrationScenario, start=1):
        axes = figure.add_subplot(2, 2, index)
        axes.set_title(scenario.display_label)
        axes.grid(alpha=0.25)
        axes.set_xscale("log")
        axes.set_yscale("log")
        for rank, mode in enumerate(CalibrationMode):
            selected = _sensitivity_series(records, scenario, mode, budget, experts)
            if selected:
                axes.plot(
                    [item.rows for item in selected],
                    [max(item.budget_ratio, 1e-4) for item in selected],
                    marker=PlotMarker.CIRCLE,
                    markersize=11 - 1.5 * rank,
                    linewidth=3.2 - 0.5 * rank,
                    label=mode.display_label,
                )
        vertical = PlotInk.BLACK
        axes.axhline(
            1.0,
            color=vertical,
            linewidth=0.8,
            linestyle=PlotLineStyle.DASHED,
            label="FPR equals budget",
        )
        axes.set_xlabel("Calibration rows")
        axes.set_ylabel(f"Realised FPR / budget\n({experts} experts, tier {budget}; floor 10⁻⁴)")
    handles, _ = figure.axes[0].get_legend_handles_labels()
    figure.legend(handles=handles, loc=PlotLegendAnchor.OUTSIDE_RIGHT, fontsize=8, frameon=False)
    return figure


def _method_points(
    inputs: ReportInputs,
) -> dict[
    tuple[AlphaLevel, LocalTrainingRowCount, OperatingMethod],
    list[tuple[DetectionRate, FalseAlertRate, DetectionRate]],
]:
    cohort = inputs.cohort(BudgetAnalysisKind.UNIVERSAL_PRIMARY)
    budgets = inputs.budgets(BudgetAnalysisKind.UNIVERSAL_PRIMARY)
    gathered: dict[
        tuple[AlphaLevel, LocalTrainingRowCount, OperatingMethod],
        list[tuple[DetectionRate, FalseAlertRate, DetectionRate]],
    ] = defaultdict(list)
    for (alpha, rows), targets in unique_results(inputs.performances).items():
        if rows not in budgets:
            continue
        for target, seeds in targets.items():
            if target not in cohort:
                continue
            for result in seeds:
                reference = curve_of(result, OperatingMethod.MEAN_ENSEMBLE)
                if reference is None:
                    continue
                ensemble = point_detection(point_at(reference, alpha.fraction))
                for curve in result.curves:
                    point = point_at(curve, alpha.fraction)
                    gathered[(alpha, rows, curve.method)].append(
                        (point_detection(point), point.false_alert_rate, ensemble)
                    )
    return gathered


def calibration_variants_table(inputs: ReportInputs) -> ResultTable:
    rows: list[tuple[DisplayText, ...]] = []
    for (alpha, budget, method), values in sorted(
        _method_points(inputs).items(),
        key=lambda item: (item[0][1], item[0][0].fraction, item[0][2].name),
    ):
        detection = np.asarray([item[0] for item in values])
        false_alert = np.asarray([item[1] for item in values])
        ensemble = np.asarray([item[2] for item in values])
        rows.append(
            (
                DisplayText(f"{alpha.fraction}"),
                count(budget),
                method.display_label,
                count(len(values)),
                number(detection.mean().item()),
                number(false_alert.mean().item()),
                number(np.abs(false_alert - alpha.fraction).mean().item()),
                number(np.mean(false_alert < 0.1 * alpha.fraction).item()),
                count(np.sum((detection < 0.05) & (ensemble >= 0.5)).item()),
                number((detection - ensemble).mean().item()),
            )
        )
    return ResultTable(
        TableName.CALIBRATION_VARIANTS,
        (
            ColumnHeading.ALPHA,
            ColumnHeading.LOCAL_ROWS,
            ColumnHeading.VARIANT,
            ColumnHeading.TARGETS,
            ColumnHeading.DETECTION_RATE,
            ColumnHeading.FALSE_ALERT_RATE,
            ColumnHeading.CALIBRATION_ERROR,
            ColumnHeading.SHARE_OVER_CONSERVATIVE,
            ColumnHeading.COLLAPSED_TARGETS,
            ColumnHeading.GAIN,
        ),
        tuple(rows),
    )
