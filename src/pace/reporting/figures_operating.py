from collections import defaultdict
from itertools import cycle

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from pace.reporting.figure_canvas import new_figure, note_overlapping_curves, side_legend
from pace.reporting.operating import (
    MeanOperatingPoint,
    comparison_references,
    mean_operating_points,
    primary_comparisons,
)
from pace.reporting.records import ReportInputs
from pace.types import (
    AlphaLevel,
    BudgetAnalysisKind,
    CriterionSatisfied,
    DetectionRate,
    FalseAlertRate,
    LocalTrainingRowCount,
    MatchDirection,
    OperatingMethod,
    PlotInk,
    PlotMarker,
    SeriesStyle,
)


def _by_panel(
    inputs: ReportInputs,
) -> dict[LocalTrainingRowCount, dict[OperatingMethod, list[MeanOperatingPoint]]]:
    grouped: dict[LocalTrainingRowCount, dict[OperatingMethod, list[MeanOperatingPoint]]] = (
        defaultdict(lambda: defaultdict(list))
    )
    for point in mean_operating_points(inputs):
        grouped[point.rows][point.method].append(point)
    return grouped


def _series_styles(methods: set[OperatingMethod]) -> dict[OperatingMethod, SeriesStyle]:
    styles = {
        OperatingMethod.PACE: SeriesStyle.BOLD,
        OperatingMethod.MEAN_ENSEMBLE_SAME_CALIBRATION: SeriesStyle.DASHED,
    }
    background = cycle((SeriesStyle.DASH_DOT, SeriesStyle.DOTTED, SeriesStyle.THIN))
    others = sorted(methods - styles.keys(), key=lambda method: method.name)
    styles.update(zip(others, background, strict=False))
    return styles


def _plot_operating_methods(
    axes: Axes,
    methods: dict[OperatingMethod, list[MeanOperatingPoint]],
    visible: set[OperatingMethod],
    references: set[OperatingMethod],
    styles: dict[OperatingMethod, SeriesStyle],
    limit: FalseAlertRate,
) -> CriterionSatisfied:
    has_points = False
    for method, points in methods.items():
        if method not in visible:
            continue
        shown = [point for point in points if point.false_alert_rate <= limit]
        if shown:
            has_points = True
            reference = method in references
            axes.plot(
                [point.false_alert_rate for point in shown],
                [point.detection_rate for point in shown],
                marker=PlotMarker.CIRCLE,
                color=method.color,
                label=method.display_label,
                linestyle=styles[method].line,
                linewidth=styles[method].width,
                alpha=1.0 if reference else 0.65,
                markersize=4 if reference else 2.5,
                zorder=3 if reference else 2,
            )
    return has_points


def comparable_operating_curves(inputs: ReportInputs) -> Figure:
    grouped = _by_panel(inputs)
    figure = new_figure(13.0, 13.0)
    if not grouped:
        figure.add_subplot(1, 1, 1).set_title("No operating curves were recorded")
        return figure
    baselines = {
        baseline.operating_method for baseline in inputs.configuration.scientific.baselines.methods
    }
    references = {OperatingMethod.PACE, OperatingMethod.MEAN_ENSEMBLE_SAME_CALIBRATION}
    variants = set(comparison_references(inputs)) - baselines - references
    styles = _series_styles(baselines | variants | references)
    panels = (
        ("Baselines", 1.0, baselines),
        ("Baselines, low FPR", 0.06, baselines),
        ("Variants", 1.0, variants),
        ("Variants, low FPR", 0.06, variants),
    )
    for row, (label, limit, selected) in enumerate(panels):
        for column, (rows, methods) in enumerate(sorted(grouped.items())):
            axes = figure.add_subplot(
                len(panels), max(1, len(grouped)), row * len(grouped) + column + 1
            )
            axes.grid(alpha=0.25)
            has_points = _plot_operating_methods(
                axes, methods, selected | references, references, styles, limit
            )
            axes.set_title(
                f"{label}, {rows} local rows" + (" — no points" if not has_points else "")
            )
            if limit < 1.0:
                axes.set_xlim(0.0, limit)
            axes.set_ylim(0.0, 1.02)
            axes.set_xlabel("Mean FPR (target-seed runs;\nnominal-budget sweep)")
            axes.set_ylabel("Mean detection rate\n(target-seed runs)")
    handles, labels = figure.axes[0].get_legend_handles_labels()
    variant_handles, variant_labels = figure.axes[2 * len(grouped)].get_legend_handles_labels()
    handles.extend(
        handle
        for handle, label in zip(variant_handles, variant_labels, strict=True)
        if label not in labels
    )
    note_overlapping_curves(figure)
    side_legend(figure, handles)
    return figure


def _direction_ink(direction: MatchDirection) -> PlotInk:
    match direction:
        case MatchDirection.SUBJECT_AT_REFERENCE:
            return PlotInk.CYCLE_0
        case MatchDirection.REFERENCE_AT_SUBJECT:
            return PlotInk.CYCLE_1


def matching_coverage(inputs: ReportInputs) -> Figure:
    references = comparison_references(inputs)
    figure = new_figure(14.0, max(7.0, 2.0 + 0.4 * len(references)))
    axes = figure.add_subplot(1, 1, 1)
    axes.grid(alpha=0.25)
    width = 0.8 / max(1, len(AlphaLevel) * len(MatchDirection))
    offset = 0
    for alpha in AlphaLevel:
        for direction in MatchDirection:
            coverage: list[DetectionRate] = []
            for reference in references:
                cells = [
                    cell for cell in primary_comparisons(inputs, reference) if cell.alpha is alpha
                ]
                total = sum(len(cell.comparisons) for cell in cells)
                matched = sum(
                    item.matched_difference(direction) is not None
                    for cell in cells
                    for item in cell.comparisons
                )
                coverage.append(matched / total if total else np.nan)
            axes.barh(
                [index + offset * width for index in range(len(references))],
                coverage,
                height=width,
                color=_direction_ink(direction),
                alpha=0.4 + 0.3 * (alpha is AlphaLevel.ALPHA_005),
                label=f"α = {alpha.fraction}: {direction.display_label}",
            )
            offset += 1
    axes.set_yticks(
        [index + width * (offset - 1) / 2 for index in range(len(references))],
        [reference.display_label for reference in references],
        fontsize=8,
    )
    budgets = " · ".join(
        f"{value}" for value in inputs.budgets(BudgetAnalysisKind.UNIVERSAL_PRIMARY)
    )
    axes.set_title(f"Training budgets pooled: {budgets} rows")
    axes.set_xlabel("Matched share of available target-budget pairs (no bar: unavailable)")
    handles, _ = axes.get_legend_handles_labels()
    side_legend(figure, handles)
    return figure
