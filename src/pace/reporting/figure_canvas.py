from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from math import ceil

from matplotlib.axes import Axes
from matplotlib.figure import Figure

from pace.reporting.records import ReportInputs, TargetRecord
from pace.types import (
    AlphaLevel,
    BudgetAnalysisKind,
    DisplayText,
    FigureExtent,
    FigureLayout,
    LocalTrainingRowCount,
    PlotInk,
    PlotLegendAnchor,
    PlotLineStyle,
    PlotPosition,
)


@dataclass(frozen=True, slots=True)
class Panel:
    alpha: AlphaLevel
    rows: LocalTrainingRowCount
    records: tuple[TargetRecord, ...]

    @property
    def title(self) -> DisplayText:
        return DisplayText(f"α = {self.alpha.fraction}, {self.rows} local rows")


def primary_panels(inputs: ReportInputs) -> tuple[Panel, ...]:
    return tuple(
        Panel(alpha, rows, inputs.primary(alpha, rows))
        for alpha in inputs.alphas()
        for rows in inputs.budgets(BudgetAnalysisKind.UNIVERSAL_PRIMARY)
        if inputs.primary(alpha, rows)
    )


def axes_center(figure: Figure) -> FigureExtent:
    figure.draw_without_rendering()
    positions = [axes.get_position() for axes in figure.axes]
    return (min(item.x0 for item in positions) + max(item.x1 for item in positions)) / 2.0


def new_figure(width: FigureExtent, height: FigureExtent) -> Figure:
    return Figure(figsize=(width, height), layout=FigureLayout.CONSTRAINED)


def panel_grid(figure: Figure, panels: tuple[Panel, ...]) -> Iterator[tuple[Panel, Axes]]:
    columns = 2 if len(panels) > 1 else 1
    rows = ceil(len(panels) / columns)
    for index, panel in enumerate(panels, start=1):
        axes = figure.add_subplot(rows, columns, index)
        axes.set_title(panel.title)
        axes.grid(alpha=0.25)
        yield panel, axes


def legend_outside(axes: Axes) -> None:
    handles, labels = axes.get_legend_handles_labels()
    if handles:
        axes.legend(handles, labels, frameon=False, fontsize=8)


def legend_beside(figure: Figure) -> None:
    if figure.axes:
        handles, labels = figure.axes[0].get_legend_handles_labels()
        if handles:
            figure.legend(
                handles=handles,
                labels=labels,
                loc=PlotLegendAnchor.OUTSIDE_RIGHT,
                frameon=False,
                fontsize=8,
            )


def stratum_ticks(
    axes: Axes, positions: Sequence[PlotPosition], labels: Sequence[DisplayText]
) -> None:
    axes.set_xticks(positions, labels, rotation=30, ha="right", rotation_mode="anchor", fontsize=8)


def vertical_guide(
    axes: Axes,
    value: PlotPosition,
    *,
    ink: PlotInk = PlotInk.BLACK,
    style: PlotLineStyle = PlotLineStyle.DASHED,
    width: FigureExtent = 0.6,
    label: DisplayText | None = None,
) -> None:
    axes.axvline(value, color=ink, linewidth=width, linestyle=style, label=label)


def horizontal_guide(
    axes: Axes,
    value: PlotPosition,
    *,
    ink: PlotInk = PlotInk.BLACK,
    style: PlotLineStyle = PlotLineStyle.DASHED,
    width: FigureExtent = 0.6,
    label: DisplayText | None = None,
) -> None:
    axes.axhline(value, color=ink, linewidth=width, linestyle=style, label=label)
