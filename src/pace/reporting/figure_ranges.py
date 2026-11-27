from collections.abc import Iterable

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from pace.types import DisplayText, FigureExtent, FigureLimit, PlotPosition


def _reach(low: PlotPosition, high: PlotPosition) -> FigureExtent:
    if np.isfinite(low) and np.isfinite(high):
        return max(abs(low), abs(high))
    return 0.0


def _shared_half_width(intervals: Iterable[tuple[PlotPosition, PlotPosition]]) -> FigureExtent:
    reach = max(_reach(low, high) for low, high in intervals)
    return max(reach * (1.0 + FigureLimit.DATA_MARGIN.size), FigureLimit.MINIMUM_HALF_WIDTH.size)


def _flag_negligible(axes: Axes, low: PlotPosition, high: PlotPosition) -> None:
    if (
        np.isfinite(low)
        and np.isfinite(high)
        and _reach(low, high) < FigureLimit.NEGLIGIBLE_RATE.size
    ):
        axes.annotate(
            DisplayText(f"no difference (all |gain| < {FigureLimit.NEGLIGIBLE_RATE.size})"),
            xy=(0.5, FigureLimit.NOTE_HEIGHT.size),
            xycoords="axes fraction",
            ha="center",
            va="center",
            fontsize=8,
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="0.7"),
        )


def share_gain_ordinate(figure: Figure) -> None:
    if not figure.axes:
        return
    half = _shared_half_width(axes.dataLim.intervaly for axes in figure.axes)
    for axes in figure.axes:
        axes.set_ylim(-half, half)
        _flag_negligible(axes, *axes.dataLim.intervaly)


def share_gain_abscissa(figure: Figure) -> None:
    if not figure.axes:
        return
    half = _shared_half_width(axes.dataLim.intervalx for axes in figure.axes)
    for axes in figure.axes:
        axes.set_xlim(-half, half)
        _flag_negligible(axes, *axes.dataLim.intervalx)
