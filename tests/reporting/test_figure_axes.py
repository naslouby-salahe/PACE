from pathlib import Path

from pace.reporting.figure_canvas import new_figure
from pace.reporting.figures import save_figure
from pace.types import FigureName


def test_axes_that_only_show_floating_point_noise_get_a_readable_range(tmp_path: Path) -> None:
    figure = new_figure(4.0, 3.0)
    axes = figure.add_subplot(1, 1, 1)
    axes.scatter([0.0, 1.0, 2.0], [0.0, -2.7e-17, 0.0])
    save_figure(figure, FigureName.TARGET_GAINS, tmp_path)
    bottom, top = axes.get_ylim()
    assert top - bottom >= 0.08


def test_axes_spanning_less_than_a_thousandth_are_widened_instead_of_using_offset_notation(
    tmp_path: Path,
) -> None:
    figure = new_figure(4.0, 3.0)
    axes = figure.add_subplot(1, 1, 1)
    axes.scatter([0.99990, 0.99999], [1e-5, 3e-5])
    save_figure(figure, FigureName.TARGET_GAINS, tmp_path)
    bottom, top = axes.get_ylim()
    left, right = axes.get_xlim()
    assert top - bottom >= 1e-3 - 1e-12
    assert right - left >= 1e-3 - 1e-12


def test_axes_with_a_meaningful_range_are_left_alone(tmp_path: Path) -> None:
    figure = new_figure(4.0, 3.0)
    axes = figure.add_subplot(1, 1, 1)
    axes.scatter([0.0, 1.0], [0.0, 0.2])
    before = axes.get_ylim()
    save_figure(figure, FigureName.TARGET_GAINS, tmp_path)
    assert axes.get_ylim() == before
