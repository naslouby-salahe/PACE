import matplotlib
import pytest
from matplotlib.figure import Figure

from pace.reporting.figure_canvas import new_figure, stratum_ticks, wrap_label
from pace.reporting.figure_ranges import share_gain_abscissa, share_gain_ordinate
from pace.types import DisplayText, FigureCount, FigureLimit, StudyStratum

NO_DIFFERENCE_NOTE = "no difference (all |gain| < 0.001)"


def _panels(*columns: tuple[float, ...]) -> Figure:
    figure = new_figure(6.0, 3.0)
    for index, values in enumerate(columns, start=1):
        axes = figure.add_subplot(1, len(columns), index)
        axes.scatter(list(range(len(values))), list(values))
    return figure


def _notes(figure: Figure) -> list[bool]:
    return [any(text.get_text() == NO_DIFFERENCE_NOTE for text in ax.texts) for ax in figure.axes]


def test_panels_share_one_symmetric_range_covering_the_largest_panel() -> None:
    figure = _panels((0.0, 0.2), (0.0, -0.4))
    share_gain_ordinate(figure)
    limits = {axes.get_ylim() for axes in figure.axes}
    assert len(limits) == 1
    bottom, top = limits.pop()
    assert bottom == -top
    assert top >= 0.4


def test_shared_range_has_a_minimum_half_width() -> None:
    figure = _panels((0.0, 2.7e-17), (0.0, 0.0))
    share_gain_ordinate(figure)
    for axes in figure.axes:
        assert axes.get_ylim() == (
            -FigureLimit.MINIMUM_HALF_WIDTH.size,
            FigureLimit.MINIMUM_HALF_WIDTH.size,
        )


def test_only_panels_without_any_gain_are_annotated() -> None:
    figure = _panels((0.0, 2.7e-17), (0.0, 0.2), (0.0, 0.0009))
    share_gain_ordinate(figure)
    assert _notes(figure) == [True, False, True]


def test_the_note_sits_away_from_zero_inside_a_box() -> None:
    figure = _panels((0.0, 0.0))
    share_gain_ordinate(figure)
    notes = [text for text in figure.axes[0].texts if text.get_text() == NO_DIFFERENCE_NOTE]
    assert [text.get_position() for text in notes] == [(0.5, FigureLimit.NOTE_HEIGHT.size)]
    assert FigureLimit.NOTE_HEIGHT.size > 0.75
    assert all(text.get_bbox_patch() is not None for text in notes)


def test_the_abscissa_variant_shares_the_horizontal_range() -> None:
    figure = _panels((0.0, 0.0), (0.0, 0.0))
    for axes in figure.axes:
        axes.scatter([0.0, 0.3], [0.0, 1.0])
    share_gain_abscissa(figure)
    limits = {axes.get_xlim() for axes in figure.axes}
    assert len(limits) == 1
    assert limits.pop()[1] >= 0.3
    assert _notes(figure) == [False, False]


def test_a_figure_without_axes_is_left_alone() -> None:
    figure = new_figure(4.0, 3.0)
    share_gain_ordinate(figure)
    share_gain_abscissa(figure)
    assert not figure.axes


def test_a_single_category_label_is_not_rotated() -> None:
    figure = new_figure(4.0, 3.0)
    single = figure.add_subplot(1, 2, 1)
    stratum_ticks(single, [0], [StudyStratum.UNSW_IOT_ATTACK_FLOWS_G1])
    several = figure.add_subplot(1, 2, 2)
    stratum_ticks(several, [0, 1], [StudyStratum.CTU_13, StudyStratum.IOT_23])
    assert [text.get_rotation() for text in single.get_xticklabels()] == [0.0]
    assert all(text.get_rotation() > 0.0 for text in several.get_xticklabels())


@pytest.mark.parametrize("label", ["Mean-ensemble local branch (block-max calibration arm)"])
def test_wrapped_labels_never_break_inside_a_hyphenated_word(label: str) -> None:
    wrapped = wrap_label(DisplayText(label), FigureCount.ARM_LABEL_CHARACTERS)
    assert all(len(line) <= FigureCount.ARM_LABEL_CHARACTERS.size for line in wrapped.split("\n"))
    assert "block-max" in wrapped.replace("\n", " ")
    assert "block-\n" not in wrapped


def test_figures_never_use_offset_notation() -> None:
    assert matplotlib.rcParams["axes.formatter.useoffset"] is False
