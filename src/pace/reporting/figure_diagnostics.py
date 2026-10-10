import numpy as np
from matplotlib.figure import Figure

from pace.core.decision import (
    calibration_block_indices,
    sorted_block_max_threshold,
    sorted_blocks,
)
from pace.reporting.figure_canvas import (
    horizontal_guide,
    new_figure,
    primary_panels,
    vertical_guide,
)
from pace.reporting.metrics import paired_minimum_detectable_effect
from pace.reporting.records import ReportInputs
from pace.types import (
    AlphaLevel,
    BudgetAnalysisKind,
    BudgetGuide,
    CalibrationBlockSize,
    DisplayText,
    FalseAlarmBudget,
    LearningSeed,
    PlotInk,
    PlotLineStyle,
    PlotMarker,
    ReferenceAnnotation,
    Threshold,
)


def power(inputs: ReportInputs) -> Figure:
    protocol = inputs.configuration.protocol_matrix
    sizes = range(5, 120)
    figure = new_figure(7.0, 4.0)
    axes = figure.add_subplot(1, 1, 1)
    axes.grid(alpha=0.25)
    for sd in protocol.power.difference_sds:
        axes.plot(
            sizes,
            [paired_minimum_detectable_effect(size, sd, protocol.power) for size in sizes],
            label=DisplayText(f"assumed paired SD {sd}"),
        )
    horizontal_guide(axes, protocol.power.minimum_meaningful_effect, width=0.8)
    observed_gain = DisplayText("observed mean gain")
    labeled_observed_gain = False
    for panel in primary_panels(inputs):
        gain = np.mean([item.gain for item in panel.records]).item()
        axes.scatter(
            [len(panel.records)],
            [gain],
            s=30,
            marker=PlotMarker.CROSS,
            color=PlotInk.BLACK,
            label=None if labeled_observed_gain else observed_gain,
        )
        labeled_observed_gain = True
    axes.set_xlabel("Number of targets")
    axes.set_ylabel("Minimum detectable effect")
    axes.legend(frameon=False, fontsize=7)
    return figure


def census(inputs: ReportInputs) -> Figure:
    figure = new_figure(8.0, 4.5)
    axes = figure.add_subplot(1, 1, 1)
    axes.grid(alpha=0.25)
    for index, item in enumerate(sorted(inputs.training, key=lambda entry: entry.stratum.name)):
        rows = np.asarray([target.training_rows for target in item.targets])
        jitter = np.random.default_rng(index).uniform(-0.2, 0.2, len(rows))
        axes.scatter(index + jitter, rows, s=14, color=item.stratum.color)
    axes.set_yscale("log")
    axes.set_xticks(
        range(len(inputs.training)),
        [
            item.stratum.display_label
            for item in sorted(inputs.training, key=lambda entry: entry.stratum.name)
        ],
        rotation=45,
    )
    for budget in inputs.budgets(BudgetAnalysisKind.HIGH_RESOURCE):
        axes.axhline(budget, color=PlotInk.BLACK, linewidth=0.5, linestyle=PlotLineStyle.DOTTED)
    axes.set_ylabel("Training-role rows per target")
    return figure


def _threshold_curve(
    seed: LearningSeed, block_size: CalibrationBlockSize, budgets: tuple[FalseAlarmBudget, ...]
) -> tuple[Threshold, ...]:
    evidence = np.random.default_rng(seed).pareto(2.5, size=block_size * 16)
    blocks = sorted_blocks(
        tuple(evidence[indices] for indices in calibration_block_indices(evidence.size, block_size))
    )
    return tuple(sorted_block_max_threshold(blocks, budget) for budget in budgets)


def resolution_mechanism(inputs: ReportInputs) -> Figure:
    reserve = inputs.configuration.scientific.reserve_fraction
    block = inputs.configuration.scientific.calibration_block_size
    budgets = tuple(np.linspace(0.001, 0.03, 300).tolist())
    figure = new_figure(7.5, 4.0)
    axes = figure.add_subplot(1, 1, 1)
    axes.grid(alpha=0.25)
    axes.plot(
        budgets,
        _threshold_curve(inputs.configuration.scientific.seed, block, budgets),
        color=PlotInk.CYCLE_0,
        label=DisplayText("Pareto(2.5) illustration of the block-max map"),
    )
    vertical_guide(
        axes,
        1.0 / block,
        style=PlotLineStyle.DOTTED,
        width=0.8,
        label=ReferenceAnnotation.ONE_ALERT_PER_BLOCK.label,
    )
    for alpha in AlphaLevel:
        vertical_guide(
            axes,
            BudgetGuide.HALF_LOCAL.budget(alpha, reserve),
            ink=PlotInk.CYCLE_3,
            width=0.8,
            label=BudgetGuide.HALF_LOCAL.text(alpha),
        )
        vertical_guide(
            axes,
            BudgetGuide.LOCAL.budget(alpha, reserve),
            ink=PlotInk.CYCLE_2,
            width=0.8,
            label=BudgetGuide.LOCAL.text(alpha),
        )
    axes.set_xlabel("Calibration budget")
    axes.set_ylabel("Block-max threshold")
    axes.legend(frameon=False, fontsize=7)
    return figure
