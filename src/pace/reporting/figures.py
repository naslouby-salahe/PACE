from collections import defaultdict
from pathlib import Path

import numpy as np
import structlog
from matplotlib.figure import Figure

from pace.reporting.figure_canvas import (
    horizontal_guide,
    legend_outside,
    new_figure,
    panel_grid,
    primary_panels,
    vertical_guide,
)
from pace.reporting.figure_diagnostics import census, power, resolution_mechanism
from pace.reporting.records import (
    MethodRates,
    ReportInputs,
    TargetRecord,
    decomposition_of,
    method_false_alert,
    pooled_difference,
    sensitivity_rows,
)
from pace.types import (
    AcceptanceCell,
    BudgetAnalysisKind,
    DatasetName,
    DetectionContribution,
    DisplayText,
    FalseAlertGuide,
    FigureName,
    FigureResolution,
    FloatArray,
    GainSeries,
    HexColor,
    ImageFormat,
    LearningSeed,
    LeaveOneOutScope,
    LogEvent,
    PdfMetadataField,
    PlotInk,
    PlotLegendAnchor,
    PlotMarker,
    PlotPosition,
    PlotStepWhere,
    RateConfidenceInterval,
    RateDifference,
    ReferenceAnnotation,
    ReportedMethod,
    SeedPanel,
    StudyStratum,
    UnavailableReading,
)

logger = structlog.get_logger()


def save_figure(figure: Figure, name: FigureName, directory: Path) -> tuple[Path, ...]:
    directory.mkdir(parents=True, exist_ok=True)
    pdf = directory / f"{name}.{ImageFormat.PDF}"
    png = directory / f"{name}.{ImageFormat.PNG}"
    figure.savefig(
        pdf,
        format=ImageFormat.PDF,
        metadata=dict.fromkeys(
            (
                PdfMetadataField.CREATOR,
                PdfMetadataField.PRODUCER,
                PdfMetadataField.CREATION_DATE,
            )
        ),
    )
    figure.savefig(
        png,
        format=ImageFormat.PNG,
        dpi=FigureResolution.PNG,
        metadata=dict.fromkeys((PdfMetadataField.SOFTWARE,)),
    )
    logger.info(LogEvent.FIGURE_WRITTEN, figure=name, directory=directory.as_posix())
    return pdf, png


def stratum_color(record: TargetRecord) -> HexColor:
    return record.stratum.color


def _method_rates(method: ReportedMethod, record: TargetRecord) -> MethodRates:
    match method:
        case ReportedMethod.PACE:
            return record.pace
        case ReportedMethod.MEAN_ENSEMBLE:
            return record.mean_ensemble
        case ReportedMethod.LOCAL_BRANCH:
            return record.local
        case ReportedMethod.PEER_UNION:
            return record.peer_union
        case ReportedMethod.MAX_FUSION:
            return record.max_fusion
        case ReportedMethod.BONFERRONI:
            return record.bonferroni


def _mean(values: tuple[RateDifference, ...]) -> RateDifference:
    return np.mean(values).item()


def operating_curves(inputs: ReportInputs) -> Figure:
    panels = primary_panels(inputs)
    figure = new_figure(10.0, 3.6 * ((len(panels) + 1) // 2))
    for panel, axes in panel_grid(figure, panels):
        multipliers = [point.multiplier for point in panel.records[0].sweep]
        false_alert = [
            _mean(tuple(item.sweep[index].false_alert_rate for item in panel.records))
            for index in range(len(multipliers))
        ]
        detection = [
            _mean(tuple(item.sweep[index].detection_rate for item in panel.records))
            for index in range(len(multipliers))
        ]
        axes.plot(
            false_alert,
            detection,
            marker=PlotMarker.CIRCLE,
            color=ReportedMethod.PACE.color,
            label=ReferenceAnnotation.PACE_SWEEP.label,
        )
        for method in ReportedMethod:
            rates = [_method_rates(method, item) for item in panel.records]
            axes.scatter(
                _mean(tuple(item.false_alert for item in rates)),
                _mean(tuple(item.detection for item in rates)),
                color=method.color,
                s=40,
                label=method.display_label,
            )
        vertical_guide(axes, panel.alpha.fraction)
        axes.set_xlabel("Mean realised false-alert rate")
        axes.set_ylabel("Mean detection rate")
    handles, _ = figure.axes[0].get_legend_handles_labels()
    figure.legend(handles=handles, loc=PlotLegendAnchor.OUTSIDE_RIGHT, fontsize=8, frameon=False)
    return figure


def false_alert_distribution(inputs: ReportInputs) -> Figure:
    panels = primary_panels(inputs)
    figure = new_figure(10.0, 3.4 * ((len(panels) + 1) // 2))
    methods = (ReportedMethod.PACE, ReportedMethod.MEAN_ENSEMBLE, ReportedMethod.MAX_FUSION)
    for panel, axes in panel_grid(figure, panels):
        for method in methods:
            rates = np.sort([method_false_alert(method, item) for item in panel.records])
            axes.step(
                rates,
                np.arange(1, len(rates) + 1) / len(rates),
                where=PlotStepWhere.POST,
                color=method.color,
                label=method.display_label,
            )
        for guide in FalseAlertGuide:
            vertical_guide(axes, guide.factor * panel.alpha.fraction)
        axes.set_xlabel("Realised false-alert rate per target")
        axes.set_ylabel("Cumulative share of targets")
        legend_outside(axes)
    return figure


def detection_decomposition(inputs: ReportInputs) -> Figure:
    panels = primary_panels(inputs)
    figure = new_figure(10.0, 3.6 * ((len(panels) + 1) // 2))
    for panel, axes in panel_grid(figure, panels):
        strata = sorted({item.stratum for item in panel.records}, key=lambda item: item.name)
        bottom = np.zeros(len(strata))
        for part in DetectionContribution.stack_order():
            heights = np.asarray(
                [
                    _mean(
                        tuple(
                            decomposition_of(item).share(part)
                            for item in panel.records
                            if item.stratum is stratum
                        )
                    )
                    for stratum in strata
                ]
            )
            axes.bar(range(len(strata)), heights, bottom=bottom, label=part.label)
            bottom = bottom + heights
        axes.set_xticks(range(len(strata)), [item.display_label for item in strata], rotation=45)
        axes.set_ylabel("Unweighted mean across attack families")
        legend_outside(axes)
    return figure


def family_gains(inputs: ReportInputs) -> Figure:
    panels = primary_panels(inputs)
    figure = new_figure(10.0, 3.6 * ((len(panels) + 1) // 2))
    for panel, axes in panel_grid(figure, panels):
        strata = sorted({item.stratum for item in panel.records}, key=lambda item: item.name)
        for index, stratum in enumerate(strata):
            gains = np.asarray(
                [
                    family.pace - family.mean_ensemble
                    for item in panel.records
                    if item.stratum is stratum
                    for family in item.families
                ]
            )
            jitter = np.random.default_rng(index).uniform(-0.25, 0.25, len(gains))
            axes.scatter(index + jitter, gains, s=6, color=stratum.color, alpha=0.7)
        axes.set_xticks(range(len(strata)), [item.display_label for item in strata], rotation=45)
        axes.axhline(0.0, color=PlotInk.BLACK, linewidth=0.8)
        axes.set_ylabel("Attack-family gain over mean-ensemble variant")
    return figure


def peer_drowning(inputs: ReportInputs) -> Figure:
    panels = primary_panels(inputs)
    figure = new_figure(10.0, 3.4 * ((len(panels) + 1) // 2))
    methods = (
        ReportedMethod.PACE,
        ReportedMethod.MAX_FUSION,
        ReportedMethod.PEER_UNION,
        ReportedMethod.BONFERRONI,
    )
    for panel, axes in panel_grid(figure, panels):
        for method in methods:
            differences = np.sort(
                [
                    _method_rates(method, item).detection - item.local.detection
                    for item in panel.records
                ]
            )
            axes.step(
                differences,
                np.arange(1, len(differences) + 1) / len(differences),
                where=PlotStepWhere.POST,
                color=method.color,
                label=method.display_label,
            )
        axes.axvline(0.0, color=PlotInk.BLACK, linewidth=0.8)
        axes.set_xlabel("Detection rate minus the PACE local branch alone")
        axes.set_ylabel("Cumulative share of targets")
        legend_outside(axes)
    return figure


def gain_against_peer_coverage(inputs: ReportInputs) -> Figure:
    panels = primary_panels(inputs)
    figure = new_figure(10.0, 3.4 * ((len(panels) + 1) // 2))
    for panel, axes in panel_grid(figure, panels):
        axes.scatter(
            [item.peer_union.detection for item in panel.records],
            [item.gain for item in panel.records],
            c=[stratum_color(item) for item in panel.records],
            s=16,
        )
        axes.axhline(0.0, color=PlotInk.BLACK, linewidth=0.8)
        axes.set_xlabel("Peer-union detection rate")
        axes.set_ylabel("Gain over mean-ensemble variant")
    return figure


def false_alert_decomposition(inputs: ReportInputs) -> Figure:
    panels = primary_panels(inputs)
    figure = new_figure(10.0, 3.6 * ((len(panels) + 1) // 2))
    series = (ReportedMethod.PEER_UNION, ReportedMethod.LOCAL_BRANCH, ReportedMethod.PACE)
    for panel, axes in panel_grid(figure, panels):
        strata: list[StudyStratum] = sorted(
            {item.stratum for item in panel.records}, key=lambda item: item.name
        )
        positions: FloatArray = np.arange(len(strata), dtype=np.float64)
        for offset, method in enumerate(series):
            values = [
                _mean(
                    tuple(
                        method_false_alert(method, item)
                        for item in panel.records
                        if item.stratum is stratum
                    )
                )
                for stratum in strata
            ]
            axes.bar(
                positions + 0.27 * (offset - 1),
                values,
                width=0.27,
                color=method.color,
                label=method.display_label,
            )
        horizontal_guide(axes, panel.alpha.fraction)
        axes.set_xticks(positions, [item.display_label for item in strata], rotation=45)
        axes.set_ylabel("Mean realised false-alert rate")
        legend_outside(axes)
    return figure


def _gain_series(
    cell: AcceptanceCell,
) -> tuple[tuple[GainSeries, RateDifference | None, RateConfidenceInterval | None], ...]:
    return tuple((series, *series.point(cell)) for series in GainSeries)


def headline_gain(inputs: ReportInputs) -> Figure:
    cells = tuple(
        cell
        for cell in inputs.acceptance.cells
        if cell.kind is BudgetAnalysisKind.UNIVERSAL_PRIMARY
    )
    figure = new_figure(8.0, 1.2 + 0.55 * len(cells) * 3)
    axes = figure.add_subplot(1, 1, 1)
    axes.grid(alpha=0.25)
    position = 0.0
    labels: list[DisplayText] = []
    ticks: list[PlotPosition] = []
    for cell in cells:
        for label, gain, interval in _gain_series(cell):
            if gain is None:
                labels.append(
                    DisplayText(
                        f"α={cell.alpha.fraction}, {cell.local_training_rows} rows: "
                        f"{label.label} ({UnavailableReading.NOT_AVAILABLE})"
                    )
                )
                ticks.append(position)
                position -= 1.0
                continue
            error = (
                [[0.0], [0.0]]
                if interval is None
                else [[gain - interval.lower_bound], [interval.upper_bound - gain]]
            )
            axes.errorbar(
                gain, position, xerr=error, fmt=PlotMarker.CIRCLE, color=label.ink, capsize=3
            )
            labels.append(
                DisplayText(
                    f"α={cell.alpha.fraction}, {cell.local_training_rows} rows: {label.label}"
                )
            )
            ticks.append(position)
            position -= 1.0
        position -= 0.6
    axes.axvline(0.0, color=PlotInk.BLACK, linewidth=0.8)
    axes.set_yticks(ticks, labels, fontsize=7)
    axes.set_xlabel("Detection-rate gain over mean-ensemble variant (95% interval)")
    return figure


def target_gains(inputs: ReportInputs) -> Figure:
    panels = primary_panels(inputs)
    figure = new_figure(9.0, 3.2 * ((len(panels) + 1) // 2))
    for panel, axes in panel_grid(figure, panels):
        ordered = sorted(panel.records, key=lambda item: item.gain)
        axes.scatter(
            range(len(ordered)),
            [item.gain for item in ordered],
            c=[stratum_color(item) for item in ordered],
            s=14,
        )
        axes.axhline(0.0, color=PlotInk.BLACK, linewidth=0.8)
        axes.set_xlabel("Target (sorted by gain)")
        axes.set_ylabel("Gain over mean-ensemble variant")
    handles = [
        figure.axes[0].scatter([], [], c=[stratum.color], s=14, label=stratum.display_label)
        for stratum in StudyStratum
    ]
    figure.legend(handles=handles, loc=PlotLegendAnchor.OUTSIDE_RIGHT, fontsize=7, frameon=False)
    return figure


def stratum_gains(inputs: ReportInputs) -> Figure:
    settings = inputs.configuration.reporting
    panels = primary_panels(inputs)
    figure = new_figure(9.0, 3.2 * ((len(panels) + 1) // 2))
    for panel, axes in panel_grid(figure, panels):
        grouped: dict[StudyStratum, list[TargetRecord]] = defaultdict(list)
        for record in panel.records:
            grouped[record.stratum].append(record)
        strata = sorted(grouped, key=lambda item: item.name)
        for index, stratum in enumerate(strata):
            gain, interval = pooled_difference(tuple(grouped[stratum]), lambda r: r.gain, settings)
            axes.bar(index, gain, color=stratum.color)
            axes.errorbar(
                index,
                gain,
                yerr=[[gain - interval.lower_bound], [interval.upper_bound - gain]],
                color=PlotInk.BLACK,
                capsize=3,
            )
        axes.set_xticks(range(len(strata)), [item.display_label for item in strata], rotation=45)
        axes.set_ylabel("Gain over mean-ensemble variant")
        axes.axhline(0.0, color=PlotInk.BLACK, linewidth=0.8)
    return figure


def ablations(inputs: ReportInputs) -> Figure:
    settings = inputs.configuration.reporting
    arms = inputs.configuration.scientific.study_arms
    panels = primary_panels(inputs)
    figure = new_figure(10.0, 3.6 * ((len(panels) + 1) // 2))
    for panel, axes in panel_grid(figure, panels):
        for index, arm in enumerate(arms):
            gain, interval = pooled_difference(
                panel.records, lambda r, arm=arm: r.pace.detection - r.arm(arm).detection, settings
            )
            axes.barh(index, gain, color=PlotInk.CYCLE_0)
            axes.errorbar(
                gain,
                index,
                xerr=[[gain - interval.lower_bound], [interval.upper_bound - gain]],
                color=PlotInk.BLACK,
                capsize=3,
            )
        axes.set_yticks(range(len(arms)), [arm.display_label for arm in arms], fontsize=7)
        axes.axvline(0.0, color=PlotInk.BLACK, linewidth=0.8)
        axes.set_xlabel("PACE detection rate minus arm")
    return figure


def budget_scaling(inputs: ReportInputs) -> Figure:
    settings = inputs.configuration.reporting
    figure = new_figure(9.0, 3.6)
    for index, alpha in enumerate(inputs.alphas(), start=1):
        axes = figure.add_subplot(1, len(inputs.alphas()), index)
        axes.set_title(DisplayText(f"α = {alpha.fraction}"))
        axes.grid(alpha=0.25)
        for kind in BudgetAnalysisKind:
            cohort = inputs.cohort(kind)
            budgets = [b for b in inputs.budgets(kind) if inputs.store.select(alpha, b, cohort)]
            summaries = [
                pooled_difference(inputs.store.select(alpha, b, cohort), lambda r: r.gain, settings)
                for b in budgets
            ]
            if not summaries:
                continue
            gains = np.asarray([gain for gain, _ in summaries])
            lower = np.asarray([interval.lower_bound for _, interval in summaries])
            upper = np.asarray([interval.upper_bound for _, interval in summaries])
            axes.plot(budgets, gains, marker=PlotMarker.CIRCLE, label=kind.title)
            axes.fill_between(budgets, lower, upper, alpha=0.15)
        axes.set_xscale("log")
        axes.axhline(0.0, color=PlotInk.BLACK, linewidth=0.8)
        axes.set_xlabel("Local training rows")
        axes.set_ylabel("Gain over mean-ensemble variant")
        axes.legend(frameon=False, fontsize=7)
    return figure


def _seed_gain(
    records: tuple[TargetRecord, ...], seed: LearningSeed, excluded: DatasetName | None
) -> RateDifference:
    return np.mean(
        [
            item.difference
            for record in records
            if record.target.dataset is not excluded
            for item in record.seed_gains
            if item.seed == seed
        ]
    ).item()


def seed_stability(inputs: ReportInputs) -> Figure:
    panels = primary_panels(inputs)
    figure = new_figure(9.0, 3.2 * ((len(panels) + 1) // 2))
    seeds = inputs.store.seeds()
    for panel, axes in panel_grid(figure, panels):
        for panel_series in SeedPanel:
            values = [
                _seed_gain(panel.records, seed, panel_series.excluded_dataset) for seed in seeds
            ]
            axes.plot(
                range(len(seeds)),
                values,
                marker=PlotMarker.CIRCLE,
                color=panel_series.ink,
                label=panel_series.label,
            )
        axes.set_xticks(range(len(seeds)), [f"{seed}" for seed in seeds])
        axes.axhline(0.0, color=PlotInk.BLACK, linewidth=0.8)
        axes.set_xlabel("Learning seed")
        axes.set_ylabel("Pooled gain")
        legend_outside(axes)
    return figure


def sensitivity(inputs: ReportInputs) -> Figure:
    panels = primary_panels(inputs)
    figure = new_figure(10.0, 3.4 * ((len(panels) + 1) // 2))
    for panel, axes in panel_grid(figure, panels):
        grouped = {
            LeaveOneOutScope.STRATUM: sensitivity_rows(
                panel.records, lambda record: DisplayText(record.stratum.name)
            ),
            LeaveOneOutScope.TARGET: sensitivity_rows(
                panel.records, lambda record: DisplayText(record.target.name)
            ),
        }
        for scope in LeaveOneOutScope:
            points = grouped[scope]
            axes.scatter(
                [scope.position] * len(points),
                [gain for _, gain in points],
                s=scope.marker_area,
                color=scope.ink,
                alpha=scope.opacity,
            )
        axes.set_xticks(
            [scope.position for scope in LeaveOneOutScope],
            [scope.axis_label for scope in LeaveOneOutScope],
        )
        axes.axhline(0.0, color=PlotInk.BLACK, linewidth=0.8)
        axes.set_ylabel("Pooled gain")
    return figure


def draw(name: FigureName, inputs: ReportInputs) -> Figure:
    logger.info(LogEvent.FIGURE_DRAWN, figure=name)
    match name:
        case FigureName.HEADLINE_GAIN:
            return headline_gain(inputs)
        case FigureName.TARGET_GAINS:
            return target_gains(inputs)
        case FigureName.STRATUM_GAINS:
            return stratum_gains(inputs)
        case FigureName.OPERATING_CURVES:
            return operating_curves(inputs)
        case FigureName.FALSE_ALERT_DISTRIBUTION:
            return false_alert_distribution(inputs)
        case FigureName.ABLATIONS:
            return ablations(inputs)
        case FigureName.DETECTION_DECOMPOSITION:
            return detection_decomposition(inputs)
        case FigureName.FAMILY_GAINS:
            return family_gains(inputs)
        case FigureName.BUDGET_SCALING:
            return budget_scaling(inputs)
        case FigureName.PEER_DROWNING:
            return peer_drowning(inputs)
        case FigureName.SEED_STABILITY:
            return seed_stability(inputs)
        case FigureName.SENSITIVITY:
            return sensitivity(inputs)
        case FigureName.POWER:
            return power(inputs)
        case FigureName.CENSUS:
            return census(inputs)
        case FigureName.RESOLUTION_MECHANISM:
            return resolution_mechanism(inputs)
        case FigureName.GAIN_AGAINST_PEER_COVERAGE:
            return gain_against_peer_coverage(inputs)
        case FigureName.FALSE_ALERT_DECOMPOSITION:
            return false_alert_decomposition(inputs)


def write_figures(
    inputs: ReportInputs, names: tuple[FigureName, ...], directory: Path
) -> tuple[Path, ...]:
    paths = tuple(
        path for name in names for path in save_figure(draw(name, inputs), name, directory)
    )
    logger.info(LogEvent.FIGURES_WRITTEN, figure_count=len(names), directory=directory.as_posix())
    return paths
