from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from textwrap import fill

import numpy as np
import structlog
from matplotlib.figure import Figure

from pace.reporting.calibration import calibration_sensitivity
from pace.reporting.figure_canvas import (
    axes_center,
    horizontal_guide,
    legend_beside,
    legend_outside,
    new_figure,
    panel_grid,
    primary_panels,
    stratum_ticks,
    vertical_guide,
)
from pace.reporting.figure_diagnostics import census, power, resolution_mechanism
from pace.reporting.figures_operating import comparable_operating_curves, matching_coverage
from pace.reporting.records import (
    MethodRates,
    ReportInputs,
    TargetRecord,
    decomposition_of,
    method_false_alert,
    pooled_difference,
    sensitivity_rows,
)
from pace.reporting.scaling import scaling_points
from pace.types import (
    BudgetAnalysisKind,
    DatasetName,
    DetectionContribution,
    DisplayText,
    FalseAlertGuide,
    FigureExtent,
    FigureName,
    FigureResolution,
    FloatArray,
    FprLevel,
    HexColor,
    ImageFormat,
    LearningSeed,
    LeaveOneOutScope,
    LogEvent,
    PdfMetadataField,
    PlotInk,
    PlotLegendAnchor,
    PlotLineStyle,
    PlotMarker,
    PlotPosition,
    PlotStepWhere,
    RateDifference,
    ReferenceAnnotation,
    ReportedMethod,
    ScalingCohort,
    SeedPanel,
    StudyStratum,
    UnavailableReading,
)

logger = structlog.get_logger()


def _nominal_gain_label() -> DisplayText:
    return DisplayText("Gain vs same-calibration\nmean ensemble (nominal α)")


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
            vertical_guide(
                axes,
                guide.factor * panel.alpha.fraction,
                style=PlotLineStyle.DASHED
                if guide is FalseAlertGuide.NOMINAL
                else PlotLineStyle.DOTTED,
                label=DisplayText("α" if guide is FalseAlertGuide.NOMINAL else "2α"),
            )
        axes.set_xlabel("Realised false-alert rate per target")
        axes.set_ylabel("Cumulative share of targets")
    if panels:
        handles, _ = figure.axes[0].get_legend_handles_labels()
        figure.legend(
            handles=handles, loc=PlotLegendAnchor.OUTSIDE_RIGHT, fontsize=8, frameon=False
        )
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
        stratum_ticks(axes, range(len(strata)), [item.display_label for item in strata])
        axes.set_ylabel("Mean of per-target\nattack-family means")
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
        stratum_ticks(axes, range(len(strata)), [item.display_label for item in strata])
        axes.axhline(0.0, color=PlotInk.BLACK, linewidth=0.8)
        axes.set_ylabel("Attack-family gain vs\nsame-calibration mean\nensemble (nominal α)")
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
        axes.set_ylabel(_nominal_gain_label())
    if panels:
        strata = sorted(
            {record.stratum for panel in panels for record in panel.records}, key=lambda s: s.name
        )
        handles = [
            figure.axes[0].scatter([], [], c=[stratum.color], s=14, label=stratum.display_label)
            for stratum in strata
        ]
        figure.legend(
            handles=handles, loc=PlotLegendAnchor.OUTSIDE_RIGHT, fontsize=7, frameon=False
        )
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
        horizontal_guide(axes, panel.alpha.fraction, label=DisplayText("Nominal α"))
        stratum_ticks(axes, positions.tolist(), [item.display_label for item in strata])
        axes.set_ylabel("Mean realised false-alert rate")
    return figure


def headline_gain(inputs: ReportInputs) -> Figure:
    cells = tuple(
        cell
        for cell in (() if inputs.acceptance is None else inputs.acceptance.cells)
        if cell.kind is BudgetAnalysisKind.UNIVERSAL_PRIMARY
    )
    figure = new_figure(8.0, 2.0 + 0.55 * len(cells))
    axes = figure.add_subplot(1, 1, 1)
    axes.grid(alpha=0.25)
    position = 0.0
    labels: list[DisplayText] = []
    ticks: list[PlotPosition] = []
    for cell in cells:
        level = next(item for item in cell.levels if item.level is FprLevel.ALPHA)
        gain, interval = level.gain, level.interval
        if gain is not None and interval is not None:
            axes.errorbar(
                gain,
                position,
                xerr=[[gain - interval.lower_bound], [interval.upper_bound - gain]],
                fmt=PlotMarker.CIRCLE,
                color=PlotInk.CYCLE_0,
                capsize=3,
            )
        labels.append(
            DisplayText(
                f"α={cell.alpha.fraction}, N={cell.local_training_rows} "
                f"({level.matched_targets}/{cell.target_count} supported)"
                + (
                    f" — {UnavailableReading.NOT_AVAILABLE}"
                    if gain is None or interval is None
                    else ""
                )
            )
        )
        ticks.append(position)
        position -= 1.0
    axes.axvline(0.0, color=PlotInk.BLACK, linewidth=0.8)
    axes.set_yticks(ticks, labels, fontsize=7)
    axes.set_title("PACE vs same-calibration mean-ensemble local branch")
    confidence = inputs.configuration.reporting.confidence_level
    axes.set_xlabel(
        f"Detection-rate gain at realised FPR = α ({confidence:.0%} interval); N = training rows"
    )
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
        axes.set_ylabel(_nominal_gain_label())
    if panels:
        strata = sorted(
            {record.stratum for panel in panels for record in panel.records}, key=lambda s: s.name
        )
        handles = [
            figure.axes[0].scatter([], [], c=[stratum.color], s=14, label=stratum.display_label)
            for stratum in strata
        ]
        figure.legend(
            handles=handles, loc=PlotLegendAnchor.OUTSIDE_RIGHT, fontsize=7, frameon=False
        )
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
        stratum_ticks(axes, range(len(strata)), [item.display_label for item in strata])
        confidence = settings.confidence_level
        axes.set_ylabel(
            f"Gain vs same-calibration\nmean ensemble\n(nominal α; {confidence:.0%} CI)"
        )
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
        axes.set_yticks(
            range(len(arms)),
            [fill(arm.display_label, width=34) for arm in arms],
            fontsize=7,
        )
        axes.axvline(0.0, color=PlotInk.BLACK, linewidth=0.8)
        axes.set_xlabel(
            f"PACE detection rate minus arm\n(nominal α; {settings.confidence_level:.0%} CI)"
        )
    return figure


def _scaling_style(
    kind: BudgetAnalysisKind,
) -> tuple[PlotInk, PlotMarker, PlotLineStyle, FigureExtent]:
    match kind:
        case BudgetAnalysisKind.UNIVERSAL_PRIMARY:
            return PlotInk.CYCLE_0, PlotMarker.CIRCLE, PlotLineStyle.SOLID, 9.0
        case BudgetAnalysisKind.UNIVERSAL_ROBUSTNESS:
            return PlotInk.CYCLE_1, PlotMarker.SQUARE, PlotLineStyle.DASHED, 6.5
        case BudgetAnalysisKind.BROAD_SCALING:
            return PlotInk.CYCLE_2, PlotMarker.TRIANGLE, PlotLineStyle.DASHDOT, 6.0
        case BudgetAnalysisKind.HIGH_RESOURCE:
            return PlotInk.CYCLE_3, PlotMarker.DIAMOND, PlotLineStyle.DOTTED, 5.0


def budget_scaling(inputs: ReportInputs) -> Figure:
    settings = inputs.configuration.reporting
    points = scaling_points(inputs)
    alphas = inputs.alphas()
    cohorts = tuple(ScalingCohort)
    figure = new_figure(4.6 * len(alphas) + 2.6, 3.6 * len(cohorts))
    for row, cohort in enumerate(cohorts):
        for column, alpha in enumerate(alphas):
            axes = figure.add_subplot(len(cohorts), len(alphas), row * len(alphas) + column + 1)
            axes.set_title(DisplayText(f"α = {alpha.fraction}\n{cohort.label}"))
            axes.grid(alpha=0.25)
            for kind in BudgetAnalysisKind:
                ink, marker, style, size = _scaling_style(kind)
                chosen = sorted(
                    (
                        point
                        for point in points
                        if point.kind is kind
                        and point.alpha is alpha
                        and point.cohort is cohort
                        and point.gain is not None
                        and point.interval is not None
                    ),
                    key=lambda point: point.rows,
                )
                if not chosen:
                    continue
                rows = [point.rows for point in chosen]
                gains = [point.gain for point in chosen if point.gain is not None]
                bounds = [point.interval for point in chosen if point.interval is not None]
                axes.plot(
                    rows,
                    gains,
                    color=ink,
                    marker=marker,
                    linestyle=style,
                    markersize=size,
                    markerfacecolor="none",
                    label=DisplayText(f"{kind.title} ({chosen[0].target_count} targets)"),
                )
                axes.fill_between(
                    rows,
                    [item.lower_bound for item in bounds],
                    [item.upper_bound for item in bounds],
                    color=ink,
                    alpha=0.10,
                )
            axes.set_xscale("log")
            axes.axhline(0.0, color=PlotInk.BLACK, linewidth=0.8)
            axes.set_xlabel("Local training rows")
            axes.set_ylabel(
                "Matched-FPR gain at FPR = α\n"
                f"(PACE − same-calibration comparator;\n{settings.confidence_level:.0%} CI bands)"
            )
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
            excluded = panel_series.excluded_dataset
            if excluded is not None and all(
                record.target.dataset is not excluded for record in panel.records
            ):
                continue
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
        axes.set_ylabel(_nominal_gain_label())
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
        scopes = tuple(scope for scope in LeaveOneOutScope if grouped[scope])
        for scope in scopes:
            points = grouped[scope]
            axes.scatter(
                [scope.position] * len(points),
                [gain for _, gain in points],
                s=scope.marker_area,
                color=scope.ink,
                alpha=scope.opacity,
            )
        positions = [scope.position for scope in scopes]
        axes.set_xticks(positions, [scope.axis_label for scope in scopes])
        if positions:
            axes.set_xlim(min(positions) - 0.5, max(positions) + 0.5)
        if len(scopes) < len(LeaveOneOutScope):
            axes.set_title(f"{panel.title}\nStratum omission unavailable")
        axes.axhline(0.0, color=PlotInk.BLACK, linewidth=0.8)
        axes.set_ylabel(_nominal_gain_label())
    return figure


def _builders() -> dict[FigureName, Callable[[ReportInputs], Figure]]:
    return {
        FigureName.HEADLINE_GAIN: headline_gain,
        FigureName.TARGET_GAINS: target_gains,
        FigureName.STRATUM_GAINS: stratum_gains,
        FigureName.OPERATING_CURVES: operating_curves,
        FigureName.FALSE_ALERT_DISTRIBUTION: false_alert_distribution,
        FigureName.ABLATIONS: ablations,
        FigureName.DETECTION_DECOMPOSITION: detection_decomposition,
        FigureName.FAMILY_GAINS: family_gains,
        FigureName.BUDGET_SCALING: budget_scaling,
        FigureName.PEER_DROWNING: peer_drowning,
        FigureName.SEED_STABILITY: seed_stability,
        FigureName.SENSITIVITY: sensitivity,
        FigureName.POWER: power,
        FigureName.CENSUS: census,
        FigureName.RESOLUTION_MECHANISM: resolution_mechanism,
        FigureName.GAIN_AGAINST_PEER_COVERAGE: gain_against_peer_coverage,
        FigureName.FALSE_ALERT_DECOMPOSITION: false_alert_decomposition,
        FigureName.COMPARABLE_OPERATING_CURVES: comparable_operating_curves,
        FigureName.MATCHING_COVERAGE: matching_coverage,
        FigureName.CALIBRATION_SENSITIVITY: calibration_sensitivity,
    }


def draw(name: FigureName, inputs: ReportInputs) -> Figure:
    logger.info(LogEvent.FIGURE_DRAWN, figure=name)
    figure = _builders()[name](inputs)
    if name in (
        FigureName.DETECTION_DECOMPOSITION,
        FigureName.FALSE_ALERT_DECOMPOSITION,
        FigureName.BUDGET_SCALING,
    ):
        legend_beside(figure)
    if name not in (FigureName.RESOLUTION_MECHANISM, FigureName.CALIBRATION_SENSITIVITY):
        confirmation = inputs.evidence and all(
            item.stratum.is_confirmation for item in inputs.evidence
        )
        figure.suptitle(
            "Confirmation cohort" if confirmation else "Primary population",
            x=axes_center(figure),
        )
    return figure


def write_figures(
    inputs: ReportInputs, names: tuple[FigureName, ...], directory: Path
) -> tuple[Path, ...]:
    paths = tuple(
        path for name in names for path in save_figure(draw(name, inputs), name, directory)
    )
    logger.info(LogEvent.FIGURES_WRITTEN, figure_count=len(names), directory=directory.as_posix())
    return paths
