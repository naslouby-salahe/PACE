from pace.reporting.figure_canvas import corpus_label, primary_panels, stratum_label
from pace.reporting.records import ReportInputs, method_rates
from pace.reporting.text import resampling_rule_text
from pace.types import (
    BudgetAnalysisKind,
    DisplayText,
    FigureLimit,
    FigureName,
    ReportedMethod,
)


def _population(inputs: ReportInputs) -> DisplayText:
    strata = {item.stratum for item in inputs.evidence}
    if strata and all(stratum.is_confirmation for stratum in strata):
        targets = len(inputs.cohort(BudgetAnalysisKind.UNIVERSAL_PRIMARY))
        corpus = corpus_label(next(iter(strata)))
        return DisplayText(f"Independent confirmation cohort ({targets} {corpus} targets)")
    if len(strata) == 1:
        return DisplayText(f"Primary population: {stratum_label(next(iter(strata)))}")
    return DisplayText("Primary population")


def _is_saturated(inputs: ReportInputs) -> bool:
    records = tuple(record for panel in primary_panels(inputs) for record in panel.records)
    floor = 1.0 - FigureLimit.NEGLIGIBLE_RATE.size
    return len(records) > 0 and all(
        method_rates(method, record).detection >= floor
        for method in ReportedMethod
        for record in records
    )


def figure_title(name: FigureName, inputs: ReportInputs) -> DisplayText:
    title = _population(inputs)
    if name.has_interval_estimates:
        title = DisplayText(f"{title}\nIntervals: {resampling_rule_text()}")
    if name.is_stratum_level and _is_saturated(inputs):
        title = DisplayText(f"{title}\nsaturated: every method detects all attack rows")
    return title
