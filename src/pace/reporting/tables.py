import json
from collections import defaultdict
from collections.abc import Callable

import numpy as np
import structlog
from pydantic import BaseModel, JsonValue
from scipy.stats import binomtest

from pace.reporting.acceptance import acceptance_table
from pace.reporting.metrics import paired_minimum_detectable_effect, paired_power_at_effect
from pace.reporting.records import (
    ReportInputs,
    ResultTable,
    TargetRecord,
    alpha_text,
    analysis_text,
    count,
    decomposition_of,
    interval_cells,
    mean_of,
    method_false_alert,
    number,
    optional_number,
    pooled_difference,
    sensitivity_rows,
    summarize_budget_effects,
    verdict_text,
)
from pace.types import (
    AcceptanceRule,
    AlphaLevel,
    BudgetAnalysisKind,
    BudgetAnalysisScope,
    ClientName,
    ColumnHeading,
    ConfigurationSection,
    DatasetName,
    DetectionContribution,
    DisplayText,
    FalseAlertGuide,
    LeaveOneOutScope,
    LogEvent,
    ObservationCount,
    PValue,
    RateDifference,
    ReportedMethod,
    StudyArm,
    StudyStratum,
    SummaryQuantile,
    TableName,
)


def headline_table(inputs: ReportInputs) -> ResultTable:
    rows = tuple(
        (
            analysis_text(cell.kind),
            alpha_text(cell.alpha),
            count(cell.local_training_rows),
            count(cell.target_count),
            *interval_cells(cell.nominal_gain, cell.nominal_interval),
            *interval_cells(cell.matched_gain, cell.matched_interval),
            *interval_cells(cell.gain_without_excluded, cell.gain_without_excluded_interval),
            count(cell.harmed_target_count),
            count(cell.severe_target_count),
            number(cell.worst_target_difference),
            number(cell.false_alert_increase),
            verdict_text(cell.is_accepted),
        )
        for cell in inputs.acceptance.cells
    )
    return ResultTable(
        TableName.HEADLINE,
        (
            ColumnHeading.ANALYSIS,
            ColumnHeading.ALPHA,
            ColumnHeading.LOCAL_ROWS,
            ColumnHeading.TARGETS,
            ColumnHeading.NOMINAL_GAIN,
            ColumnHeading.NOMINAL_LOWER,
            ColumnHeading.NOMINAL_UPPER,
            ColumnHeading.MATCHED_GAIN,
            ColumnHeading.MATCHED_LOWER,
            ColumnHeading.MATCHED_UPPER,
            ColumnHeading.GAIN_WITHOUT_EXCLUDED,
            ColumnHeading.WITHOUT_LOWER,
            ColumnHeading.WITHOUT_UPPER,
            ColumnHeading.HARMED_TARGETS,
            ColumnHeading.SEVERE_TARGETS,
            ColumnHeading.WORST_TARGET,
            ColumnHeading.FALSE_ALERT_INCREASE,
            ColumnHeading.ACCEPTED,
        ),
        rows,
    )


def _outcome_counts(
    gains: tuple[RateDifference, ...],
) -> tuple[ObservationCount, ObservationCount, ObservationCount, PValue]:
    band = AcceptanceRule.TIE_BAND.threshold
    wins = sum(gain > band for gain in gains)
    losses = sum(gain < -band for gain in gains)
    decided = wins + losses
    p_value = binomtest(wins, decided).pvalue if decided else 1.0
    return wins, len(gains) - wins - losses, losses, p_value


def strata_table(inputs: ReportInputs) -> ResultTable:
    settings = inputs.configuration.reporting
    rows: list[tuple[DisplayText, ...]] = []
    for alpha in inputs.alphas():
        for budget in inputs.budgets(BudgetAnalysisKind.UNIVERSAL_PRIMARY):
            grouped: dict[StudyStratum, list[TargetRecord]] = defaultdict(list)
            for record in inputs.primary(alpha, budget):
                grouped[record.stratum].append(record)
            for stratum in sorted(grouped, key=lambda item: item.name):
                members = tuple(grouped[stratum])
                gain, interval = pooled_difference(members, lambda item: item.gain, settings)
                wins, ties, losses, p_value = _outcome_counts(tuple(item.gain for item in members))
                rows.append(
                    (
                        DisplayText(stratum.name),
                        alpha_text(alpha),
                        count(budget),
                        count(len(members)),
                        number(mean_of(tuple(item.pace.detection for item in members))),
                        number(mean_of(tuple(item.mean_ensemble.detection for item in members))),
                        number(gain),
                        number(interval.lower_bound),
                        number(interval.upper_bound),
                        number(np.median([item.gain for item in members]).item()),
                        count(wins),
                        count(ties),
                        count(losses),
                        number(p_value),
                    )
                )
    return ResultTable(
        TableName.STRATA,
        (
            ColumnHeading.STRATUM,
            ColumnHeading.ALPHA,
            ColumnHeading.LOCAL_ROWS,
            ColumnHeading.TARGETS,
            ColumnHeading.PACE_DETECTION,
            ColumnHeading.MEAN_ENSEMBLE_DETECTION,
            ColumnHeading.GAIN,
            ColumnHeading.LOWER,
            ColumnHeading.UPPER,
            ColumnHeading.MEDIAN_GAIN,
            ColumnHeading.WINS,
            ColumnHeading.TIES,
            ColumnHeading.LOSSES,
            ColumnHeading.SIGN_TEST_P,
        ),
        tuple(rows),
    )


def targets_table(inputs: ReportInputs) -> ResultTable:
    ordered = sorted(
        inputs.store.records,
        key=lambda item: (item.stratum.name, item.target.name, item.alpha.fraction, item.rows),
    )
    return ResultTable(
        TableName.TARGETS,
        (
            ColumnHeading.STRATUM,
            ColumnHeading.TARGET,
            ColumnHeading.ALPHA,
            ColumnHeading.LOCAL_ROWS,
            ColumnHeading.BENIGN_TEST_ROWS,
            ColumnHeading.PACE_DETECTION,
            ColumnHeading.PACE_FALSE_ALERT,
            ColumnHeading.MEAN_ENSEMBLE_DETECTION,
            ColumnHeading.MEAN_ENSEMBLE_FALSE_ALERT,
            ColumnHeading.LOCAL_DETECTION,
            ColumnHeading.PEER_UNION_DETECTION,
            ColumnHeading.MAX_FUSION_DETECTION,
            ColumnHeading.BONFERRONI_DETECTION,
            ColumnHeading.GAIN,
        ),
        tuple(
            (
                DisplayText(record.stratum.name),
                DisplayText(record.target.name),
                alpha_text(record.alpha),
                count(record.rows),
                count(record.benign_test_rows),
                number(record.pace.detection),
                number(record.pace.false_alert),
                number(record.mean_ensemble.detection),
                number(record.mean_ensemble.false_alert),
                number(record.local.detection),
                number(record.peer_union.detection),
                number(record.max_fusion.detection),
                number(record.bonferroni.detection),
                number(record.gain),
            )
            for record in ordered
        ),
    )


def ablations_table(inputs: ReportInputs) -> ResultTable:
    settings = inputs.configuration.reporting
    rows: list[tuple[DisplayText, ...]] = []
    for alpha in inputs.alphas():
        for budget in inputs.budgets(BudgetAnalysisKind.UNIVERSAL_PRIMARY):
            members = inputs.primary(alpha, budget)
            for arm in inputs.configuration.scientific.study_arms:
                gain, interval = pooled_difference(
                    members,
                    lambda item, arm=arm: item.pace.detection - item.arm(arm).detection,
                    settings,
                )
                rows.append(
                    (
                        alpha_text(alpha),
                        count(budget),
                        arm.display_label,
                        count(len(members)),
                        number(mean_of(tuple(item.arm(arm).detection for item in members))),
                        number(gain),
                        number(interval.lower_bound),
                        number(interval.upper_bound),
                        number(mean_of(tuple(item.arm(arm).false_alert for item in members))),
                        number(mean_of(tuple(item.pace.false_alert for item in members))),
                    )
                )
    return ResultTable(
        TableName.ABLATIONS,
        (
            ColumnHeading.ALPHA,
            ColumnHeading.LOCAL_ROWS,
            ColumnHeading.ARM,
            ColumnHeading.TARGETS,
            ColumnHeading.ARM_DETECTION,
            ColumnHeading.GAIN,
            ColumnHeading.LOWER,
            ColumnHeading.UPPER,
            ColumnHeading.ARM_FALSE_ALERT,
            ColumnHeading.PACE_FALSE_ALERT,
        ),
        tuple(rows),
    )


logger = structlog.get_logger()


def scaling_table(inputs: ReportInputs) -> ResultTable:
    settings = inputs.configuration.reporting
    rows: list[tuple[DisplayText, ...]] = []
    for kind in BudgetAnalysisKind:
        cohort = inputs.cohort(kind)
        for alpha in inputs.alphas():
            for budget in inputs.budgets(kind):
                members = inputs.store.select(alpha, budget, cohort)
                if not members:
                    continue
                gain, interval = pooled_difference(members, lambda item: item.gain, settings)
                rows.append(
                    (
                        analysis_text(kind),
                        alpha_text(alpha),
                        count(budget),
                        count(len(members)),
                        number(mean_of(tuple(item.pace.detection for item in members))),
                        number(mean_of(tuple(item.mean_ensemble.detection for item in members))),
                        number(gain),
                        number(interval.lower_bound),
                        number(interval.upper_bound),
                    )
                )
    return ResultTable(
        TableName.SCALING,
        (
            ColumnHeading.ANALYSIS,
            ColumnHeading.ALPHA,
            ColumnHeading.LOCAL_ROWS,
            ColumnHeading.TARGETS,
            ColumnHeading.PACE_DETECTION,
            ColumnHeading.MEAN_ENSEMBLE_DETECTION,
            ColumnHeading.GAIN,
            ColumnHeading.LOWER,
            ColumnHeading.UPPER,
        ),
        tuple(rows),
    )


def false_alert_rates_table(inputs: ReportInputs) -> ResultTable:
    rows: list[tuple[DisplayText, ...]] = []
    for alpha in inputs.alphas():
        for budget in inputs.budgets(BudgetAnalysisKind.UNIVERSAL_PRIMARY):
            members = inputs.primary(alpha, budget)
            for method in ReportedMethod:
                rates = np.asarray([method_false_alert(method, item) for item in members])
                rows.append(
                    (
                        method.display_label,
                        alpha_text(alpha),
                        count(budget),
                        number(np.mean(rates).item()),
                        number(np.median(rates).item()),
                        number(
                            np.quantile(
                                rates,
                                SummaryQuantile.UPPER_DECILE.level,
                                method="linear",
                            ).item()
                        ),
                        number(np.max(rates).item()),
                        number(
                            np.mean(rates > FalseAlertGuide.NOMINAL.factor * alpha.fraction).item()
                        ),
                        number(
                            np.mean(rates > FalseAlertGuide.TWICE.factor * alpha.fraction).item()
                        ),
                    )
                )
    return ResultTable(
        TableName.FALSE_ALERT_RATES,
        (
            ColumnHeading.METHOD,
            ColumnHeading.ALPHA,
            ColumnHeading.LOCAL_ROWS,
            ColumnHeading.MEAN,
            ColumnHeading.MEDIAN,
            ColumnHeading.P90,
            ColumnHeading.WORST,
            ColumnHeading.SHARE_ABOVE_ALPHA,
            ColumnHeading.SHARE_ABOVE_TWICE_ALPHA,
        ),
        tuple(rows),
    )


def seeds_table(inputs: ReportInputs) -> ResultTable:
    excluded = DatasetName.TON_IOT
    rows: list[tuple[DisplayText, ...]] = []
    for alpha in inputs.alphas():
        for budget in inputs.budgets(BudgetAnalysisKind.UNIVERSAL_PRIMARY):
            members = inputs.primary(alpha, budget)
            for seed in inputs.store.seeds():
                gains = tuple(
                    item.difference
                    for record in members
                    for item in record.seed_gains
                    if item.seed == seed
                )
                others = tuple(
                    item.difference
                    for record in members
                    if record.target.dataset is not excluded
                    for item in record.seed_gains
                    if item.seed == seed
                )
                rows.append(
                    (
                        alpha_text(alpha),
                        count(budget),
                        count(seed),
                        count(len(members)),
                        number(mean_of(gains)),
                        optional_number(mean_of(others) if others else None),
                    )
                )
    return ResultTable(
        TableName.SEEDS,
        (
            ColumnHeading.ALPHA,
            ColumnHeading.LOCAL_ROWS,
            ColumnHeading.SEED,
            ColumnHeading.TARGETS,
            ColumnHeading.GAIN,
            ColumnHeading.GAIN_WITHOUT_EXCLUDED,
        ),
        tuple(rows),
    )


def _stratum_name(record: TargetRecord) -> DisplayText:
    return DisplayText(record.stratum.name)


def _target_name(record: TargetRecord) -> DisplayText:
    return DisplayText(record.target.name)


def sensitivity_table(inputs: ReportInputs) -> ResultTable:
    scopes: tuple[tuple[LeaveOneOutScope, Callable[[TargetRecord], DisplayText]], ...] = (
        (LeaveOneOutScope.STRATUM, _stratum_name),
        (LeaveOneOutScope.TARGET, _target_name),
    )
    rows: list[tuple[DisplayText, ...]] = []
    for alpha in inputs.alphas():
        for budget in inputs.budgets(BudgetAnalysisKind.UNIVERSAL_PRIMARY):
            members = inputs.primary(alpha, budget)
            for scope, group in scopes:
                rows.extend(
                    (alpha_text(alpha), count(budget), scope.column_label, name, number(gain))
                    for name, gain in sensitivity_rows(members, group)
                )
    return ResultTable(
        TableName.SENSITIVITY,
        (
            ColumnHeading.ALPHA,
            ColumnHeading.LOCAL_ROWS,
            ColumnHeading.SCOPE,
            ColumnHeading.LEFT_OUT,
            ColumnHeading.GAIN,
        ),
        tuple(rows),
    )


def decomposition_table(inputs: ReportInputs) -> ResultTable:
    rows: list[tuple[DisplayText, ...]] = []
    for alpha in inputs.alphas():
        for budget in inputs.budgets(BudgetAnalysisKind.UNIVERSAL_PRIMARY):
            grouped: dict[StudyStratum, list[TargetRecord]] = defaultdict(list)
            for record in inputs.primary(alpha, budget):
                grouped[record.stratum].append(record)
            for stratum in sorted(grouped, key=lambda item: item.name):
                members = tuple(grouped[stratum])
                parts = tuple(decomposition_of(item) for item in members)
                rows.append(
                    (
                        DisplayText(stratum.name),
                        alpha_text(alpha),
                        count(budget),
                        *(
                            number(mean_of(tuple(part.share(contribution) for part in parts)))
                            for contribution in DetectionContribution
                        ),
                    )
                )
    return ResultTable(
        TableName.DECOMPOSITION,
        (
            ColumnHeading.STRATUM,
            ColumnHeading.ALPHA,
            ColumnHeading.LOCAL_ROWS,
            ColumnHeading.PEERS_ONLY,
            ColumnHeading.LOCAL_ONLY,
            ColumnHeading.BOTH,
            ColumnHeading.NEITHER,
        ),
        tuple(rows),
    )


def false_alert_decomposition_table(inputs: ReportInputs) -> ResultTable:
    rows: list[tuple[DisplayText, ...]] = []
    for alpha in inputs.alphas():
        for budget in inputs.budgets(BudgetAnalysisKind.UNIVERSAL_PRIMARY):
            grouped: dict[StudyStratum, list[TargetRecord]] = defaultdict(list)
            for record in inputs.primary(alpha, budget):
                grouped[record.stratum].append(record)
            for stratum in sorted(grouped, key=lambda item: item.name):
                members = grouped[stratum]
                rows.append(
                    (
                        DisplayText(stratum.name),
                        alpha_text(alpha),
                        count(budget),
                        number(mean_of(tuple(item.peer_union.false_alert for item in members))),
                        number(mean_of(tuple(item.local.false_alert for item in members))),
                        number(mean_of(tuple(item.pace.false_alert for item in members))),
                    )
                )
    return ResultTable(
        TableName.FALSE_ALERT_DECOMPOSITION,
        (
            ColumnHeading.STRATUM,
            ColumnHeading.ALPHA,
            ColumnHeading.LOCAL_ROWS,
            ColumnHeading.PEER_BRANCH,
            ColumnHeading.LOCAL_BRANCH,
            ColumnHeading.COMBINED,
        ),
        tuple(rows),
    )


def census_table(inputs: ReportInputs) -> ResultTable:
    reference = {(record.stratum, record.target.name): record for record in inputs.store.records}
    rows = tuple(
        (
            DisplayText(item.stratum.name),
            DisplayText(target.client),
            count(target.training_rows),
            count(target.attack_type_count),
            count(record.benign_test_rows),
            count(record.peer_count),
        )
        for item in inputs.training
        for target in sorted(item.targets, key=lambda entry: entry.client)
        if (record := reference.get((item.stratum, ClientName(target.client)))) is not None
    )
    return ResultTable(
        TableName.CENSUS,
        (
            ColumnHeading.STRATUM,
            ColumnHeading.TARGET,
            ColumnHeading.TRAINING_ROWS,
            ColumnHeading.ATTACK_FAMILIES,
            ColumnHeading.BENIGN_TEST_ROWS,
            ColumnHeading.PEERS,
        ),
        rows,
    )


def _setting_labels() -> dict[DisplayText, DisplayText]:
    labels: dict[DisplayText, DisplayText] = {}
    labels.update({DisplayText(item.name): DisplayText(f"{item.fraction}") for item in AlphaLevel})
    labels.update({DisplayText(item.name): item.display_label for item in StudyArm})
    labels.update({DisplayText(item.name): item.display_label for item in StudyStratum})
    return labels


def _setting_text(text: DisplayText) -> DisplayText:
    known = _setting_labels().get(DisplayText(text.upper()))
    if known is not None:
        return known
    if text.isupper():
        return DisplayText(text.lower().replace("_", " "))
    return DisplayText(text)


def _setting_value(value: JsonValue) -> DisplayText:
    if isinstance(value, dict):
        return DisplayText(
            " // ".join(
                f"{_setting_text(DisplayText(key))}: {_setting_value(item)}"
                for key, item in value.items()
            )
        )
    if isinstance(value, list):
        return DisplayText("; ".join(_setting_value(item) for item in value))
    if isinstance(value, str):
        return _setting_text(DisplayText(value))
    return DisplayText(f"{value}")


def _flatten(prefix: DisplayText, value: JsonValue) -> tuple[tuple[DisplayText, DisplayText], ...]:
    if isinstance(value, dict):
        return tuple(
            pair
            for key, item in value.items()
            for pair in _flatten(
                DisplayText(f"{prefix}.{_setting_text(DisplayText(key))}" if prefix else f"{key}"),
                item,
            )
        )
    return ((prefix, _setting_value(value)),)


def _section(model: BaseModel) -> JsonValue:
    payload: JsonValue = json.loads(model.model_dump_json())
    return payload


def configuration_table(inputs: ReportInputs) -> ResultTable:
    configuration = inputs.configuration
    sections = (
        (ConfigurationSection.SCIENTIFIC, configuration.scientific),
        (ConfigurationSection.SPLITTING, configuration.splitting),
        (ConfigurationSection.REPORTING, configuration.reporting),
        (ConfigurationSection.PROTOCOL_MATRIX, configuration.protocol_matrix),
    )
    rows = tuple(
        pair for name, model in sections for pair in _flatten(DisplayText(name), _section(model))
    )
    return ResultTable(TableName.CONFIGURATION, (ColumnHeading.SETTING, ColumnHeading.VALUE), rows)


def power_table(inputs: ReportInputs) -> ResultTable:
    protocol = inputs.configuration.protocol_matrix
    rows = tuple(
        (
            analysis_text(kind),
            count(len(cohort)),
            number(sd),
            number(paired_minimum_detectable_effect(len(cohort), sd, protocol.power)),
            number(paired_power_at_effect(len(cohort), sd, protocol.power)),
        )
        for kind in BudgetAnalysisKind
        if (cohort := inputs.cohort(kind))
        for sd in protocol.power.difference_sds
    )
    return ResultTable(
        TableName.POWER,
        (
            ColumnHeading.ANALYSIS,
            ColumnHeading.TARGETS,
            ColumnHeading.ASSUMED_SD,
            ColumnHeading.MINIMUM_DETECTABLE_EFFECT,
            ColumnHeading.POWER_AT_MEANINGFUL_EFFECT,
        ),
        rows,
    )


def budget_effects_table(inputs: ReportInputs) -> ResultTable:
    protocol = inputs.configuration.protocol_matrix
    report = summarize_budget_effects(
        inputs.evidence,
        tuple(spec.kind for spec in protocol.analyses),
        inputs.configuration.reporting,
    )
    rows = tuple(
        (
            analysis_text(analysis.kind),
            alpha_text(comparison.alpha),
            count(comparison.from_rows),
            count(comparison.to_rows),
            count(comparison.target_count),
            difference.outcome.display_label,
            *interval_cells(difference.mean_difference, difference.interval),
        )
        for analysis in report.analyses
        if analysis.scope is BudgetAnalysisScope.POOLED
        for comparison in analysis.comparisons
        for difference in comparison.pooled
    )
    return ResultTable(
        TableName.BUDGET_EFFECTS,
        (
            ColumnHeading.ANALYSIS,
            ColumnHeading.ALPHA,
            ColumnHeading.FROM_ROWS,
            ColumnHeading.TO_ROWS,
            ColumnHeading.TARGETS,
            ColumnHeading.OUTCOME,
            ColumnHeading.GAIN,
            ColumnHeading.LOWER,
            ColumnHeading.UPPER,
        ),
        rows,
    )


def build_tables(inputs: ReportInputs) -> tuple[ResultTable, ...]:
    tables = (
        headline_table(inputs),
        acceptance_table(inputs),
        strata_table(inputs),
        targets_table(inputs),
        ablations_table(inputs),
        scaling_table(inputs),
        false_alert_rates_table(inputs),
        seeds_table(inputs),
        sensitivity_table(inputs),
        decomposition_table(inputs),
        false_alert_decomposition_table(inputs),
        census_table(inputs),
        configuration_table(inputs),
        power_table(inputs),
        budget_effects_table(inputs),
    )
    logger.info(LogEvent.TABLES_WRITTEN, table_count=len(tables))
    return tables
