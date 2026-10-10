from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np

from pace.config import ReportingSettings, ResolvedConfiguration
from pace.experiment.preflight import MatrixVariant
from pace.reporting.metrics import (
    stratified_paired_interval,
)
from pace.types import (
    AcceptanceDesign,
    AcceptanceReport,
    AlphaLevel,
    AttackType,
    AttackTypePerformance,
    BudgetAnalysis,
    BudgetAnalysisKind,
    BudgetAnalysisScope,
    BudgetComparison,
    BudgetEffectAnalysis,
    BudgetEffectReport,
    BudgetOutcome,
    BudgetOutcomeDifference,
    BudgetSweepPerformance,
    CampaignMetricSummary,
    ClientName,
    ColumnHeading,
    CriterionSatisfied,
    DatasetClientId,
    DatasetValidationError,
    DetectionContribution,
    DetectionRate,
    DisplayText,
    FalseAlertRate,
    LearningSeed,
    LocalTrainingRowCount,
    ObservationCount,
    RateConfidenceInterval,
    RateDifference,
    ReportedMethod,
    SatisfiedAnswer,
    SeedRateDifference,
    StratumBudgetEffect,
    StratumFeasibility,
    StudyArm,
    StudyStratum,
    TableName,
    TargetMetricSummary,
    TargetPairedDifference,
    TargetPerformance,
    TargetTrainingRows,
)


@dataclass(frozen=True, slots=True)
class MethodRates:
    detection: DetectionRate
    false_alert: FalseAlertRate


@dataclass(frozen=True, slots=True)
class ArmRates:
    arm: StudyArm
    rates: MethodRates


@dataclass(frozen=True, slots=True)
class FamilyRates:
    attack_type: AttackType
    pace: DetectionRate
    mean_ensemble: DetectionRate
    local: DetectionRate
    peer_union: DetectionRate


@dataclass(frozen=True, slots=True)
class TargetRecord:
    stratum: StudyStratum
    target: DatasetClientId
    alpha: AlphaLevel
    rows: LocalTrainingRowCount
    benign_test_rows: ObservationCount
    peer_count: ObservationCount
    pace: MethodRates
    mean_ensemble: MethodRates
    local: MethodRates
    peer_union: MethodRates
    max_fusion: MethodRates
    bonferroni: MethodRates
    arms: tuple[ArmRates, ...]
    families: tuple[FamilyRates, ...]
    sweep: tuple[BudgetSweepPerformance, ...]
    seed_gains: tuple[SeedRateDifference, ...]

    @property
    def gain(self) -> RateDifference:
        return self.pace.detection - self.mean_ensemble.detection

    def arm(self, arm: StudyArm) -> MethodRates:
        return next(item.rates for item in self.arms if item.arm is arm)


def _mean(values: Iterable[RateDifference]) -> RateDifference:
    return np.mean(tuple(values)).item()


def _mean_detection(
    result: TargetPerformance, rate: Callable[[AttackTypePerformance], DetectionRate]
) -> DetectionRate:
    return _mean(rate(item) for item in result.attack_types)


def _arm_detection(result: TargetPerformance, arm: StudyArm) -> DetectionRate:
    return _mean(next(item for item in result.arms if item.arm is arm).detection_rates)


def _arm_false_alert(result: TargetPerformance, arm: StudyArm) -> FalseAlertRate:
    return next(item for item in result.arms if item.arm is arm).false_alert_rate


def _arm_rates(results: list[TargetPerformance], arm: StudyArm) -> MethodRates:
    return _rates(
        results,
        lambda result: _arm_detection(result, arm),
        lambda result: _arm_false_alert(result, arm),
    )


def _rates(
    results: list[TargetPerformance],
    detection: Callable[[TargetPerformance], DetectionRate],
    false_alert: Callable[[TargetPerformance], FalseAlertRate],
) -> MethodRates:
    return MethodRates(_mean(map(detection, results)), _mean(map(false_alert, results)))


def _families(results: list[TargetPerformance]) -> tuple[FamilyRates, ...]:
    mean_ensemble = StudyArm.MEAN_ENSEMBLE_LOCAL_BRANCH
    return tuple(
        FamilyRates(
            attack_type=first.attack_type,
            pace=_mean(result.attack_types[index].detection_rate for result in results),
            mean_ensemble=_mean(
                next(item for item in result.arms if item.arm is mean_ensemble).detection_rates[
                    index
                ]
                for result in results
            ),
            local=_mean(result.attack_types[index].local_detection_rate for result in results),
            peer_union=_mean(
                result.attack_types[index].peer_union_detection_rate for result in results
            ),
        )
        for index, first in enumerate(results[0].attack_types)
    )


def _sweep(results: list[TargetPerformance]) -> tuple[BudgetSweepPerformance, ...]:
    return tuple(
        BudgetSweepPerformance(
            multiplier=point.multiplier,
            detection_rate=_mean(result.budget_sweep[index].detection_rate for result in results),
            false_alert_rate=_mean(
                result.budget_sweep[index].false_alert_rate for result in results
            ),
        )
        for index, point in enumerate(results[0].budget_sweep)
    )


def target_record(
    results: list[TargetPerformance], alpha: AlphaLevel, rows: LocalTrainingRowCount
) -> TargetRecord:
    mean_ensemble = AcceptanceDesign.PROTOCOL.comparator
    first = results[0]
    return TargetRecord(
        stratum=first.study_stratum,
        target=first.target,
        alpha=alpha,
        rows=rows,
        benign_test_rows=first.benign_test_rows,
        peer_count=first.peer_count,
        pace=_rates(
            results,
            lambda result: _mean_detection(result, lambda item: item.detection_rate),
            lambda result: result.realized_false_alert_rate,
        ),
        mean_ensemble=_arm_rates(results, mean_ensemble),
        local=_rates(
            results,
            lambda result: _mean_detection(result, lambda item: item.local_detection_rate),
            lambda result: result.local_false_alert_rate,
        ),
        peer_union=_rates(
            results,
            lambda result: _mean_detection(result, lambda item: item.peer_union_detection_rate),
            lambda result: result.peer_union_false_alert_rate,
        ),
        max_fusion=_rates(
            results,
            lambda result: _mean_detection(result, lambda item: item.max_fusion_detection_rate),
            lambda result: result.max_fusion_false_alert_rate,
        ),
        bonferroni=_rates(
            results,
            lambda result: _mean_detection(result, lambda item: item.bonferroni_detection_rate),
            lambda result: result.bonferroni_false_alert_rate,
        ),
        arms=tuple(
            ArmRates(performance.arm, _arm_rates(results, performance.arm))
            for performance in first.arms
        ),
        families=_families(results),
        sweep=_sweep(results),
        seed_gains=tuple(
            SeedRateDifference(
                result.seed,
                _mean_detection(result, lambda item: item.detection_rate)
                - _arm_detection(result, mean_ensemble),
            )
            for result in results
        ),
    )


def records_of(
    results: tuple[TargetPerformance, ...], alpha: AlphaLevel, rows: LocalTrainingRowCount
) -> tuple[TargetRecord, ...]:
    grouped: dict[DatasetClientId, list[TargetPerformance]] = defaultdict(list)
    for result in results:
        grouped[result.target].append(result)
    return tuple(
        target_record(sorted(grouped[target], key=lambda item: item.seed), alpha, rows)
        for target in sorted(grouped, key=lambda item: item.name)
    )


@dataclass(frozen=True, slots=True)
class RecordStore:
    records: tuple[TargetRecord, ...]

    def select(
        self,
        alpha: AlphaLevel,
        rows: LocalTrainingRowCount,
        cohort: frozenset[DatasetClientId] | None = None,
    ) -> tuple[TargetRecord, ...]:
        return tuple(
            record
            for record in self.records
            if record.alpha is alpha
            and record.rows == rows
            and (cohort is None or record.target in cohort)
        )

    def seeds(self) -> tuple[LearningSeed, ...]:
        return tuple(sorted({item.seed for record in self.records for item in record.seed_gains}))


def pooled_difference(
    records: tuple[TargetRecord, ...],
    value: Callable[[TargetRecord], RateDifference],
    settings: ReportingSettings,
) -> tuple[RateDifference, RateConfidenceInterval]:
    differences = tuple(TargetPairedDifference(record.target, value(record)) for record in records)
    return _mean(item.difference for item in differences), stratified_paired_interval(
        differences, settings
    )


@dataclass(frozen=True, slots=True)
class VariantMetrics:
    variant: MatrixVariant
    metrics: CampaignMetricSummary


@dataclass(frozen=True, slots=True)
class StratumBudgetEvidence:
    stratum: StudyStratum
    analyses: tuple[BudgetAnalysis, ...]
    eligible_targets: ObservationCount
    variants: tuple[VariantMetrics, ...]


@dataclass(frozen=True, slots=True)
class _Cohort:
    stratum: StudyStratum
    targets: tuple[ClientName, ...]
    budgets: tuple[LocalTrainingRowCount, ...]


def _outcome_value(summary: TargetMetricSummary, outcome: BudgetOutcome) -> RateDifference:
    match outcome:
        case BudgetOutcome.PACE_TPR:
            return summary.method_detection_rate
        case BudgetOutcome.LOCAL_TPR:
            return summary.local_detection_rate
        case BudgetOutcome.MAX_FUSION_TPR:
            return summary.max_fusion_detection_rate
        case BudgetOutcome.PACE_FALSE_ALERT_RATE:
            return summary.realized_false_alert_rate


def _cohort_metrics(
    evidence: StratumBudgetEvidence,
    alpha: AlphaLevel,
    rows: LocalTrainingRowCount,
    targets: tuple[ClientName, ...],
) -> dict[DatasetClientId, TargetMetricSummary]:
    for entry in evidence.variants:
        variant = entry.variant
        if (variant.alpha, variant.local_training_rows, variant.targets) == (alpha, rows, targets):
            return {summary.target: summary for summary in entry.metrics.targets}
    raise DatasetValidationError(
        f"No campaign variant covers {evidence.stratum.name} alpha={alpha.name} "
        f"rows={rows} for the frozen cohort"
    )


def _stratum_cohort_differences(
    evidence: StratumBudgetEvidence,
    cohort: _Cohort,
    alpha: AlphaLevel,
    pair: tuple[LocalTrainingRowCount, LocalTrainingRowCount],
) -> dict[BudgetOutcome, tuple[TargetPairedDifference, ...]]:
    lower = _cohort_metrics(evidence, alpha, pair[0], cohort.targets)
    upper = _cohort_metrics(evidence, alpha, pair[1], cohort.targets)
    if set(lower) != set(upper):
        raise DatasetValidationError("Paired budget cells must contain the same targets")
    return {
        outcome: tuple(
            TargetPairedDifference(
                target,
                _outcome_value(upper[target], outcome) - _outcome_value(lower[target], outcome),
            )
            for target in sorted(lower, key=lambda item: item.name)
        )
        for outcome in BudgetOutcome
    }


def _mean_difference(differences: tuple[TargetPairedDifference, ...]) -> RateDifference:
    return sum(item.difference for item in differences) / len(differences)


def _comparison(
    evidence: tuple[StratumBudgetEvidence, ...],
    cohorts: tuple[_Cohort, ...],
    alpha: AlphaLevel,
    pair: tuple[LocalTrainingRowCount, LocalTrainingRowCount],
    reporting: ReportingSettings,
) -> BudgetComparison:
    by_stratum = {item.stratum: item for item in evidence}
    per_stratum = tuple(
        (
            cohort.stratum,
            _stratum_cohort_differences(by_stratum[cohort.stratum], cohort, alpha, pair),
        )
        for cohort in cohorts
    )
    pooled_by_outcome: dict[BudgetOutcome, list[TargetPairedDifference]] = defaultdict(list)
    for _, differences in per_stratum:
        for outcome, items in differences.items():
            pooled_by_outcome[outcome].extend(items)
    return BudgetComparison(
        alpha=alpha,
        from_rows=pair[0],
        to_rows=pair[1],
        target_count=len(pooled_by_outcome[BudgetOutcome.PACE_TPR]),
        pooled=tuple(
            BudgetOutcomeDifference(
                outcome,
                _mean_difference(tuple(pooled_by_outcome[outcome])),
                stratified_paired_interval(tuple(pooled_by_outcome[outcome]), reporting),
            )
            for outcome in BudgetOutcome
        ),
        strata=tuple(
            StratumBudgetEffect(
                stratum,
                len(differences[BudgetOutcome.PACE_TPR]),
                tuple(
                    BudgetOutcomeDifference(outcome, _mean_difference(differences[outcome]), None)
                    for outcome in BudgetOutcome
                ),
            )
            for stratum, differences in per_stratum
        ),
    )


def _budget_pairs(
    budgets: tuple[LocalTrainingRowCount, ...],
) -> tuple[tuple[LocalTrainingRowCount, LocalTrainingRowCount], ...]:
    consecutive = tuple(zip(budgets, budgets[1:], strict=False))
    if len(budgets) > 2:
        return (*consecutive, (budgets[0], budgets[-1]))
    return consecutive


def _analysis(
    kind: BudgetAnalysisKind,
    scope: BudgetAnalysisScope,
    evidence: tuple[StratumBudgetEvidence, ...],
    cohorts: tuple[_Cohort, ...],
    reporting: ReportingSettings,
) -> BudgetEffectAnalysis | None:
    if not cohorts:
        return None
    budgets = cohorts[0].budgets
    if any(cohort.budgets != budgets for cohort in cohorts):
        raise DatasetValidationError("Pooled analyses need identical budgets in every stratum")
    pairs = _budget_pairs(budgets)
    alphas = sorted(
        {entry.variant.alpha for item in evidence for entry in item.variants},
        key=lambda alpha: alpha.fraction,
    )
    return BudgetEffectAnalysis(
        kind,
        scope,
        tuple(cohort.stratum for cohort in cohorts),
        budgets,
        sum(len(cohort.targets) for cohort in cohorts),
        tuple(
            _comparison(evidence, cohorts, alpha, pair, reporting)
            for alpha in alphas
            for pair in pairs
        ),
    )


def _cohorts_of(
    evidence: tuple[StratumBudgetEvidence, ...], kind: BudgetAnalysisKind
) -> tuple[_Cohort, ...]:
    return tuple(
        _Cohort(item.stratum, analysis.targets, analysis.budgets)
        for item in evidence
        for analysis in item.analyses
        if analysis.kind is kind
    )


def _feasibility(
    evidence: tuple[StratumBudgetEvidence, ...], kinds: tuple[BudgetAnalysisKind, ...]
) -> tuple[StratumFeasibility, ...]:
    return tuple(
        StratumFeasibility(
            item.stratum,
            item.eligible_targets,
            tuple(
                sum(len(analysis.targets) for analysis in item.analyses if analysis.kind is kind)
                for kind in kinds
            ),
        )
        for item in evidence
    )


def _complete_strata(
    evidence: tuple[StratumBudgetEvidence, ...],
) -> tuple[StratumBudgetEvidence, ...]:
    return tuple(
        item
        for item in evidence
        if any(
            analysis.kind is BudgetAnalysisKind.HIGH_RESOURCE
            and len(analysis.targets) == item.eligible_targets
            for analysis in item.analyses
        )
    )


def _complete_stratum_analyses(
    evidence: tuple[StratumBudgetEvidence, ...], reporting: ReportingSettings
) -> tuple[BudgetEffectAnalysis | None, ...]:
    return tuple(
        _analysis(
            BudgetAnalysisKind.HIGH_RESOURCE,
            BudgetAnalysisScope.COMPLETE_STRATUM,
            (item,),
            _cohorts_of((item,), BudgetAnalysisKind.HIGH_RESOURCE),
            reporting,
        )
        for item in _complete_strata(evidence)
    )


def summarize_budget_effects(
    evidence: tuple[StratumBudgetEvidence, ...],
    kinds: tuple[BudgetAnalysisKind, ...],
    reporting: ReportingSettings,
) -> BudgetEffectReport:
    if not evidence:
        raise DatasetValidationError("Budget effects require at least one study stratum")
    pooled = tuple(
        _analysis(
            kind, BudgetAnalysisScope.POOLED, evidence, _cohorts_of(evidence, kind), reporting
        )
        for kind in kinds
    )
    return BudgetEffectReport(
        kinds,
        _feasibility(evidence, kinds),
        tuple(
            analysis
            for analysis in (*pooled, *_complete_stratum_analyses(evidence, reporting))
            if analysis is not None
        ),
    )


@dataclass(frozen=True, slots=True)
class ResultTable:
    name: TableName
    columns: tuple[ColumnHeading, ...]
    rows: tuple[tuple[DisplayText, ...], ...]


@dataclass(frozen=True, slots=True)
class StratumTraining:
    stratum: StudyStratum
    targets: tuple[TargetTrainingRows, ...]


@dataclass(frozen=True, slots=True)
class ReportInputs:
    configuration: ResolvedConfiguration
    store: RecordStore
    acceptance: AcceptanceReport
    evidence: tuple[StratumBudgetEvidence, ...]
    training: tuple[StratumTraining, ...]

    def cohort(self, kind: BudgetAnalysisKind) -> frozenset[DatasetClientId]:
        return frozenset(
            DatasetClientId(item.stratum.dataset, name)
            for item in self.evidence
            for analysis in item.analyses
            if analysis.kind is kind
            for name in analysis.targets
        )

    def budgets(self, kind: BudgetAnalysisKind) -> tuple[LocalTrainingRowCount, ...]:
        budgets = {
            budget
            for item in self.evidence
            for analysis in item.analyses
            if analysis.kind is kind
            for budget in analysis.budgets
        }
        return tuple(sorted(budgets))

    def primary(self, alpha: AlphaLevel, rows: LocalTrainingRowCount) -> tuple[TargetRecord, ...]:
        return self.store.select(alpha, rows, self.cohort(BudgetAnalysisKind.UNIVERSAL_PRIMARY))

    def alphas(self) -> tuple[AlphaLevel, ...]:
        observed = {record.alpha for record in self.store.records}
        configured = self.configuration.protocol_matrix.alpha_levels
        return tuple(sorted((alpha for alpha in configured if alpha in observed), key=_alpha_order))


def _alpha_order(alpha: AlphaLevel) -> RateDifference:
    return alpha.fraction


def number(value: RateDifference) -> DisplayText:
    return DisplayText(f"{value:.6f}")


def count(value: LocalTrainingRowCount) -> DisplayText:
    return DisplayText(f"{value}")


def alpha_text(alpha: AlphaLevel) -> DisplayText:
    return DisplayText(f"{alpha.fraction}")


def optional_number(value: RateDifference | None) -> DisplayText:
    return DisplayText("") if value is None else number(value)


def interval_cells(
    value: RateDifference | None, interval: RateConfidenceInterval | None
) -> tuple[DisplayText, DisplayText, DisplayText]:
    if value is None or interval is None:
        return DisplayText(""), DisplayText(""), DisplayText("")
    return number(value), number(interval.lower_bound), number(interval.upper_bound)


def analysis_text(kind: BudgetAnalysisKind) -> DisplayText:
    return DisplayText(kind.name.lower())


def verdict_text(satisfied: CriterionSatisfied) -> DisplayText:
    answer = SatisfiedAnswer.YES if satisfied else SatisfiedAnswer.NO
    return DisplayText(answer)


def mean_of(values: tuple[RateDifference, ...]) -> RateDifference:
    return np.mean(values).item()


def method_false_alert(method: ReportedMethod, record: TargetRecord) -> FalseAlertRate:
    match method:
        case ReportedMethod.PACE:
            return record.pace.false_alert
        case ReportedMethod.MEAN_ENSEMBLE:
            return record.mean_ensemble.false_alert
        case ReportedMethod.LOCAL_BRANCH:
            return record.local.false_alert
        case ReportedMethod.PEER_UNION:
            return record.peer_union.false_alert
        case ReportedMethod.MAX_FUSION:
            return record.max_fusion.false_alert
        case ReportedMethod.BONFERRONI:
            return record.bonferroni.false_alert


def sensitivity_rows(
    members: tuple[TargetRecord, ...],
    group: Callable[[TargetRecord], DisplayText],
) -> tuple[tuple[DisplayText, RateDifference], ...]:
    names = sorted({group(item) for item in members})
    remaining = {
        name: tuple(item.gain for item in members if group(item) != name) for name in names
    }
    return tuple((name, mean_of(gains)) for name, gains in remaining.items() if gains)


def _family_mean(
    record: TargetRecord, value: Callable[[TargetRecord, ObservationCount], RateDifference]
) -> RateDifference:
    return mean_of(tuple(value(record, index) for index in range(len(record.families))))


@dataclass(frozen=True, slots=True)
class DetectionShares:
    peers_only: RateDifference
    local_only: RateDifference
    both: RateDifference
    neither: RateDifference

    def share(self, part: DetectionContribution) -> RateDifference:
        match part:
            case DetectionContribution.PEERS_ONLY:
                return self.peers_only
            case DetectionContribution.LOCAL_ONLY:
                return self.local_only
            case DetectionContribution.BOTH:
                return self.both
            case DetectionContribution.NEITHER:
                return self.neither


def decomposition_of(record: TargetRecord) -> DetectionShares:
    pace = _family_mean(record, lambda item, index: item.families[index].pace)
    local = _family_mean(record, lambda item, index: item.families[index].local)
    peers = _family_mean(record, lambda item, index: item.families[index].peer_union)
    return DetectionShares(
        peers_only=pace - local,
        local_only=pace - peers,
        both=peers + local - pace,
        neither=1.0 - pace,
    )
