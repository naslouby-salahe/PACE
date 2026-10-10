from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from functools import lru_cache
from math import sqrt
from statistics import NormalDist

import numpy as np

from pace.config import PowerSettings, ProtocolMatrixSettings, ReportingSettings
from pace.types import (
    AlphaLevel,
    AnalysisPower,
    ArmMetricSummary,
    ArmTargetSummary,
    AttackType,
    AttackTypePerformance,
    BootstrapReplicateCount,
    BootstrapSeed,
    BudgetSweepPerformance,
    CampaignMetricSummary,
    DatasetClientId,
    DatasetName,
    DetectionRate,
    EvaluationRule,
    FalseAlertRate,
    FloatArray,
    ObservationCount,
    PairedDifferenceSd,
    PowerPoint,
    RateConfidenceInterval,
    RateDifference,
    RowIndexArray,
    SeedFalseAlertRates,
    SeedRateDifference,
    StatisticalPower,
    StratumBudgetPlan,
    StudyArm,
    SummaryQuantile,
    TargetMetricSummary,
    TargetPairedDifference,
    TargetPerformance,
)


@lru_cache(maxsize=64)
def _bootstrap_selections(
    seed: BootstrapSeed, replicates: BootstrapReplicateCount, sizes: tuple[ObservationCount, ...]
) -> tuple[RowIndexArray, ...]:
    generator = np.random.default_rng(seed)
    selections = tuple(generator.integers(0, size, size=(replicates, size)) for size in sizes)
    for selection in selections:
        selection.setflags(write=False)
    return selections


def stratified_paired_interval(
    target_differences: tuple[TargetPairedDifference, ...],
    settings: ReportingSettings,
) -> RateConfidenceInterval:
    if not target_differences:
        raise ValueError("A paired interval requires target-level differences")
    target_ids = tuple(item.target for item in target_differences)
    if len(set(target_ids)) != len(target_ids):
        raise ValueError("Paired differences must contain each target exactly once")
    strata = [
        sorted(
            (item.target.name, item.difference)
            for item in target_differences
            if item.target.dataset is dataset
        )
        for dataset in DatasetName
    ]
    strata = [stratum for stratum in strata if stratum]
    selections = _bootstrap_selections(
        settings.bootstrap_seed,
        settings.bootstrap_replicates,
        tuple(len(stratum) for stratum in strata),
    )
    bootstrap_estimates = np.zeros(settings.bootstrap_replicates, dtype=np.float64)
    for stratum, selection in zip(strata, selections, strict=True):
        values = np.asarray([difference for _, difference in stratum], dtype=np.float64)
        bootstrap_estimates += np.mean(values[selection], axis=1) * (
            values.size / len(target_differences)
        )
    tail_probability = (1.0 - settings.confidence_level) / 2.0
    lower_bound, upper_bound = np.quantile(
        bootstrap_estimates,
        (tail_probability, 1.0 - tail_probability),
        method="linear",
    ).tolist()
    return RateConfidenceInterval(
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        confidence_level=settings.confidence_level,
    )


def _fractions(rates: Iterable[RateDifference]) -> FloatArray:
    return np.asarray(tuple(rates), dtype=np.float64)


def matched_fpr_detection_rate(
    sweep: tuple[BudgetSweepPerformance, ...], reference_rate: FalseAlertRate
) -> DetectionRate | None:
    if len(sweep) < 2:
        raise ValueError("Matched-FPR interpolation requires at least two sweep points")
    multipliers = tuple(point.multiplier for point in sweep)
    if tuple(sorted(set(multipliers))) != multipliers:
        raise ValueError("Matched-FPR sweep multipliers must be unique and increasing")
    points = np.asarray(
        sorted((point.false_alert_rate, point.detection_rate) for point in sweep),
        dtype=np.float64,
    )
    false_alert_rates, inverse = np.unique(points[:, 0], return_inverse=True)
    if false_alert_rates.size < 2:
        return None
    detection_rates = np.zeros(false_alert_rates.size, dtype=np.float64)
    np.maximum.at(detection_rates, inverse, points[:, 1])
    envelope = np.maximum.accumulate(detection_rates)
    reference = reference_rate
    if reference < false_alert_rates[0] or reference > false_alert_rates[-1]:
        return None
    return np.interp(reference, false_alert_rates, envelope).item()


def _matched_target_difference(
    repetitions: list[TargetPerformance],
    max_fusion_tpr: DetectionRate,
    max_fusion_fpr: FalseAlertRate,
) -> RateDifference | None:
    sweep_multipliers = tuple(point.multiplier for point in repetitions[0].budget_sweep)
    if any(
        tuple(point.multiplier for point in result.budget_sweep) != sweep_multipliers
        for result in repetitions[1:]
    ):
        raise ValueError("Budget sweep coverage must match across target seeds")
    averaged_sweep = tuple(
        BudgetSweepPerformance(
            multiplier=multiplier,
            detection_rate=np.mean(
                _fractions(result.budget_sweep[index].detection_rate for result in repetitions)
            ).item(),
            false_alert_rate=np.mean(
                _fractions(result.budget_sweep[index].false_alert_rate for result in repetitions)
            ).item(),
        )
        for index, multiplier in enumerate(sweep_multipliers)
    )
    matched_rate = matched_fpr_detection_rate(averaged_sweep, max_fusion_fpr)
    if matched_rate is None:
        return None
    return matched_rate - max_fusion_tpr


def _target_detection_rate(
    repetitions: list[TargetPerformance],
    attack_rate: Callable[[AttackTypePerformance], DetectionRate],
) -> DetectionRate:
    return np.mean(
        tuple(
            np.mean(_fractions(attack_rate(item) for item in result.attack_types))
            for result in repetitions
        )
    ).item()


def _attack_type_detection_rate(
    repetitions: list[TargetPerformance],
    attack_type: AttackType,
    attack_rate: Callable[[AttackTypePerformance], DetectionRate],
) -> DetectionRate:
    return np.mean(
        _fractions(
            attack_rate(
                next(item for item in result.attack_types if item.attack_type == attack_type)
            )
            for result in repetitions
        )
    ).item()


def _target_false_alert_rate(
    repetitions: list[TargetPerformance],
    false_alert_rate: Callable[[TargetPerformance], FalseAlertRate],
) -> FalseAlertRate:
    return np.mean(_fractions(false_alert_rate(item) for item in repetitions)).item()


def _arm_attack_rates(repetitions: list[TargetPerformance], arm: StudyArm) -> FloatArray | None:
    per_seed: list[FloatArray] = []
    for result in repetitions:
        performance = next((item for item in result.arms if item.arm is arm), None)
        if performance is None:
            return None
        per_seed.append(_fractions(performance.detection_rates))
    return np.asarray(np.mean(np.stack(per_seed), axis=0), dtype=np.float64)


def _local_attack_rates(repetitions: list[TargetPerformance]) -> FloatArray:
    return np.asarray(
        np.mean(
            np.stack(
                tuple(
                    _fractions(item.local_detection_rate for item in result.attack_types)
                    for result in repetitions
                )
            ),
            axis=0,
        ),
        dtype=np.float64,
    )


@dataclass(frozen=True, slots=True)
class _ArmTargetRecord:
    arm_rate: DetectionRate
    local_difference: RateDifference
    method_difference: TargetPairedDifference
    false_alert_rate: FalseAlertRate
    catastrophic_cells: ObservationCount
    summary: ArmTargetSummary


def _clamped_matched_detection_rate(
    sweep: tuple[BudgetSweepPerformance, ...], reference_rate: FalseAlertRate
) -> DetectionRate:
    ordered = sorted(sweep, key=lambda point: point.false_alert_rate)
    false_alert_rates = _fractions(point.false_alert_rate for point in ordered)
    detection_rates = _fractions(point.detection_rate for point in ordered)
    if reference_rate <= false_alert_rates[0]:
        return detection_rates[0].item()
    if reference_rate >= false_alert_rates[-1]:
        return detection_rates[-1].item()
    return np.interp(reference_rate, false_alert_rates, detection_rates).item()


def _arm_detection_rate(result: TargetPerformance, arm: StudyArm) -> DetectionRate:
    performance = next(item for item in result.arms if item.arm is arm)
    return np.mean(_fractions(performance.detection_rates)).item()


def _seed_matched_difference(repetitions: list[TargetPerformance], arm: StudyArm) -> RateDifference:
    return np.mean(
        _fractions(
            _clamped_matched_detection_rate(
                result.budget_sweep,
                next(item for item in result.arms if item.arm is arm).false_alert_rate,
            )
            - _arm_detection_rate(result, arm)
            for result in repetitions
        )
    ).item()


def _seed_false_alert_rates(
    repetitions: list[TargetPerformance], arm: StudyArm
) -> tuple[SeedFalseAlertRates, ...]:
    return tuple(
        SeedFalseAlertRates(
            result.seed,
            result.realized_false_alert_rate,
            next(item for item in result.arms if item.arm is arm).false_alert_rate,
        )
        for result in repetitions
    )


def _seed_differences(
    repetitions: list[TargetPerformance], arm: StudyArm
) -> tuple[SeedRateDifference, ...]:
    return tuple(
        SeedRateDifference(
            result.seed,
            np.mean(_fractions(item.detection_rate for item in result.attack_types)).item()
            - np.mean(
                _fractions(next(item for item in result.arms if item.arm is arm).detection_rates)
            ).item(),
        )
        for result in repetitions
    )


def _arm_target_record(
    arm: StudyArm,
    target: DatasetClientId,
    repetitions: list[TargetPerformance],
    summary: TargetMetricSummary,
) -> _ArmTargetRecord | None:
    attack_rates = _arm_attack_rates(repetitions, arm)
    if attack_rates is None:
        return None
    arm_rate = np.mean(attack_rates).item()
    false_alert_rate = _target_false_alert_rate(
        repetitions,
        lambda result: next(item for item in result.arms if item.arm is arm).false_alert_rate,
    )
    return _ArmTargetRecord(
        arm_rate,
        arm_rate - summary.local_detection_rate,
        TargetPairedDifference(target, (summary.method_detection_rate - arm_rate)),
        false_alert_rate,
        np.sum(
            attack_rates - _local_attack_rates(repetitions)
            < EvaluationRule.CATASTROPHIC_DROP.threshold
        ).item(),
        ArmTargetSummary(
            target=target,
            nominal_difference=summary.method_detection_rate - arm_rate,
            matched_difference=_seed_matched_difference(repetitions, arm),
            method_false_alert_rate=summary.realized_false_alert_rate,
            arm_false_alert_rate=false_alert_rate,
            seed_differences=_seed_differences(repetitions, arm),
            seed_false_alert_rates=_seed_false_alert_rates(repetitions, arm),
        ),
    )


def _summarize_arm(
    arm: StudyArm,
    by_target: dict[DatasetClientId, list[TargetPerformance]],
    target_summaries: dict[DatasetClientId, TargetMetricSummary],
    reporting: ReportingSettings,
) -> ArmMetricSummary | None:
    records = tuple(
        record
        for target, repetitions in by_target.items()
        if (record := _arm_target_record(arm, target, repetitions, target_summaries[target]))
        is not None
    )
    if not records:
        return None
    method_differences = tuple(record.method_difference for record in records)
    local_differences = _fractions(record.local_difference for record in records)
    false_alert_rates = _fractions(record.false_alert_rate for record in records)
    return ArmMetricSummary(
        arm=arm,
        target_count=len(records),
        macro_detection_rate=np.mean(_fractions(record.arm_rate for record in records)).item(),
        macro_detection_difference=np.mean(local_differences).item(),
        method_arm_difference=np.mean(
            _fractions(item.difference for item in method_differences)
        ).item(),
        method_arm_confidence_interval=stratified_paired_interval(method_differences, reporting),
        worst_detection_difference=np.min(local_differences).item(),
        catastrophic_cell_count=sum(record.catastrophic_cells for record in records),
        harmed_target_count=np.sum(local_differences < EvaluationRule.HARMED_DROP.threshold).item(),
        mean_false_alert_rate=np.mean(false_alert_rates).item(),
        worst_false_alert_rate=np.max(false_alert_rates).item(),
        targets=tuple(record.summary for record in records),
    )


def _target_summary(
    target: DatasetClientId, repetitions: list[TargetPerformance]
) -> TargetMetricSummary:
    method_tpr = _target_detection_rate(repetitions, lambda item: item.detection_rate)
    local_tpr = _target_detection_rate(repetitions, lambda item: item.local_detection_rate)
    max_fusion_tpr = _target_detection_rate(
        repetitions, lambda item: item.max_fusion_detection_rate
    )
    max_fusion_false_alert_rate = _target_false_alert_rate(
        repetitions, lambda result: result.max_fusion_false_alert_rate
    )
    return TargetMetricSummary(
        target=target,
        method_detection_rate=method_tpr,
        local_detection_rate=local_tpr,
        max_fusion_detection_rate=max_fusion_tpr,
        bonferroni_detection_rate=_target_detection_rate(
            repetitions, lambda item: item.bonferroni_detection_rate
        ),
        detection_difference=method_tpr - local_tpr,
        max_fusion_detection_difference=max_fusion_tpr - local_tpr,
        method_max_fusion_difference=method_tpr - max_fusion_tpr,
        realized_false_alert_rate=_target_false_alert_rate(
            repetitions, lambda result: result.realized_false_alert_rate
        ),
        local_false_alert_rate=_target_false_alert_rate(
            repetitions, lambda result: result.local_false_alert_rate
        ),
        max_fusion_false_alert_rate=max_fusion_false_alert_rate,
        bonferroni_false_alert_rate=_target_false_alert_rate(
            repetitions, lambda result: result.bonferroni_false_alert_rate
        ),
        matched_fpr_difference=_matched_target_difference(
            repetitions, max_fusion_tpr, max_fusion_false_alert_rate
        ),
    )


def _group_by_target(
    results: tuple[TargetPerformance, ...],
) -> dict[DatasetClientId, list[TargetPerformance]]:
    if not results:
        raise ValueError("Campaign reporting requires target results")
    by_target: dict[DatasetClientId, list[TargetPerformance]] = defaultdict(list)
    result_keys = tuple((result.target, result.seed) for result in results)
    if len(set(result_keys)) != len(result_keys):
        raise ValueError("Campaign reporting cannot include repeated target seeds")
    for result in results:
        by_target[result.target].append(result)
    campaign_schedules = {
        tuple(point.multiplier for point in result.budget_sweep) for result in results
    }
    if len(campaign_schedules) != 1:
        raise ValueError("Campaign budget sweep coverage must match across targets and seeds")
    return by_target


def _summaries_and_catastrophic_cells(
    by_target: dict[DatasetClientId, list[TargetPerformance]],
) -> tuple[list[TargetMetricSummary], ObservationCount]:
    target_summaries: list[TargetMetricSummary] = []
    catastrophic_cells = 0
    for target, repetitions in by_target.items():
        repetitions.sort(key=lambda result: result.seed)
        attack_types = tuple(item.attack_type for item in repetitions[0].attack_types)
        if any(
            tuple(item.attack_type for item in result.attack_types) != attack_types
            for result in repetitions
        ):
            raise ValueError("Attack type coverage must match across target seeds")
        target_summaries.append(_target_summary(target, repetitions))
        for attack_type in attack_types:
            method_type_rate = _attack_type_detection_rate(
                repetitions, attack_type, lambda item: item.detection_rate
            )
            local_type_rate = _attack_type_detection_rate(
                repetitions, attack_type, lambda item: item.local_detection_rate
            )
            catastrophic_cells += (
                method_type_rate - local_type_rate < EvaluationRule.CATASTROPHIC_DROP.threshold
            )
    seed_sets = tuple(
        frozenset(result.seed for result in repetitions) for repetitions in by_target.values()
    )
    if any(seeds != seed_sets[0] for seeds in seed_sets[1:]):
        raise ValueError("Eligible target coverage must match across seeds")
    return target_summaries, catastrophic_cells


@dataclass(frozen=True, slots=True)
class _TargetColumns:
    method_rates: FloatArray
    local_rates: FloatArray
    max_fusion_rates: FloatArray
    bonferroni_rates: FloatArray
    differences: FloatArray
    false_alert_rates: FloatArray
    local_false_alert_rates: FloatArray
    max_fusion_false_alert_rates: FloatArray
    bonferroni_false_alert_rates: FloatArray
    max_fusion_differences: FloatArray
    method_max_fusion_differences: FloatArray

    @classmethod
    def of(cls, summaries: list[TargetMetricSummary]) -> "_TargetColumns":
        def column(field: Callable[[TargetMetricSummary], RateDifference]) -> FloatArray:
            return _fractions(field(summary) for summary in summaries)

        return cls(
            method_rates=column(lambda summary: summary.method_detection_rate),
            local_rates=column(lambda summary: summary.local_detection_rate),
            max_fusion_rates=column(lambda summary: summary.max_fusion_detection_rate),
            bonferroni_rates=column(lambda summary: summary.bonferroni_detection_rate),
            differences=column(lambda summary: summary.detection_difference),
            false_alert_rates=column(lambda summary: summary.realized_false_alert_rate),
            local_false_alert_rates=column(lambda summary: summary.local_false_alert_rate),
            max_fusion_false_alert_rates=column(
                lambda summary: summary.max_fusion_false_alert_rate
            ),
            bonferroni_false_alert_rates=column(
                lambda summary: summary.bonferroni_false_alert_rate
            ),
            max_fusion_differences=column(lambda summary: summary.max_fusion_detection_difference),
            method_max_fusion_differences=column(
                lambda summary: summary.method_max_fusion_difference
            ),
        )


def _arm_summaries(
    results: tuple[TargetPerformance, ...],
    by_target: dict[DatasetClientId, list[TargetPerformance]],
    target_summaries: list[TargetMetricSummary],
    reporting: ReportingSettings,
) -> tuple[ArmMetricSummary, ...]:
    summaries_by_target = {summary.target: summary for summary in target_summaries}
    return tuple(
        summary
        for arm in StudyArm
        if any(item.arm is arm for result in results for item in result.arms)
        if (summary := _summarize_arm(arm, by_target, summaries_by_target, reporting)) is not None
    )


def _paired_differences(
    summaries: list[TargetMetricSummary], pick: Callable[[TargetMetricSummary], RateDifference]
) -> tuple[TargetPairedDifference, ...]:
    return tuple(TargetPairedDifference(summary.target, pick(summary)) for summary in summaries)


@dataclass(frozen=True, slots=True)
class _MatchedFalseAlert:
    target_count: ObservationCount
    macro_difference: RateDifference | None
    interval: RateConfidenceInterval | None

    @classmethod
    def of(
        cls, summaries: Sequence[TargetMetricSummary], reporting: ReportingSettings
    ) -> "_MatchedFalseAlert":
        targets = tuple(
            TargetPairedDifference(summary.target, summary.matched_fpr_difference)
            for summary in summaries
            if summary.matched_fpr_difference is not None
        )
        if not targets:
            return cls(0, None, None)
        return cls(
            len(targets),
            np.mean(tuple(item.difference for item in targets)).item(),
            stratified_paired_interval(targets, reporting),
        )


def summarize_campaign(
    results: tuple[TargetPerformance, ...],
    alpha: AlphaLevel,
    reporting: ReportingSettings,
) -> CampaignMetricSummary:
    by_target = _group_by_target(results)
    target_summaries, catastrophic_cells = _summaries_and_catastrophic_cells(by_target)
    columns = _TargetColumns.of(target_summaries)
    matched = _MatchedFalseAlert.of(target_summaries, reporting)
    return CampaignMetricSummary(
        target_count=len(target_summaries),
        targets=tuple(target_summaries),
        macro_detection_rate=np.mean(columns.method_rates).item(),
        local_macro_detection_rate=np.mean(columns.local_rates).item(),
        max_fusion_macro_detection_rate=np.mean(columns.max_fusion_rates).item(),
        bonferroni_macro_detection_rate=np.mean(columns.bonferroni_rates).item(),
        macro_detection_difference=np.mean(columns.differences).item(),
        method_local_confidence_interval=stratified_paired_interval(
            _paired_differences(target_summaries, lambda summary: summary.detection_difference),
            reporting,
        ),
        max_fusion_macro_detection_difference=np.mean(columns.max_fusion_differences).item(),
        method_max_fusion_confidence_interval=stratified_paired_interval(
            _paired_differences(
                target_summaries, lambda summary: summary.method_max_fusion_difference
            ),
            reporting,
        ),
        matched_fpr_target_count=matched.target_count,
        matched_fpr_macro_difference=matched.macro_difference,
        matched_fpr_confidence_interval=matched.interval,
        mean_method_max_fusion_difference=np.mean(columns.method_max_fusion_differences).item(),
        worst_method_max_fusion_difference=np.min(columns.method_max_fusion_differences).item(),
        improved_target_share=np.mean(columns.method_max_fusion_differences > 0.0).item(),
        worst_detection_difference=np.min(columns.differences).item(),
        tenth_percentile_detection_difference=np.quantile(
            columns.differences,
            SummaryQuantile.LOWER_DECILE.level,
            method="linear",
        ).item(),
        catastrophic_cell_count=catastrophic_cells,
        harmed_target_count=sum(
            1
            for difference in columns.differences
            if difference < EvaluationRule.HARMED_DROP.threshold
        ),
        mean_false_alert_rate=np.mean(columns.false_alert_rates).item(),
        median_false_alert_rate=np.median(columns.false_alert_rates).item(),
        ninetieth_percentile_false_alert_rate=np.quantile(
            columns.false_alert_rates,
            SummaryQuantile.UPPER_DECILE.level,
            method="linear",
        ).item(),
        worst_false_alert_rate=np.max(columns.false_alert_rates).item(),
        exceedance_share=np.mean(
            columns.false_alert_rates
            > EvaluationRule.FALSE_ALERT_EXCEEDANCE.threshold * alpha.fraction
        ).item(),
        local_mean_false_alert_rate=np.mean(columns.local_false_alert_rates).item(),
        max_fusion_mean_false_alert_rate=np.mean(columns.max_fusion_false_alert_rates).item(),
        max_fusion_worst_false_alert_rate=np.max(columns.max_fusion_false_alert_rates).item(),
        bonferroni_mean_false_alert_rate=np.mean(columns.bonferroni_false_alert_rates).item(),
        bonferroni_worst_false_alert_rate=np.max(columns.bonferroni_false_alert_rates).item(),
        arms=_arm_summaries(results, by_target, target_summaries, reporting),
    )


def paired_minimum_detectable_effect(
    target_count: ObservationCount, difference_sd: PairedDifferenceSd, settings: PowerSettings
) -> RateDifference:
    normal = NormalDist()
    critical = normal.inv_cdf(1.0 - settings.significance_level / 2.0) + normal.inv_cdf(
        settings.power
    )
    return critical * difference_sd / sqrt(target_count)


def paired_power_at_effect(
    target_count: ObservationCount, difference_sd: PairedDifferenceSd, settings: PowerSettings
) -> StatisticalPower:
    normal = NormalDist()
    critical = normal.inv_cdf(1.0 - settings.significance_level / 2.0)
    shift = settings.minimum_meaningful_effect * sqrt(target_count) / difference_sd
    return min(normal.cdf(shift - critical) + normal.cdf(-shift - critical), 1.0 - 1e-12)


def summarize_budget_power(
    plans: tuple[StratumBudgetPlan, ...], protocol: ProtocolMatrixSettings
) -> tuple[AnalysisPower, ...]:
    return tuple(
        AnalysisPower(
            spec.kind,
            count,
            tuple(
                PowerPoint(
                    sd,
                    paired_minimum_detectable_effect(count, sd, protocol.power),
                    paired_power_at_effect(count, sd, protocol.power),
                )
                for sd in protocol.power.difference_sds
            ),
        )
        for spec in protocol.analyses
        if (
            count := sum(
                len(analysis.targets)
                for plan in plans
                for analysis in plan.analyses
                if analysis.kind is spec.kind
            )
        )
    )
