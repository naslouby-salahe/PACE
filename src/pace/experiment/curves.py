from collections.abc import Callable

import numpy as np

from pace.config import ScientificSettings
from pace.core.decision import (
    TieredLocalEvidence,
    apply_protected_local_peer_rule,
    evaluate_peer_union,
)
from pace.experiment.arms import (
    calibration_rule,
    designed_decisions,
    protected_channel_decisions,
    union_calibration,
)
from pace.experiment.inputs import LocalEvidence, PeerEvidence, TargetFrame
from pace.types import (
    AllocationAudit,
    BooleanArray,
    ChannelDecisions,
    FalseAlarmBudget,
    FloatArray,
    OperatingCurve,
    OperatingMethod,
    OperatingPoint,
    TieredCalibrationRule,
)


def curve_from_alerts(
    method: OperatingMethod,
    frame: TargetFrame,
    grid: tuple[FalseAlarmBudget, ...],
    alerts_at: Callable[[FalseAlarmBudget], BooleanArray],
) -> OperatingCurve:
    points: list[OperatingPoint] = []
    for budget in grid:
        alerts = alerts_at(budget)
        points.append(
            OperatingPoint(
                nominal_budget=budget,
                false_alert_rate=frame.false_alert_rate(alerts),
                detection_rates=frame.detection_rates(alerts),
            )
        )
    return OperatingCurve(method=method, points=tuple(points))


def tiered_curve(
    method: OperatingMethod,
    frame: TargetFrame,
    peers: PeerEvidence,
    local: LocalEvidence,
    scientific: ScientificSettings,
    rule: TieredCalibrationRule,
) -> OperatingCurve:
    reserve = scientific.reserve_fraction

    def alerts_at(budget: FalseAlarmBudget) -> BooleanArray:
        return protected_channel_decisions(
            local,
            peers.calibration_evidence,
            peers.query_evidence,
            scientific,
            budget * reserve,
            budget * (1.0 - reserve),
            rule,
        ).combined_alerts

    return curve_from_alerts(method, frame, scientific.operating_grid, alerts_at)


def mean_ensemble_curve(
    frame: TargetFrame,
    peers: PeerEvidence,
    local: LocalEvidence,
    scientific: ScientificSettings,
) -> OperatingCurve:
    reserve = scientific.reserve_fraction

    def alerts_at(budget: FalseAlarmBudget) -> BooleanArray:
        return apply_protected_local_peer_rule(
            local.calibration,
            local.query,
            peers.calibration_evidence,
            peers.query_evidence,
            scientific.calibration_block_size,
            budget * reserve,
            budget * (1.0 - reserve),
            scientific.joint_scale_cap,
            scientific.joint_scale_iterations,
        ).combined_alerts

    return curve_from_alerts(
        OperatingMethod.MEAN_ENSEMBLE_BLOCK_MAXIMUM, frame, scientific.operating_grid, alerts_at
    )


def matched_mean_ensemble_curve(
    frame: TargetFrame,
    peers: PeerEvidence,
    local: LocalEvidence,
    scientific: ScientificSettings,
) -> OperatingCurve:
    reserve = scientific.reserve_fraction
    rule = calibration_rule(scientific)
    anchor_only = TieredLocalEvidence(
        local.calibration,
        local.query,
        np.empty((0, local.calibration.size)),
        np.empty((0, local.query.size)),
    )

    def alerts_at(budget: FalseAlarmBudget) -> BooleanArray:
        return designed_decisions(
            anchor_only,
            peers.calibration_evidence,
            peers.query_evidence,
            scientific,
            budget * reserve,
            budget * (1.0 - reserve),
            rule,
        ).combined_alerts

    return curve_from_alerts(
        OperatingMethod.MEAN_ENSEMBLE_SAME_CALIBRATION, frame, scientific.operating_grid, alerts_at
    )


def score_threshold_curve(
    method: OperatingMethod,
    frame: TargetFrame,
    calibration_scores: FloatArray,
    query_scores: FloatArray,
    scientific: ScientificSettings,
) -> OperatingCurve:
    union = union_calibration(scientific)

    def alerts_at(budget: FalseAlarmBudget) -> BooleanArray:
        _, alerts = evaluate_peer_union(
            calibration_scores[np.newaxis, :],
            query_scores[np.newaxis, :],
            scientific.calibration_block_size,
            budget,
            union.scale_cap,
            union.scale_iterations,
            None,
            union.mode,
            union.folds,
            union.resolution_rows,
        )
        return alerts

    return curve_from_alerts(method, frame, scientific.operating_grid, alerts_at)


def standard_curves(
    frame: TargetFrame,
    peers: PeerEvidence,
    local: LocalEvidence,
    scientific: ScientificSettings,
) -> tuple[OperatingCurve, ...]:
    curves = [
        tiered_curve(
            OperatingMethod.PACE, frame, peers, local, scientific, calibration_rule(scientific)
        )
    ]
    for ablation in scientific.ablation_curves:
        rule = calibration_rule(scientific, ablation.policy, ablation.mode, ablation.design)
        curves.append(
            tiered_curve(ablation.operating_method, frame, peers, local, scientific, rule)
        )
    curves.append(mean_ensemble_curve(frame, peers, local, scientific))
    curves.append(matched_mean_ensemble_curve(frame, peers, local, scientific))
    return tuple(curves)


def allocation_audit(
    decisions: ChannelDecisions,
    frame: TargetFrame,
    peers: PeerEvidence,
    scientific: ScientificSettings,
) -> AllocationAudit | None:
    if decisions.anchor_budget is None:
        return None
    alpha = scientific.alpha.fraction
    return AllocationAudit(
        nominal_budget=alpha,
        peer_budget=alpha * (1.0 - scientific.reserve_fraction),
        local_budget=alpha * scientific.reserve_fraction,
        anchor_budget=decisions.anchor_budget,
        bank_budget=decisions.bank_budget,
        calibration_rows=frame.calibration.shape[0],
        peer_expert_count=len(peers.experts),
        bank_expert_count=0 if decisions.bank_budget is None else len(decisions.bank_thresholds),
    )
