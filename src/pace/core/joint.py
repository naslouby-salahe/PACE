import numpy as np

from pace.core.decision import TieredLocalEvidence, evaluate_peer_union
from pace.core.union import expert_union_alerts
from pace.types import (
    CalibrationBlockSize,
    ChannelDecisions,
    FalseAlarmBudget,
    FloatMatrix,
    TieredCalibrationRule,
)


def apply_joint_local_peer_rule(
    local: TieredLocalEvidence,
    peer_calibration_evidence: FloatMatrix,
    peer_evaluation_evidence: FloatMatrix,
    calibration_block_size: CalibrationBlockSize,
    local_budget: FalseAlarmBudget,
    peer_budget: FalseAlarmBudget,
    rule: TieredCalibrationRule,
) -> ChannelDecisions:
    banked = local.has_bank
    anchor_budget = local_budget * rule.anchor_share if banked else local_budget
    bank_budget = local_budget - anchor_budget if banked else 0.0
    bank_count = local.bank_calibration.shape[0]
    relative = local.bank_shares or ((1.0 / bank_count,) * bank_count if banked else ())
    peer_count = peer_calibration_evidence.shape[0]
    shares = (
        anchor_budget,
        *(bank_budget * share for share in relative),
        *((peer_budget / peer_count,) * peer_count),
    )
    calibration = np.vstack(
        (local.anchor_calibration[np.newaxis, :], local.bank_calibration, peer_calibration_evidence)
    )
    evaluation = np.vstack(
        (local.anchor_evaluation[np.newaxis, :], local.bank_evaluation, peer_evaluation_evidence)
    )
    fitted, _ = evaluate_peer_union(
        calibration,
        evaluation,
        calibration_block_size,
        local_budget + peer_budget,
        rule.union.scale_cap,
        rule.union.scale_iterations,
        shares,
        rule.union.mode,
        rule.union.folds,
        rule.union.resolution_rows,
    )
    split = 1 + bank_count
    local_alerts = expert_union_alerts(evaluation[:split], fitted.thresholds[:split])
    peer_alerts = expert_union_alerts(evaluation[split:], fitted.thresholds[split:])
    return ChannelDecisions(
        local_threshold=fitted.thresholds[0],
        expert_thresholds=fitted.thresholds[split:],
        joint_multiplier=fitted.joint_multiplier,
        peer_block_alert_rates=fitted.block_alert_rates,
        local_alerts=local_alerts,
        peer_alerts=peer_alerts,
        combined_alerts=np.logical_or(local_alerts, peer_alerts),
        bank_thresholds=fitted.thresholds[1:split],
        anchor_budget=anchor_budget,
        bank_budget=bank_budget if banked else None,
    )
