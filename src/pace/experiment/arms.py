from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from pace.config import ResolvedConfiguration, ScientificSettings
from pace.core.decision import (
    apply_local_union_peer_rule,
    apply_protected_local_peer_rule,
    apply_tiered_local_peer_rule,
    block_max_threshold,
    consecutive_blocks,
    evaluate_peer_union,
)
from pace.experiment.inputs import LocalEvidence, PeerEvidence, PreparedDevice, TargetFrame
from pace.experiment.models import (
    LocalBankEvidence,
    LocalBankResponses,
    margin_response_bank,
    select_local_training_rows,
)
from pace.types import (
    ArmPerformance,
    BooleanArray,
    ChannelDecisions,
    FalseAlarmBudget,
    FloatMatrix,
    LocalDetectorSeed,
    ReserveFraction,
    StudyArm,
)


def protected_channel_decisions(
    local: LocalEvidence,
    peer_calibration_evidence: FloatMatrix,
    peer_query_evidence: FloatMatrix,
    scientific: ScientificSettings,
    local_budget: FalseAlarmBudget,
    peer_budget: FalseAlarmBudget,
) -> ChannelDecisions:
    return apply_tiered_local_peer_rule(
        local.tiered(),
        peer_calibration_evidence,
        peer_query_evidence,
        scientific.calibration_block_size,
        local_budget,
        peer_budget,
        scientific.local_bank.anchor_share,
        scientific.joint_scale_cap,
        scientific.joint_scale_iterations,
    )


@dataclass(frozen=True, slots=True)
class BankArmContext:
    scientific: ScientificSettings
    target: PreparedDevice
    peers: PeerEvidence
    local: LocalEvidence
    local_budget: FalseAlarmBudget
    peer_budget: FalseAlarmBudget


def _tiered_arm_alerts(context: BankArmContext, bank: LocalBankEvidence) -> BooleanArray:
    scientific = context.scientific
    local_budget, peer_budget = context.local_budget, context.peer_budget
    return apply_tiered_local_peer_rule(
        context.local.tiered(bank),
        context.peers.calibration_evidence,
        context.peers.query_evidence,
        scientific.calibration_block_size,
        local_budget,
        peer_budget,
        scientific.local_bank.anchor_share,
        scientific.joint_scale_cap,
        scientific.joint_scale_iterations,
    ).combined_alerts


def _union_arm_alerts(
    context: BankArmContext,
    calibration: FloatMatrix,
    query: FloatMatrix,
    shares: tuple[FalseAlarmBudget, ...] | None,
) -> BooleanArray:
    scientific = context.scientific
    local_budget, peer_budget = context.local_budget, context.peer_budget
    return apply_local_union_peer_rule(
        calibration,
        query,
        context.peers.calibration_evidence,
        context.peers.query_evidence,
        scientific.calibration_block_size,
        local_budget,
        peer_budget,
        scientific.joint_scale_cap,
        scientific.joint_scale_iterations,
        shares,
    ).combined_alerts


def _single_union_alerts(context: BankArmContext, bank: LocalBankEvidence) -> BooleanArray:
    anchor_share = context.scientific.local_bank.anchor_share
    expert_count = bank.calibration.shape[0]
    return _union_arm_alerts(
        context,
        np.vstack((bank.calibration, bank.anchor_calibration[np.newaxis, :])),
        np.vstack((bank.query, bank.anchor_query[np.newaxis, :])),
        (*((1.0 - anchor_share) / expert_count,) * expert_count, anchor_share),
    )


def mean_ensemble_local_alerts(context: BankArmContext) -> BooleanArray:
    scientific = context.scientific
    local_budget, peer_budget = context.local_budget, context.peer_budget
    return apply_protected_local_peer_rule(
        context.local.calibration,
        context.local.query,
        context.peers.calibration_evidence,
        context.peers.query_evidence,
        scientific.calibration_block_size,
        local_budget,
        peer_budget,
        scientific.joint_scale_cap,
        scientific.joint_scale_iterations,
    ).combined_alerts


def _margin_response_alerts(context: BankArmContext, bank: LocalBankEvidence) -> BooleanArray:
    scientific = context.scientific
    peers = context.peers
    training = select_local_training_rows(
        context.target.training, scientific.local_detectors.local_training_rows
    )
    responses = LocalBankResponses(
        np.stack([expert.model.margins(training) for expert in peers.experts], axis=1),
        peers.reference_margins.T,
        peers.calibration_margins.T,
        peers.query_margins.T,
    )
    replaced = margin_response_bank(
        bank, responses, scientific.local_bank, LocalDetectorSeed(context.local.seed)
    )
    return _tiered_arm_alerts(context, replaced)


def bank_arm_alerts(
    context: BankArmContext, arm: StudyArm, bank: LocalBankEvidence
) -> BooleanArray:
    match arm:
        case StudyArm.BANK_WITHOUT_ANCHOR:
            return _union_arm_alerts(context, bank.calibration, bank.query, None)
        case StudyArm.BANK_SINGLE_UNION:
            return _single_union_alerts(context, bank)
        case StudyArm.RESPONSE_ON_MARGINS:
            return _margin_response_alerts(context, bank)
        case _:
            return _tiered_arm_alerts(context, bank.without_family(arm.removed_expert_family))


@dataclass(frozen=True, slots=True)
class ArmContext:
    configuration: ResolvedConfiguration
    target: PreparedDevice
    frame: TargetFrame
    peers: PeerEvidence
    local: LocalEvidence


def _alpha_budgets(
    context: ArmContext, reserve: ReserveFraction | None = None
) -> tuple[FalseAlarmBudget, FalseAlarmBudget]:
    scientific = context.configuration.scientific
    alpha = scientific.alpha.fraction
    reserve_fraction = scientific.reserve_fraction if reserve is None else reserve
    return (alpha * reserve_fraction), (alpha * (1.0 - reserve_fraction))


def _reserve_alerts(context: ArmContext, reserve: ReserveFraction) -> BooleanArray:
    local_budget, peer_budget = _alpha_budgets(context, reserve)
    return protected_channel_decisions(
        context.local,
        context.peers.calibration_evidence,
        context.peers.query_evidence,
        context.configuration.scientific,
        local_budget,
        peer_budget,
    ).combined_alerts


def _fused_peer_alerts(
    context: ArmContext, fuse: Callable[[FloatMatrix], FloatMatrix]
) -> BooleanArray:
    local_budget, peer_budget = _alpha_budgets(context)
    return protected_channel_decisions(
        context.local,
        fuse(context.peers.calibration_evidence),
        fuse(context.peers.query_evidence),
        context.configuration.scientific,
        local_budget,
        peer_budget,
    ).combined_alerts


def _mean_fusion(evidence: FloatMatrix) -> FloatMatrix:
    return np.mean(evidence, axis=0, keepdims=True)


def _max_fusion(evidence: FloatMatrix) -> FloatMatrix:
    return np.max(evidence, axis=0, keepdims=True)


def _local_only_alerts(context: ArmContext) -> BooleanArray:
    scientific = context.configuration.scientific
    threshold = block_max_threshold(
        consecutive_blocks(context.local.calibration, scientific.calibration_block_size),
        scientific.alpha.fraction,
    )
    return context.local.query > threshold


def _no_local_alerts(context: ArmContext) -> BooleanArray:
    scientific = context.configuration.scientific
    _, alerts = evaluate_peer_union(
        context.peers.calibration_evidence,
        context.peers.query_evidence,
        scientific.calibration_block_size,
        scientific.alpha.fraction,
        scientific.joint_scale_cap,
        scientific.joint_scale_iterations,
    )
    return alerts


def _local_branch_alerts(context: ArmContext, arm: StudyArm) -> BooleanArray:
    local_budget, peer_budget = _alpha_budgets(context)
    bank_context = BankArmContext(
        context.configuration.scientific,
        context.target,
        context.peers,
        context.local,
        local_budget,
        peer_budget,
    )
    if arm is StudyArm.MEAN_ENSEMBLE_LOCAL_BRANCH:
        return mean_ensemble_local_alerts(bank_context)
    return bank_arm_alerts(bank_context, arm, context.local.bank)


def _arm_alerts(context: ArmContext, arm: StudyArm) -> BooleanArray:
    match arm:
        case StudyArm.NO_LOCAL_CHANNEL:
            return _no_local_alerts(context)
        case StudyArm.LOCAL_ONLY:
            return _local_only_alerts(context)
        case StudyArm.RESERVE_QUARTER:
            return _reserve_alerts(context, arm.reserve_variant)
        case StudyArm.PEER_MEAN_FUSION:
            return _fused_peer_alerts(context, _mean_fusion)
        case StudyArm.PEER_MAX_FUSION:
            return _fused_peer_alerts(context, _max_fusion)
        case _:
            return _local_branch_alerts(context, arm)


def _arm_performance(context: ArmContext, arm: StudyArm) -> ArmPerformance:
    alerts = _arm_alerts(context, arm)
    return ArmPerformance(
        arm=arm,
        false_alert_rate=context.frame.false_alert_rate(alerts),
        detection_rates=context.frame.detection_rates(alerts),
    )


def evaluate_arms(context: ArmContext) -> tuple[ArmPerformance, ...]:
    return tuple(
        _arm_performance(context, arm) for arm in context.configuration.scientific.study_arms
    )
