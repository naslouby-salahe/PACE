import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from pace.types import (
    AnchorShare,
    BooleanArray,
    CalibrationBlockSize,
    ChannelDecisions,
    FalseAlarmBudget,
    FalseAlertRate,
    FloatArray,
    FloatMatrix,
    JointMultiplier,
    JointScaleCap,
    MaxFusionDecisions,
    ObservationCount,
    QuantileLevel,
    RowIndexArray,
    ScaleSearchIterations,
    SearchBound,
    Threshold,
)


def calibration_block_indices(
    row_count: ObservationCount,
    block_size: CalibrationBlockSize,
) -> tuple[RowIndexArray, ...]:
    if row_count < 1:
        raise ValueError("At least one calibration row is required")
    block_count = max(1, row_count // block_size)
    return tuple(
        np.asarray(indices, dtype=np.int64)
        for indices in np.array_split(np.arange(row_count, dtype=np.int64), block_count)
    )


def consecutive_blocks(
    values: FloatArray, block_size: CalibrationBlockSize
) -> tuple[FloatArray, ...]:
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("Calibration values must be a non-empty finite one-dimensional array")
    return tuple(values[indices] for indices in calibration_block_indices(values.size, block_size))


def higher_quantile(values: FloatArray, quantile: QuantileLevel) -> Threshold:
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("Quantile values must be a non-empty finite one-dimensional array")
    return np.quantile(values, quantile, method="higher").item()


def block_max_threshold(blocks: Sequence[FloatArray], budget: FalseAlarmBudget) -> Threshold:
    if not blocks:
        raise ValueError("At least one calibration block is required")
    return max(higher_quantile(block, 1.0 - budget) for block in blocks)


def sorted_blocks(blocks: Sequence[FloatArray]) -> tuple[FloatArray, ...]:
    if not blocks:
        raise ValueError("At least one calibration block is required")
    for block in blocks:
        if block.ndim != 1 or block.size == 0 or not np.isfinite(block).all():
            raise ValueError("Quantile values must be a non-empty finite one-dimensional array")
    return tuple(np.sort(block) for block in blocks)


def sorted_block_max_threshold(
    ordered_blocks: Sequence[FloatArray], budget: FalseAlarmBudget
) -> Threshold:
    level = 1.0 - budget
    return max(block[math.ceil(level * (block.size - 1))].item() for block in ordered_blocks)


def expert_union_alerts(
    expert_evidence: FloatArray,
    thresholds: Sequence[Threshold],
) -> BooleanArray:
    if expert_evidence.ndim != 2 or expert_evidence.shape[0] != len(thresholds):
        raise ValueError("Evidence rows and expert thresholds must align")
    if expert_evidence.shape[0] == 0 or expert_evidence.shape[1] == 0:
        raise ValueError("Expert evidence must include experts and samples")
    numeric_thresholds = np.asarray(thresholds)
    return np.any(expert_evidence > numeric_thresholds[:, np.newaxis], axis=0)


@dataclass(frozen=True, slots=True)
class PeerCalibration:
    thresholds: tuple[Threshold, ...]
    joint_multiplier: JointMultiplier
    block_alert_rates: tuple[FalseAlertRate, ...]

    @property
    def worst_block_rate(self) -> FalseAlertRate:
        return max(self.block_alert_rates)


def _expert_thresholds(
    ordered_blocks: Sequence[Sequence[FloatArray]],
    shares: Sequence[FalseAlarmBudget],
    multiplier: JointMultiplier,
) -> tuple[Threshold, ...]:
    scaled_fractions = tuple(share * multiplier for share in shares)
    if not all(0.0 < fraction < 1.0 for fraction in scaled_fractions):
        raise ValueError("Scaled expert shares must remain strictly between zero and one")
    return tuple(
        sorted_block_max_threshold(blocks, fraction)
        for blocks, fraction in zip(ordered_blocks, scaled_fractions, strict=True)
    )


def _sorted_expert_blocks(
    calibration_evidence: FloatArray, block_indices: Sequence[RowIndexArray]
) -> tuple[tuple[FloatArray, ...], ...]:
    return tuple(
        sorted_blocks(
            tuple(calibration_evidence[expert_index, indices] for indices in block_indices)
        )
        for expert_index in range(calibration_evidence.shape[0])
    )


def _peer_union_rates(
    calibration_evidence: FloatArray,
    block_indices: Sequence[RowIndexArray],
    thresholds: Sequence[Threshold],
) -> tuple[FalseAlertRate, ...]:
    return tuple(
        np.mean(expert_union_alerts(calibration_evidence[:, indices], thresholds)).item()
        for indices in block_indices
    )


def _validate_peer_calibration_matrix(evidence: FloatArray) -> None:
    if evidence.ndim != 2 or evidence.size == 0:
        raise ValueError("Peer calibration evidence must be a non-empty expert-by-row matrix")
    if not np.isfinite(evidence).all():
        raise ValueError("Peer calibration evidence must be finite")


def _validate_channel_evidence(
    local_calibration_evidence: FloatArray,
    local_evaluation_evidence: FloatArray,
    peer_calibration_evidence: FloatMatrix,
    peer_evaluation_evidence: FloatMatrix,
) -> None:
    if (
        local_calibration_evidence.ndim != 1
        or local_calibration_evidence.size == 0
        or local_evaluation_evidence.ndim != 1
        or local_evaluation_evidence.size == 0
        or peer_calibration_evidence.ndim != 2
        or peer_calibration_evidence.shape[0] == 0
        or peer_calibration_evidence.shape[1] != local_calibration_evidence.size
        or peer_evaluation_evidence.ndim != 2
        or peer_evaluation_evidence.shape[0] != peer_calibration_evidence.shape[0]
        or peer_evaluation_evidence.shape[1] != local_evaluation_evidence.size
        or not np.isfinite(local_calibration_evidence).all()
        or not np.isfinite(local_evaluation_evidence).all()
        or not np.isfinite(peer_calibration_evidence).all()
        or not np.isfinite(peer_evaluation_evidence).all()
    ):
        raise ValueError("Local and peer evidence must be finite and row-aligned")


def _validate_union_inputs(
    calibration_evidence: FloatMatrix,
    evaluation_evidence: FloatMatrix,
    calibration_block_size: CalibrationBlockSize,
) -> None:
    _validate_peer_calibration_matrix(calibration_evidence)
    if calibration_block_size < 1:
        raise ValueError("Calibration block size must be positive")
    if calibration_evidence.shape[0] != evaluation_evidence.shape[0]:
        raise ValueError("Calibration and evaluation evidence must have the same experts")
    if evaluation_evidence.ndim != 2 or evaluation_evidence.shape[1] == 0:
        raise ValueError("Peer evaluation evidence must contain rows")
    if not np.isfinite(evaluation_evidence).all():
        raise ValueError("Peer evaluation evidence must be finite")


@dataclass(frozen=True, slots=True)
class _UnionCalibrator:
    evidence: FloatArray
    block_indices: tuple[RowIndexArray, ...]
    ordered: tuple[tuple[FloatArray, ...], ...]
    shares: tuple[FalseAlarmBudget, ...]

    def at(self, multiplier: JointMultiplier) -> PeerCalibration:
        thresholds = _expert_thresholds(self.ordered, self.shares, multiplier)
        return PeerCalibration(
            thresholds,
            multiplier,
            _peer_union_rates(self.evidence, self.block_indices, thresholds),
        )


def _union_calibrator(
    calibration_evidence: FloatArray,
    calibration_block_size: CalibrationBlockSize,
    peer_budget: FalseAlarmBudget,
    expert_shares: Sequence[FalseAlarmBudget] | None,
) -> _UnionCalibrator:
    _validate_peer_calibration_matrix(calibration_evidence)
    if calibration_block_size < 1:
        raise ValueError("Calibration block size must be positive")
    expert_count, row_count = calibration_evidence.shape
    block_indices = calibration_block_indices(row_count, calibration_block_size)
    shares = (
        (peer_budget / expert_count,) * expert_count
        if expert_shares is None
        else tuple(expert_shares)
    )
    if len(shares) != expert_count:
        raise ValueError("Every expert needs one budget share")
    return _UnionCalibrator(
        calibration_evidence,
        block_indices,
        _sorted_expert_blocks(calibration_evidence, block_indices),
        shares,
    )


def fit_peer_union(
    calibration_evidence: FloatArray,
    calibration_block_size: CalibrationBlockSize,
    peer_budget: FalseAlarmBudget,
    scale_cap: JointScaleCap,
    scale_iterations: ScaleSearchIterations,
    expert_shares: Sequence[FalseAlarmBudget] | None = None,
) -> PeerCalibration:
    calibrator = _union_calibrator(
        calibration_evidence, calibration_block_size, peer_budget, expert_shares
    )
    largest_share = max(calibrator.shares)
    maximum_multiplier = scale_cap
    if largest_share * maximum_multiplier >= 1.0:
        maximum_multiplier = SearchBound.SHARE_CEILING.bound / largest_share
        if maximum_multiplier < 1.0:
            raise ValueError("Expert budget shares must remain strictly below one")

    lower = calibrator.at(1.0)
    if lower.worst_block_rate > peer_budget:
        return lower
    upper_multiplier = min(2.0, maximum_multiplier)
    upper = calibrator.at(upper_multiplier)
    while upper.worst_block_rate <= peer_budget:
        if upper_multiplier == maximum_multiplier:
            return upper
        lower = upper
        upper_multiplier = min(upper_multiplier * 2.0, maximum_multiplier)
        upper = calibrator.at(upper_multiplier)
    for _ in range(scale_iterations):
        midpoint = (lower.joint_multiplier + upper_multiplier) / 2.0
        candidate = calibrator.at(midpoint)
        if candidate.worst_block_rate <= peer_budget:
            lower = candidate
        else:
            upper_multiplier = midpoint
    return lower


def fit_bonferroni_peer_union(
    calibration_evidence: FloatArray,
    calibration_block_size: CalibrationBlockSize,
    peer_budget: FalseAlarmBudget,
) -> PeerCalibration:
    return _union_calibrator(calibration_evidence, calibration_block_size, peer_budget, None).at(
        1.0
    )


def evaluate_peer_union(
    calibration_evidence: FloatArray,
    evaluation_evidence: FloatArray,
    calibration_block_size: CalibrationBlockSize,
    peer_budget: FalseAlarmBudget,
    scale_cap: JointScaleCap,
    scale_iterations: ScaleSearchIterations,
    expert_shares: Sequence[FalseAlarmBudget] | None = None,
) -> tuple[PeerCalibration, BooleanArray]:
    _validate_union_inputs(calibration_evidence, evaluation_evidence, calibration_block_size)
    calibration = fit_peer_union(
        calibration_evidence,
        calibration_block_size,
        peer_budget,
        scale_cap,
        scale_iterations,
        expert_shares,
    )
    return calibration, expert_union_alerts(evaluation_evidence, calibration.thresholds)


def evaluate_bonferroni_peer_union(
    calibration_evidence: FloatArray,
    evaluation_evidence: FloatArray,
    calibration_block_size: CalibrationBlockSize,
    peer_budget: FalseAlarmBudget,
) -> tuple[PeerCalibration, BooleanArray]:
    _validate_union_inputs(calibration_evidence, evaluation_evidence, calibration_block_size)
    calibration = fit_bonferroni_peer_union(
        calibration_evidence, calibration_block_size, peer_budget
    )
    return calibration, expert_union_alerts(evaluation_evidence, calibration.thresholds)


def _local_threshold(
    local_calibration_evidence: FloatArray,
    calibration_block_size: CalibrationBlockSize,
    local_budget: FalseAlarmBudget,
) -> Threshold:
    return block_max_threshold(
        consecutive_blocks(local_calibration_evidence, calibration_block_size), local_budget
    )


def _channel_decisions(
    local_threshold: Threshold,
    local_evaluation_evidence: FloatArray,
    peer_calibration: PeerCalibration,
    peer_alerts: BooleanArray,
) -> ChannelDecisions:
    local_alerts = local_evaluation_evidence > local_threshold
    return ChannelDecisions(
        local_threshold=local_threshold,
        expert_thresholds=peer_calibration.thresholds,
        joint_multiplier=peer_calibration.joint_multiplier,
        peer_block_alert_rates=peer_calibration.block_alert_rates,
        local_alerts=local_alerts,
        peer_alerts=peer_alerts,
        combined_alerts=np.logical_or(local_alerts, peer_alerts),
    )


def apply_protected_local_peer_rule(
    local_calibration_evidence: FloatArray,
    local_evaluation_evidence: FloatArray,
    peer_calibration_evidence: FloatMatrix,
    peer_evaluation_evidence: FloatMatrix,
    calibration_block_size: CalibrationBlockSize,
    local_budget: FalseAlarmBudget,
    peer_budget: FalseAlarmBudget,
    scale_cap: JointScaleCap,
    scale_iterations: ScaleSearchIterations,
) -> ChannelDecisions:
    _validate_channel_evidence(
        local_calibration_evidence,
        local_evaluation_evidence,
        peer_calibration_evidence,
        peer_evaluation_evidence,
    )
    peer_calibration, peer_alerts = evaluate_peer_union(
        peer_calibration_evidence,
        peer_evaluation_evidence,
        calibration_block_size,
        peer_budget,
        scale_cap,
        scale_iterations,
    )
    return _channel_decisions(
        _local_threshold(local_calibration_evidence, calibration_block_size, local_budget),
        local_evaluation_evidence,
        peer_calibration,
        peer_alerts,
    )


@dataclass(frozen=True, slots=True)
class TieredLocalEvidence:
    anchor_calibration: FloatArray
    anchor_evaluation: FloatArray
    bank_calibration: FloatMatrix
    bank_evaluation: FloatMatrix

    def __post_init__(self) -> None:
        if (
            self.anchor_calibration.ndim != 1
            or self.anchor_evaluation.ndim != 1
            or self.bank_calibration.ndim != 2
            or self.bank_evaluation.ndim != 2
            or self.bank_calibration.shape[1] != self.anchor_calibration.size
            or self.bank_evaluation.shape[1] != self.anchor_evaluation.size
            or self.bank_calibration.shape[0] != self.bank_evaluation.shape[0]
        ):
            raise ValueError("Tiered local evidence must align anchor and bank rows")

    @property
    def has_bank(self) -> bool:
        return self.bank_calibration.shape[0] > 0


def _anchor_tier_budget(
    local_budget: FalseAlarmBudget,
    anchor_share: AnchorShare,
    calibration_block_size: CalibrationBlockSize,
) -> FalseAlarmBudget:
    tier = local_budget * anchor_share
    return tier if tier >= 1.0 / calibration_block_size else local_budget


def apply_tiered_local_peer_rule(
    local: TieredLocalEvidence,
    peer_calibration_evidence: FloatMatrix,
    peer_evaluation_evidence: FloatMatrix,
    calibration_block_size: CalibrationBlockSize,
    local_budget: FalseAlarmBudget,
    peer_budget: FalseAlarmBudget,
    anchor_share: AnchorShare,
    scale_cap: JointScaleCap,
    scale_iterations: ScaleSearchIterations,
) -> ChannelDecisions:
    _validate_channel_evidence(
        local.anchor_calibration,
        local.anchor_evaluation,
        peer_calibration_evidence,
        peer_evaluation_evidence,
    )
    peer_calibration, peer_alerts = evaluate_peer_union(
        peer_calibration_evidence,
        peer_evaluation_evidence,
        calibration_block_size,
        peer_budget,
        scale_cap,
        scale_iterations,
    )
    anchor_calibration, anchor_alerts = evaluate_peer_union(
        local.anchor_calibration[np.newaxis, :],
        local.anchor_evaluation[np.newaxis, :],
        calibration_block_size,
        _anchor_tier_budget(local_budget, anchor_share, calibration_block_size)
        if local.has_bank
        else local_budget,
        scale_cap,
        scale_iterations,
    )
    bank_thresholds: tuple[Threshold, ...] = ()
    local_alerts = anchor_alerts
    if local.has_bank:
        bank_calibration, bank_alerts = evaluate_peer_union(
            local.bank_calibration,
            local.bank_evaluation,
            calibration_block_size,
            local_budget * (1.0 - anchor_share),
            scale_cap,
            scale_iterations,
        )
        bank_thresholds = bank_calibration.thresholds
        local_alerts = anchor_alerts | bank_alerts
    return ChannelDecisions(
        local_threshold=anchor_calibration.thresholds[0],
        expert_thresholds=peer_calibration.thresholds,
        joint_multiplier=peer_calibration.joint_multiplier,
        peer_block_alert_rates=peer_calibration.block_alert_rates,
        local_alerts=local_alerts,
        peer_alerts=peer_alerts,
        combined_alerts=np.logical_or(local_alerts, peer_alerts),
        bank_thresholds=bank_thresholds,
    )


def apply_local_union_peer_rule(
    local_calibration_evidence: FloatMatrix,
    local_evaluation_evidence: FloatMatrix,
    peer_calibration_evidence: FloatMatrix,
    peer_evaluation_evidence: FloatMatrix,
    calibration_block_size: CalibrationBlockSize,
    local_budget: FalseAlarmBudget,
    peer_budget: FalseAlarmBudget,
    scale_cap: JointScaleCap,
    scale_iterations: ScaleSearchIterations,
    relative_local_shares: Sequence[FalseAlarmBudget] | None = None,
) -> ChannelDecisions:
    _validate_channel_evidence(
        local_calibration_evidence[0],
        local_evaluation_evidence[0],
        peer_calibration_evidence,
        peer_evaluation_evidence,
    )
    peer_calibration, peer_alerts = evaluate_peer_union(
        peer_calibration_evidence,
        peer_evaluation_evidence,
        calibration_block_size,
        peer_budget,
        scale_cap,
        scale_iterations,
    )
    local_calibration, local_alerts = evaluate_peer_union(
        local_calibration_evidence,
        local_evaluation_evidence,
        calibration_block_size,
        local_budget,
        scale_cap,
        scale_iterations,
        None
        if relative_local_shares is None
        else tuple(share * local_budget for share in relative_local_shares),
    )
    return ChannelDecisions(
        local_threshold=local_calibration.thresholds[0],
        expert_thresholds=peer_calibration.thresholds,
        joint_multiplier=peer_calibration.joint_multiplier,
        peer_block_alert_rates=peer_calibration.block_alert_rates,
        local_alerts=local_alerts,
        peer_alerts=peer_alerts,
        combined_alerts=np.logical_or(local_alerts, peer_alerts),
        bank_thresholds=local_calibration.thresholds[1:],
    )


def apply_bonferroni_local_peer_rule(
    local_calibration_evidence: FloatArray,
    local_evaluation_evidence: FloatArray,
    peer_calibration_evidence: FloatMatrix,
    peer_evaluation_evidence: FloatMatrix,
    calibration_block_size: CalibrationBlockSize,
    local_budget: FalseAlarmBudget,
    peer_budget: FalseAlarmBudget,
) -> ChannelDecisions:
    _validate_channel_evidence(
        local_calibration_evidence,
        local_evaluation_evidence,
        peer_calibration_evidence,
        peer_evaluation_evidence,
    )
    peer_calibration, peer_alerts = evaluate_bonferroni_peer_union(
        peer_calibration_evidence,
        peer_evaluation_evidence,
        calibration_block_size,
        peer_budget,
    )
    return _channel_decisions(
        _local_threshold(local_calibration_evidence, calibration_block_size, local_budget),
        local_evaluation_evidence,
        peer_calibration,
        peer_alerts,
    )


def apply_max_fusion_rule(
    local_calibration_evidence: FloatArray,
    local_evaluation_evidence: FloatArray,
    peer_calibration_evidence: FloatMatrix,
    peer_evaluation_evidence: FloatMatrix,
    calibration_block_size: CalibrationBlockSize,
    local_budget: FalseAlarmBudget,
    peer_budget: FalseAlarmBudget,
) -> MaxFusionDecisions:
    _validate_channel_evidence(
        local_calibration_evidence,
        local_evaluation_evidence,
        peer_calibration_evidence,
        peer_evaluation_evidence,
    )
    local_threshold = _local_threshold(
        local_calibration_evidence, calibration_block_size, local_budget
    )
    peer_threshold = _local_threshold(
        np.max(peer_calibration_evidence, axis=0), calibration_block_size, peer_budget
    )
    local_alerts = local_evaluation_evidence > local_threshold
    peer_alerts = np.max(peer_evaluation_evidence, axis=0) > peer_threshold
    return MaxFusionDecisions(
        local_threshold=local_threshold,
        peer_threshold=peer_threshold,
        local_alerts=local_alerts,
        peer_alerts=peer_alerts,
        combined_alerts=np.logical_or(local_alerts, peer_alerts),
    )
