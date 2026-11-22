import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from pace.core.union import PeerCalibration, validate_peer_calibration_matrix
from pace.types import (
    FalseAlarmBudget,
    FalseAlertRate,
    FloatArray,
    JointMultiplier,
    JointScaleCap,
    ScaleSearchIterations,
    SearchBound,
    Threshold,
)


def rank_threshold(ordered: FloatArray, fraction: FalseAlarmBudget) -> Threshold:
    rows = ordered.size
    clipped = min(fraction, SearchBound.SHARE_CEILING.bound)
    rank = min(rows, max(1, math.floor(clipped * (rows + 1))))
    return ordered[rows - rank].item()


@dataclass(frozen=True, slots=True)
class _PooledCalibrator:
    evidence: FloatArray
    ordered: FloatArray
    shares: tuple[FalseAlarmBudget, ...]

    def thresholds(self, multiplier: JointMultiplier) -> tuple[Threshold, ...]:
        return tuple(
            rank_threshold(row, share * multiplier)
            for row, share in zip(self.ordered, self.shares, strict=True)
        )

    def inclusive_union_rate(self, multiplier: JointMultiplier) -> FalseAlertRate:
        thresholds = np.asarray(self.thresholds(multiplier))
        return np.mean(np.any(self.evidence >= thresholds[:, np.newaxis], axis=0)).item()


def fit_pooled_peer_union(
    calibration_evidence: FloatArray,
    peer_budget: FalseAlarmBudget,
    scale_cap: JointScaleCap,
    scale_iterations: ScaleSearchIterations,
    expert_shares: Sequence[FalseAlarmBudget] | None = None,
) -> PeerCalibration:
    validate_peer_calibration_matrix(calibration_evidence)
    expert_count = calibration_evidence.shape[0]
    shares = (
        (peer_budget / expert_count,) * expert_count
        if expert_shares is None
        else tuple(expert_shares)
    )
    if len(shares) != expert_count:
        raise ValueError("Every expert needs one budget share")
    calibrator = _PooledCalibrator(
        calibration_evidence, np.sort(calibration_evidence, axis=1), shares
    )
    lower, upper = 1.0 / scale_cap, scale_cap
    if calibrator.inclusive_union_rate(lower) > peer_budget:
        chosen = lower
    elif calibrator.inclusive_union_rate(upper) <= peer_budget:
        chosen = upper
    else:
        chosen = lower
        for _ in range(scale_iterations):
            midpoint = math.sqrt(chosen * upper)
            if calibrator.inclusive_union_rate(midpoint) <= peer_budget:
                chosen = midpoint
            else:
                upper = midpoint
    return PeerCalibration(
        calibrator.thresholds(chosen), chosen, (calibrator.inclusive_union_rate(chosen),)
    )
