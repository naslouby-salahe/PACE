from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from pace.types import (
    BooleanArray,
    FalseAlertRate,
    FloatArray,
    JointMultiplier,
    Threshold,
)


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


def validate_peer_calibration_matrix(evidence: FloatArray) -> None:
    if evidence.ndim != 2 or evidence.size == 0:
        raise ValueError("Peer calibration evidence must be a non-empty expert-by-row matrix")
    if not np.isfinite(evidence).all():
        raise ValueError("Peer calibration evidence must be finite")
