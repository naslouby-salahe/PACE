import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from pace.core.union import PeerCalibration, expert_union_alerts, validate_peer_calibration_matrix
from pace.types import (
    CrossFitFoldCount,
    FalseAlarmBudget,
    FalseAlertRate,
    FloatArray,
    JointMultiplier,
    JointScaleCap,
    ObservationCount,
    RowIndexArray,
    ScaleSearchIterations,
    SearchBound,
    Threshold,
)


def _fold_indices(
    row_count: ObservationCount, folds: CrossFitFoldCount
) -> tuple[RowIndexArray, ...]:
    if row_count < folds:
        raise ValueError("Cross-fitted calibration needs at least one row per fold")
    return tuple(
        np.asarray(indices, dtype=np.int64)
        for indices in np.array_split(np.arange(row_count, dtype=np.int64), folds)
    )


def _sorted_threshold_row(ordered: FloatArray, fraction: FalseAlarmBudget) -> Threshold:
    ceiling = SearchBound.SHARE_CEILING.bound
    return ordered[math.ceil((1.0 - min(fraction, ceiling)) * (ordered.size - 1))].item()


@dataclass(frozen=True, slots=True)
class _CrossFitter:
    evidence: FloatArray
    folds: tuple[RowIndexArray, ...]
    complements: tuple[FloatArray, ...]
    everything: FloatArray
    shares: tuple[FalseAlarmBudget, ...]

    @classmethod
    def of(
        cls,
        evidence: FloatArray,
        folds: CrossFitFoldCount,
        shares: tuple[FalseAlarmBudget, ...],
    ) -> "_CrossFitter":
        fold_rows = _fold_indices(evidence.shape[1], folds)
        everything = np.arange(evidence.shape[1], dtype=np.int64)
        return cls(
            evidence,
            fold_rows,
            tuple(
                np.sort(evidence[:, np.setdiff1d(everything, fold)], axis=1) for fold in fold_rows
            ),
            np.sort(evidence, axis=1),
            shares,
        )

    @classmethod
    def forward(
        cls,
        evidence: FloatArray,
        folds: CrossFitFoldCount,
        shares: tuple[FalseAlarmBudget, ...],
    ) -> "_CrossFitter":
        fold_rows = _fold_indices(evidence.shape[1], folds)[1:]
        return cls(
            evidence,
            fold_rows,
            tuple(np.sort(evidence[:, : fold[0]], axis=1) for fold in fold_rows),
            np.sort(evidence, axis=1),
            shares,
        )

    def thresholds(self, ordered: FloatArray, multiplier: JointMultiplier) -> tuple[Threshold, ...]:
        return tuple(
            _sorted_threshold_row(row, share * multiplier)
            for row, share in zip(ordered, self.shares, strict=True)
        )

    def held_out_rates(self, multiplier: JointMultiplier) -> tuple[FalseAlertRate, ...]:
        return tuple(
            np.mean(
                expert_union_alerts(self.evidence[:, fold], self.thresholds(complement, multiplier))
            ).item()
            for fold, complement in zip(self.folds, self.complements, strict=True)
        )

    def held_out_rate(self, multiplier: JointMultiplier) -> FalseAlertRate:
        return np.mean(self.held_out_rates(multiplier)).item()


def _fit_with_fitter(
    fitter: _CrossFitter,
    peer_budget: FalseAlarmBudget,
    scale_cap: JointScaleCap,
    scale_iterations: ScaleSearchIterations,
) -> PeerCalibration:
    lower, upper = 1.0 / scale_cap, scale_cap
    if fitter.held_out_rate(lower) > peer_budget:
        chosen = lower
    elif fitter.held_out_rate(upper) <= peer_budget:
        chosen = upper
    else:
        chosen = lower
        for _ in range(scale_iterations):
            midpoint = math.sqrt(chosen * upper)
            if fitter.held_out_rate(midpoint) <= peer_budget:
                chosen = midpoint
            else:
                upper = midpoint
    return PeerCalibration(
        fitter.thresholds(fitter.everything, chosen), chosen, fitter.held_out_rates(chosen)
    )


def _expert_shares(
    calibration_evidence: FloatArray,
    peer_budget: FalseAlarmBudget,
    expert_shares: Sequence[FalseAlarmBudget] | None,
) -> tuple[FalseAlarmBudget, ...]:
    validate_peer_calibration_matrix(calibration_evidence)
    expert_count = calibration_evidence.shape[0]
    shares = (
        (peer_budget / expert_count,) * expert_count
        if expert_shares is None
        else tuple(expert_shares)
    )
    if len(shares) != expert_count:
        raise ValueError("Every expert needs one budget share")
    return shares


def fit_cross_fitted_peer_union(
    calibration_evidence: FloatArray,
    folds: CrossFitFoldCount,
    peer_budget: FalseAlarmBudget,
    scale_cap: JointScaleCap,
    scale_iterations: ScaleSearchIterations,
    expert_shares: Sequence[FalseAlarmBudget] | None = None,
) -> PeerCalibration:
    shares = _expert_shares(calibration_evidence, peer_budget, expert_shares)
    return _fit_with_fitter(
        _CrossFitter.of(calibration_evidence, folds, shares),
        peer_budget,
        scale_cap,
        scale_iterations,
    )


def fit_forward_chained_peer_union(
    calibration_evidence: FloatArray,
    folds: CrossFitFoldCount,
    peer_budget: FalseAlarmBudget,
    scale_cap: JointScaleCap,
    scale_iterations: ScaleSearchIterations,
    expert_shares: Sequence[FalseAlarmBudget] | None = None,
) -> PeerCalibration:
    shares = _expert_shares(calibration_evidence, peer_budget, expert_shares)
    return _fit_with_fitter(
        _CrossFitter.forward(calibration_evidence, folds, shares),
        peer_budget,
        scale_cap,
        scale_iterations,
    )
