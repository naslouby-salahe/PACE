import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from pace.config import SplitSettings
from pace.types import (
    ArtifactPayload,
    AttackCapture,
    AttackCaptureSplit,
    AttackHalfIndices,
    AttackRowsPerHalfCap,
    AttackType,
    ChronologicalIndices,
    DatasetClientId,
    DatasetSplitPlan,
    FloatArray,
    LengthPrefixedHasher,
    MinimumAttackRows,
    MinimumCalibrationRows,
    MinimumPeerCount,
    ObservationCount,
    RowIndexArray,
    ScientificDigest,
    SplitRole,
    StudyStratum,
    TargetEligibility,
    TargetEligibilityIssue,
    TimestampArray,
)


def assess_target_eligibility(
    attack_type_count: ObservationCount,
    calibration_row_count: ObservationCount,
    peer_count: ObservationCount,
    minimum_calibration_rows: MinimumCalibrationRows,
    minimum_peer_count: MinimumPeerCount,
) -> TargetEligibility:
    issues: list[TargetEligibilityIssue] = []
    if attack_type_count == 0:
        issues.append(TargetEligibilityIssue.NO_ATTACK_TYPES)
    if calibration_row_count < minimum_calibration_rows:
        issues.append(TargetEligibilityIssue.INSUFFICIENT_CALIBRATION_ROWS)
    if peer_count < minimum_peer_count:
        issues.append(TargetEligibilityIssue.INSUFFICIENT_PEERS)
    return TargetEligibility(tuple(issues))


def chronological_indices(
    timestamps: TimestampArray | RowIndexArray, split: SplitSettings
) -> ChronologicalIndices:
    if timestamps.ndim != 1 or timestamps.size < 4:
        raise ValueError("Chronological roles require at least four ordered rows")
    _validate_temporal_order(timestamps)
    order = np.argsort(timestamps, kind="stable")
    training_end = math.floor(timestamps.size * split.training_fraction)
    reference_end = math.floor(
        timestamps.size * (split.training_fraction + split.reference_fraction)
    )
    calibration_end = math.floor(
        timestamps.size
        * (split.training_fraction + split.reference_fraction + split.calibration_fraction)
    )
    if min(training_end, reference_end - training_end, calibration_end - reference_end) < 1:
        raise ValueError("Every benign chronology role must contain rows")
    return ChronologicalIndices(
        training=order[:training_end],
        reference=order[training_end:reference_end],
        calibration=order[reference_end:calibration_end],
        test=order[calibration_end:],
    )


def split_attack_indices(
    timestamps: TimestampArray | RowIndexArray,
    minimum_rows: MinimumAttackRows,
    rows_per_half_cap: AttackRowsPerHalfCap,
) -> AttackHalfIndices:
    if timestamps.ndim != 1:
        raise ValueError("Attack rows require one-dimensional temporal order")
    _validate_temporal_order(timestamps)
    if timestamps.size < minimum_rows:
        raise ValueError("Attack type does not meet the configured row minimum")
    order = np.argsort(timestamps, kind="stable")
    midpoint = timestamps.size // 2
    development_rows = order[:midpoint]
    held_out_rows = order[midpoint:]
    return AttackHalfIndices(
        peer_development=development_rows[:rows_per_half_cap],
        held_out=held_out_rows[:rows_per_half_cap],
    )


def _validate_temporal_order(
    temporal_order: TimestampArray | RowIndexArray,
) -> None:
    if np.issubdtype(temporal_order.dtype, np.datetime64):
        if np.isnat(temporal_order).any():
            raise ValueError("Temporal order timestamps must not contain NaT")
        return
    if np.issubdtype(temporal_order.dtype, np.integer):
        source_order = np.array(temporal_order, dtype=np.int64, copy=False)
        if np.any(source_order < 0):
            raise ValueError("Temporal source order must be non-negative")
        return
    raise ValueError("Rows require timestamps or integer source order")


def plan_client_split(
    client: DatasetClientId,
    study_stratum: StudyStratum,
    benign_temporal_order: TimestampArray | RowIndexArray,
    attack_captures: tuple[AttackCapture, ...],
    settings: SplitSettings,
) -> DatasetSplitPlan:
    benign_roles = chronological_indices(benign_temporal_order, settings)
    minimum_attack_rows = settings.minimum_attack_rows_for(study_stratum)
    attack_splits: list[AttackCaptureSplit] = []
    excluded_attack_types: list[AttackType] = []
    for capture in attack_captures:
        if capture.source_order.size < minimum_attack_rows:
            excluded_attack_types.append(capture.attack_type)
            continue
        attack_splits.append(
            AttackCaptureSplit(
                capture.attack_type,
                split_attack_indices(
                    capture.source_order,
                    minimum_attack_rows,
                    settings.attack_rows_per_half_cap,
                ),
            )
        )
    return DatasetSplitPlan(
        client,
        benign_roles,
        tuple(attack_splits),
        tuple(excluded_attack_types),
    )


def _hash_indices(hasher: LengthPrefixedHasher, role: SplitRole, indices: RowIndexArray) -> None:
    hasher.framed_label(role)
    hasher.count(indices.size)
    hasher.raw(ArtifactPayload(indices.astype("<i8", copy=False).tobytes(order="C")))


def digest_split_plan(plan: DatasetSplitPlan) -> ScientificDigest:
    hasher = LengthPrefixedHasher()
    hasher.framed_member(plan.client.dataset)
    hasher.framed_label(plan.client.name)
    hasher.count(len(plan.attack_splits))
    for role, indices in plan.benign_roles.by_role():
        _hash_indices(hasher, role, indices)
    for capture_split in plan.attack_splits:
        hasher.framed_label(capture_split.attack_type.identifier)
        _hash_indices(hasher, SplitRole.PEER_DEVELOPMENT, capture_split.indices.peer_development)
        _hash_indices(hasher, SplitRole.HELD_OUT, capture_split.indices.held_out)
    hasher.count(len(plan.excluded_attack_types))
    for attack_type in plan.excluded_attack_types:
        hasher.framed_label(SplitRole.EXCLUDED)
        hasher.framed_label(attack_type.identifier)
    return hasher.hexdigest()


@dataclass(frozen=True, slots=True)
class PooledStandardizer:
    mean: FloatArray
    scale: FloatArray

    def transform(self, features: FloatArray) -> FloatArray:
        if features.ndim != 2 or features.shape[0] == 0 or features.shape[1] != self.mean.size:
            raise ValueError("Feature matrix does not match the pooled feature schema")
        if not np.isfinite(features).all():
            raise ValueError("Feature values must be finite before standardization")
        transformed = (features - self.mean) / self.scale
        if not np.isfinite(transformed).all():
            raise ValueError("Standardized feature values must remain finite")
        return transformed


def fit_pooled_standardizer(training_features: Sequence[FloatArray]) -> PooledStandardizer:
    if not training_features:
        raise ValueError("Pooled standardization requires training rows")
    feature_count = training_features[0].shape[1] if training_features[0].ndim == 2 else 0
    if feature_count == 0:
        raise ValueError("Training features must be non-empty two-dimensional matrices")
    for client_features in training_features:
        if (
            client_features.ndim != 2
            or client_features.shape[0] == 0
            or client_features.shape[1] != feature_count
            or not np.isfinite(client_features).all()
        ):
            raise ValueError("Pooled training matrices must be finite and feature-aligned")
    pooled = np.concatenate(tuple(training_features), axis=0)
    mean = np.mean(pooled, axis=0)
    scale = np.std(pooled, axis=0)
    scale = np.where(scale > 0.0, scale, 1.0)
    return PooledStandardizer(mean.astype(np.float64), scale.astype(np.float64))
