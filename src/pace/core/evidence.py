from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from pace.types import (
    CountSmoothing,
    DetectorScoreMatrix,
    FeatureCount,
    FloatArray,
    FloatMatrix,
    IntMatrix,
    LocalDetectorKind,
    LocalDetectorSeed,
    ObservationCount,
    PairCount,
    QuantileBinCount,
    RowIndexArray,
)


def _validated_scores(scores: FloatArray) -> FloatArray:
    if scores.ndim != 1 or scores.size == 0:
        raise ValueError("Reference scores must be a non-empty one-dimensional array")
    if not np.isfinite(scores).all():
        raise ValueError("Scores must contain only finite values")
    return np.sort(scores)


def _empirical_evidence(
    sorted_reference: FloatArray, query: FloatArray, extrapolate: Callable[[FloatArray], FloatArray]
) -> FloatArray:
    left_indices = np.searchsorted(sorted_reference, query, side="left")
    greater_or_equal_counts = sorted_reference.size - left_indices
    evidence = -np.log((greater_or_equal_counts + 1) / (sorted_reference.size + 1))
    beyond_reference = query > sorted_reference[-1]
    evidence[beyond_reference] = np.log(sorted_reference.size + 1) + extrapolate(
        query[beyond_reference]
    )
    return evidence


def target_relative_evidence(reference: FloatArray, query: FloatArray) -> FloatArray:
    sorted_reference = _validated_scores(reference)
    if query.ndim != 1 or not np.isfinite(query).all():
        raise ValueError("query must be a finite one-dimensional array")
    reference_maximum = sorted_reference[-1]
    spread = max(reference_maximum - np.median(sorted_reference).item(), 1e-6)
    return _empirical_evidence(
        sorted_reference, query, lambda beyond: np.log1p((beyond - reference_maximum) / spread)
    )


def probability_tail_evidence(reference: FloatArray, query: FloatArray) -> FloatArray:
    sorted_reference = _validated_scores(reference)
    if (
        query.ndim != 1
        or not np.isfinite(query).all()
        or np.any(query < 0.0)
        or np.any(query > 1.0)
        or np.any(sorted_reference < 0.0)
        or np.any(sorted_reference > 1.0)
    ):
        raise ValueError("Probability scores must be finite values between zero and one")
    reference_maximum = max(sorted_reference[-1], np.finfo(np.float64).tiny)
    return _empirical_evidence(
        sorted_reference, query, lambda beyond: np.log(beyond / reference_maximum)
    )


def local_detector_evidence(
    reference_scores: DetectorScoreMatrix, query_scores: DetectorScoreMatrix
) -> FloatMatrix:
    return np.stack(
        [
            target_relative_evidence(reference_scores.values[index], query_scores.values[index])
            for index in range(len(LocalDetectorKind))
        ]
    )


def local_ensemble_evidence(
    reference_scores: DetectorScoreMatrix, query_scores: DetectorScoreMatrix
) -> FloatArray:
    return np.mean(local_detector_evidence(reference_scores, query_scores), axis=0)


def _validated(features: FloatMatrix) -> None:
    if features.ndim != 2 or features.shape[0] == 0 or not np.isfinite(features).all():
        raise ValueError("Local bank features must be a finite, non-empty row-by-feature matrix")


def anchor_detector_rows() -> RowIndexArray:
    return np.asarray(
        [kind.row for kind in LocalDetectorKind if kind.is_anchor_member], dtype=np.int64
    )


@dataclass(frozen=True, slots=True)
class EcodSumModel:
    ordered_columns: FloatMatrix

    def scores(self, features: FloatMatrix) -> FloatArray:
        _validated(features)
        training_rows = self.ordered_columns.shape[0]
        total = np.zeros(features.shape[0], dtype=np.float64)
        for column_index in range(features.shape[1]):
            ordered = self.ordered_columns[:, column_index]
            values = features[:, column_index]
            at_most = np.searchsorted(ordered, values, side="right")
            below = np.searchsorted(ordered, values, side="left")
            left_tail = -np.log((at_most + 1) / (training_rows + 1))
            right_tail = -np.log((training_rows - below + 1) / (training_rows + 1))
            total += np.maximum(left_tail, right_tail)
        return total


def fit_ecod_sum(training: FloatMatrix) -> EcodSumModel:
    _validated(training)
    return EcodSumModel(np.sort(training, axis=0))


@dataclass(frozen=True, slots=True)
class PairCooccurrenceModel:
    pairs: IntMatrix
    cut_points: tuple[FloatArray, ...]
    surprisal: tuple[FloatArray, ...]
    bins: QuantileBinCount

    def scores(self, features: FloatMatrix) -> FloatArray:
        _validated(features)
        if features.shape[1] != len(self.cut_points):
            raise ValueError("Pair co-occurrence features must match the fitted schema")
        codes = _quantile_codes(self.cut_points, features)
        width = self.bins + 1
        values = np.stack(
            [
                table[codes[:, first] * width + codes[:, second]]
                for table, (first, second) in zip(self.surprisal, self.pairs, strict=True)
            ],
            axis=1,
        )
        return np.asarray(np.mean(values, axis=1), dtype=np.float64)


def _quantile_codes(cut_points: tuple[FloatArray, ...], features: FloatMatrix) -> IntMatrix:
    return np.stack(
        [np.searchsorted(cuts, features[:, index]) for index, cuts in enumerate(cut_points)],
        axis=1,
    ).astype(np.int64)


def _random_pairs(
    feature_count: FeatureCount, pair_count: PairCount, seed: LocalDetectorSeed
) -> IntMatrix:
    generator = np.random.default_rng(seed)
    pairs = np.stack(
        [
            generator.integers(0, feature_count, size=pair_count),
            generator.integers(0, feature_count, size=pair_count),
        ],
        axis=1,
    ).astype(np.int64)
    return pairs[pairs[:, 0] != pairs[:, 1]] if feature_count > 1 else pairs


def fit_pair_cooccurrence(
    training: FloatMatrix,
    pair_count: PairCount,
    bins: QuantileBinCount,
    smoothing: CountSmoothing,
    seed: LocalDetectorSeed,
) -> PairCooccurrenceModel:
    _validated(training)
    pairs = _random_pairs(training.shape[1], pair_count, seed)
    levels = np.linspace(0.0, 1.0, bins + 1)[1:-1]
    cuts = tuple(
        np.asarray(np.unique(np.quantile(training[:, index], levels)), dtype=np.float64)
        for index in range(training.shape[1])
    )
    codes = _quantile_codes(cuts, training)
    width = bins + 1
    rows: ObservationCount = training.shape[0]
    tables: list[FloatArray] = []
    for first, second in pairs:
        counts = np.zeros(width * width, dtype=np.float64)
        np.add.at(counts, codes[:, first] * width + codes[:, second], 1.0)
        tables.append(-np.log((counts + smoothing) / (rows + smoothing * width * width)))
    return PairCooccurrenceModel(pairs, cuts, tuple(tables), bins)
