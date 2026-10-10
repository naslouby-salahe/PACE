import math

import numpy as np
import pytest

from pace.core.evidence import (
    anchor_detector_rows,
    fit_ecod_sum,
    fit_pair_cooccurrence,
    local_detector_evidence,
    local_ensemble_evidence,
    probability_tail_evidence,
    target_relative_evidence,
)
from pace.types import DetectorScoreMatrix, LocalDetectorKind, LocalDetectorSeed

BINS = 8
PAIRS = 150


def test_margin_tail_evidence_and_extrapolation_match_hand_calculation() -> None:
    reference = np.asarray([0.0, 1.0, 2.0, 3.0])
    query = np.asarray([2.0, 3.0, 4.0])
    evidence = target_relative_evidence(reference, query)
    expected = np.asarray(
        [
            -math.log(3.0 / 5.0),
            -math.log(2.0 / 5.0),
            math.log(5.0) + math.log1p(1.0 / 1.5),
        ]
    )
    np.testing.assert_allclose(evidence, expected)


def test_ties_are_included_in_the_empirical_upper_tail() -> None:
    evidence = target_relative_evidence(np.asarray([1.0, 2.0, 2.0, 3.0]), np.asarray([2.0]))
    assert np.isclose(evidence[0], -math.log(4.0 / 5.0))


def test_probability_tail_evidence_matches_the_published_formula_and_extrapolation() -> None:
    reference = np.asarray([0.1, 0.2, 0.4, 0.8])
    query = np.asarray([0.2, 0.8, 1.0])

    evidence = probability_tail_evidence(reference, query)

    expected = np.asarray(
        [
            -math.log(4.0 / 5.0),
            -math.log(2.0 / 5.0),
            math.log(5.0) + math.log(1.0 / 0.8),
        ]
    )
    np.testing.assert_allclose(evidence, expected)


@pytest.mark.parametrize(
    ("reference", "query"),
    [
        (np.asarray([0.0, 0.5]), np.asarray([-0.1])),
        (np.asarray([0.0, 1.1]), np.asarray([0.5])),
        (np.asarray([0.0, np.nan]), np.asarray([0.5])),
        (np.asarray([0.0, 0.5]), np.asarray([np.inf])),
        (np.asarray([[0.0, 0.5]]), np.asarray([0.5])),
    ],
)
def test_probability_tail_evidence_rejects_invalid_probabilities(
    reference: np.ndarray, query: np.ndarray
) -> None:
    with pytest.raises(ValueError):
        probability_tail_evidence(reference, query)


def test_constant_reference_uses_the_minimum_extrapolation_spread() -> None:
    evidence = target_relative_evidence(np.asarray([2.0, 2.0]), np.asarray([3.0]))
    assert np.isclose(evidence[0], math.log(3.0) + math.log1p(1_000_000.0))


@pytest.mark.parametrize(
    ("reference", "query"),
    [
        (np.asarray([], dtype=np.float64), np.asarray([0.0])),
        (np.asarray([0.0, np.nan]), np.asarray([0.0])),
        (np.asarray([0.0]), np.asarray([np.inf])),
        (np.asarray([[0.0]]), np.asarray([0.0])),
    ],
)
def test_invalid_evidence_inputs_fail(reference: np.ndarray, query: np.ndarray) -> None:
    with pytest.raises(ValueError):
        target_relative_evidence(reference, query)


def test_local_ensemble_averages_detector_specific_reference_evidence() -> None:
    references = DetectorScoreMatrix(np.tile(np.asarray([[0.0, 1.0, 2.0]]), (7, 1)))
    queries = DetectorScoreMatrix(np.tile(np.asarray([[2.0]]), (7, 1)))
    evidence = local_ensemble_evidence(references, queries)
    expected = (-math.log(2.0 / 4.0) - math.log(2.0 / 4.0)) / 2.0
    assert np.isclose(evidence[0], expected)


def test_local_ensemble_scores_require_complete_canonical_detector_set() -> None:
    with pytest.raises(ValueError):
        DetectorScoreMatrix(np.ones((1, 3), dtype=np.float64))


def _features(rows: int = 300, columns: int = 12, seed: int = 5) -> np.ndarray:
    generator = np.random.default_rng(seed)
    return np.asarray(generator.normal(size=(rows, columns)) * np.arange(1, columns + 1) * 0.3)


def _reference_ecod_sum(train: np.ndarray, z: np.ndarray) -> np.ndarray:
    ordered = np.sort(train, axis=0)
    n = train.shape[0]
    left = np.empty_like(z)
    right = np.empty_like(z)
    for j in range(z.shape[1]):
        rank = np.searchsorted(ordered[:, j], z[:, j], side="right")
        left[:, j] = -np.log((rank + 1) / (n + 1))
        rank_r = np.searchsorted(ordered[:, j], z[:, j], side="left")
        right[:, j] = -np.log((n - rank_r + 1) / (n + 1))
    return np.asarray(np.maximum(left, right).sum(axis=1))


def _reference_pair_mean(train: np.ndarray, z: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    features = train.shape[1]
    pairs = np.stack(
        [rng.integers(0, features, size=PAIRS), rng.integers(0, features, size=PAIRS)], axis=1
    )
    pairs = pairs[pairs[:, 0] != pairs[:, 1]] if features > 1 else pairs
    edges = [
        np.unique(np.quantile(train[:, j], np.linspace(0, 1, BINS + 1)[1:-1]))
        for j in range(features)
    ]

    def digitize(values: np.ndarray) -> np.ndarray:
        return np.stack([np.searchsorted(edges[j], values[:, j]) for j in range(features)], axis=1)

    codes = digitize(train)
    tables: list[np.ndarray] = []
    for a, b in pairs:
        width = BINS + 1
        counts = np.zeros(width * width)
        np.add.at(counts, codes[:, a] * width + codes[:, b], 1.0)
        tables.append(-np.log((counts + 0.5) / (train.shape[0] + 0.5 * width * width)))
    c = digitize(z)
    values = np.stack(
        [tables[i][c[:, a] * (BINS + 1) + c[:, b]] for i, (a, b) in enumerate(pairs)], axis=1
    )
    return np.asarray(values.mean(axis=1))


def test_ecod_sum_matches_reference_implementation_and_rises_in_tails() -> None:
    train, query = _features(), _features(200, seed=9)
    model = fit_ecod_sum(train)
    np.testing.assert_allclose(model.scores(query), _reference_ecod_sum(train, query), rtol=1e-12)
    centre = np.median(train, axis=0, keepdims=True)
    far = centre + 50.0
    assert model.scores(far)[0] > model.scores(centre)[0]


def test_pair_cooccurrence_matches_reference_implementation_for_the_same_seed() -> None:
    train, query = _features(), _features(200, seed=11)
    model = fit_pair_cooccurrence(train, PAIRS, BINS, 0.5, LocalDetectorSeed(7))
    np.testing.assert_allclose(
        model.scores(query), _reference_pair_mean(train, query, 7), rtol=1e-12
    )


def test_pair_cooccurrence_is_deterministic_and_seed_dependent() -> None:
    train, query = _features(), _features(100, seed=13)
    first = fit_pair_cooccurrence(train, PAIRS, BINS, 0.5, LocalDetectorSeed(1)).scores(query)
    again = fit_pair_cooccurrence(train, PAIRS, BINS, 0.5, LocalDetectorSeed(1)).scores(query)
    other = fit_pair_cooccurrence(train, PAIRS, BINS, 0.5, LocalDetectorSeed(2)).scores(query)
    np.testing.assert_array_equal(first, again)
    assert not np.array_equal(first, other)


def test_pair_cooccurrence_scores_unseen_value_pairs_higher() -> None:
    train = _features()
    model = fit_pair_cooccurrence(train, PAIRS, BINS, 0.5, LocalDetectorSeed(3))
    typical = train[:50]
    unseen = np.full((50, train.shape[1]), 100.0)
    assert model.scores(unseen).mean() > model.scores(typical).mean()


def test_pair_cooccurrence_rejects_mismatched_schema() -> None:
    model = fit_pair_cooccurrence(_features(), PAIRS, BINS, 0.5, LocalDetectorSeed(3))
    with pytest.raises(ValueError, match="fitted schema"):
        model.scores(_features(10, columns=5))


def test_local_bank_inputs_must_be_finite() -> None:
    broken = _features()
    broken[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        fit_ecod_sum(broken)


def test_anchor_excludes_only_the_autoencoder_row() -> None:
    rows = anchor_detector_rows()
    assert len(rows) == len(LocalDetectorKind) - 1
    assert LocalDetectorKind.AUTOENCODER.row not in rows


def test_detector_evidence_matrix_averages_to_the_legacy_ensemble_evidence() -> None:
    generator = np.random.default_rng(23)
    reference = DetectorScoreMatrix(generator.normal(size=(len(LocalDetectorKind), 80)) ** 2)
    query = DetectorScoreMatrix(generator.normal(size=(len(LocalDetectorKind), 60)) ** 2)
    matrix = local_detector_evidence(reference, query)
    assert matrix.shape == (len(LocalDetectorKind), 60)
    np.testing.assert_allclose(matrix.mean(axis=0), local_ensemble_evidence(reference, query))
