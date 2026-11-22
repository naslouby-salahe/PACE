import numpy as np
import pytest

from pace.config import McdeSettings
from pace.experiment.baselines.mcde import (
    BenignModes,
    McdeDetector,
    PeerSupervision,
    fit_modes,
    fit_supervised,
    upper_tail,
)

SETTINGS = McdeSettings(
    supervised_budget_share=0.9,
    boosting_iterations=15,
    boosting_learning_rate=0.1,
    benign_modes=3,
    peer_row_limit=500,
    scale_floor=0.001,
    negligible_spread=0.000001,
    scale_floor_share=0.05,
    minimum_mode_rows=20,
    sparse_reference_rows=50,
)


def _benign(rows: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=(rows, 6))


def _attack(rows: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).normal(loc=3.0, size=(rows, 6))


def _detector(settings: McdeSettings = SETTINGS) -> McdeDetector:
    supervised = fit_supervised(PeerSupervision(_benign(600, 1), _attack(200, 2)), settings, 5)
    return McdeDetector(supervised, fit_modes(_benign(100, 3), settings, 5), settings)


def test_supervised_score_separates_peer_attacks_from_benign_traffic() -> None:
    supervised = fit_supervised(PeerSupervision(_benign(600, 1), _attack(200, 2)), SETTINGS, 5)
    benign = supervised.decision_function(_benign(300, 4))
    attacks = supervised.decision_function(_attack(300, 5))
    assert np.median(attacks) > np.percentile(benign, 90)


def test_supervised_fit_is_deterministic() -> None:
    supervision = PeerSupervision(_benign(400, 1), _attack(100, 2))
    probe = _attack(20, 9)
    first = fit_supervised(supervision, SETTINGS, 3).decision_function(probe)
    second = fit_supervised(supervision, SETTINGS, 3).decision_function(probe)
    np.testing.assert_array_equal(first, second)


def test_supervision_requires_both_classes() -> None:
    supervision = PeerSupervision(_benign(10, 1), np.empty((0, 6)))
    with pytest.raises(ValueError, match="benign and attack"):
        fit_supervised(supervision, SETTINGS, 1)


def test_supervised_sample_is_bounded_and_balanced_by_the_row_limit() -> None:
    supervision = PeerSupervision(_benign(5000, 1), _attack(40, 2))
    fit_supervised(supervision, SETTINGS, 2)


def test_mode_distance_is_the_root_mean_square_diagonal_mahalanobis_distance() -> None:
    centers = np.array([[0.0, 0.0, 0.0, 0.0], [10.0, 10.0, 10.0, 10.0]])
    modes = BenignModes(centers, np.array([[1.0, 2.0, 1.0, 2.0], [1.0, 1.0, 1.0, 1.0]]))
    probe = np.array([[2.0, 4.0, 2.0, 4.0], [10.0, 10.0, 10.0, 14.0]])
    assigned, distance = modes.assign(probe)
    np.testing.assert_array_equal(assigned, [0, 1])
    np.testing.assert_allclose(distance, [2.0, 2.0])


def test_local_scales_are_floored_and_small_modes_use_the_global_scale() -> None:
    generator = np.random.default_rng(3)
    wide = generator.normal(0.0, 5.0, size=(60, 3))
    narrow = generator.normal(100.0, 0.001, size=(60, 3))
    sparse = generator.normal(-100.0, 1.0, size=(4, 3))
    training = np.vstack((wide, narrow, sparse))
    settings = SETTINGS.model_copy(update={"benign_modes": 3})
    modes = fit_modes(training, settings, 2)
    order = np.argsort(modes.centers[:, 0])
    sparse_mode, wide_mode, narrow_mode = (modes.scales[index] for index in order)
    global_spread = np.std(training, axis=0)
    floor = max(0.001, 0.05 * float(np.median(global_spread)))
    np.testing.assert_allclose(sparse_mode, np.maximum(global_spread, floor))
    assert np.all(narrow_mode >= 0.05 * np.maximum(global_spread, floor) - 1e-12)
    assert np.all(narrow_mode < wide_mode)


def test_constant_coordinates_do_not_create_unbounded_standardised_distances() -> None:
    training = np.hstack((_benign(80, 4)[:, :3], np.zeros((80, 2))))
    modes = fit_modes(training, SETTINGS, 1)
    assert np.all(modes.scales >= SETTINGS.scale_floor)
    probe = training[:1].copy()
    probe[0, 4] = 1.0
    _, distance = modes.assign(probe)
    assert np.isfinite(distance).all()


def test_mode_count_is_capped_by_the_available_rows() -> None:
    modes = fit_modes(_benign(2, 7), SETTINGS, 1)
    assert modes.centers.shape[0] == 2


def test_upper_tail_counts_reference_scores_at_least_as_large_with_a_unit_offset() -> None:
    reference = np.array([1.0, 2.0, 2.0, 3.0])
    np.testing.assert_allclose(
        upper_tail(reference, np.array([0.0, 2.0, 2.5, 9.0])),
        [5.0 / 5.0, 4.0 / 5.0, 2.0 / 5.0, 1.0 / 5.0],
    )


def test_budget_normalised_fusion_alarms_exactly_when_a_branch_beats_its_budget_share() -> None:
    detector = _detector()
    reference = _benign(400, 11)
    query = np.vstack((_benign(150, 12), _attack(150, 13)))
    scores = detector.evidence(reference, query)
    reference_supervised = detector.supervised.decision_function(reference)
    query_supervised = detector.supervised.decision_function(query)
    supervised_tail = upper_tail(reference_supervised, query_supervised)
    share = SETTINGS.supervised_budget_share
    budget = 0.05
    from_scores = scores > -budget
    from_branches = supervised_tail < share * budget
    mode_tail = -scores * (1.0 - share)
    from_branches = from_branches | (np.where(scores < 0, mode_tail, 1.0) < (1 - share) * budget)
    np.testing.assert_array_equal(from_scores, from_branches)


def test_dual_evidence_ranks_attacks_above_benign_traffic() -> None:
    detector = _detector()
    reference = _benign(300, 11)
    benign = detector.evidence(reference, _benign(200, 12))
    attacks = detector.evidence(reference, _attack(200, 13))
    assert np.median(attacks) > np.percentile(benign, 95)


def test_a_sparse_mode_falls_back_to_the_pooled_benign_reference() -> None:
    reference = _benign(60, 21)
    query = _benign(40, 22)
    pooled = SETTINGS.model_copy(update={"sparse_reference_rows": 10_000})
    local = SETTINGS.model_copy(update={"sparse_reference_rows": 1})
    detector = _detector(pooled)
    local_detector = McdeDetector(detector.supervised, detector.modes, local)
    pooled_scores = detector.evidence(reference, query)
    local_scores = local_detector.evidence(reference, query)
    assert not np.array_equal(pooled_scores, local_scores)
    _, reference_distance = detector.modes.assign(reference)
    _, query_distance = detector.modes.assign(query)
    pooled_tail = upper_tail(reference_distance, query_distance)
    supervised_tail = upper_tail(
        detector.supervised.decision_function(reference),
        detector.supervised.decision_function(query),
    )
    expected = -np.minimum(
        supervised_tail / pooled.supervised_budget_share,
        pooled_tail / (1.0 - pooled.supervised_budget_share),
    )
    np.testing.assert_allclose(pooled_scores, expected)


def test_scores_are_deterministic_and_finite() -> None:
    first = _detector().evidence(_benign(100, 31), _attack(50, 32))
    second = _detector().evidence(_benign(100, 31), _attack(50, 32))
    np.testing.assert_array_equal(first, second)
    assert np.isfinite(first).all()
