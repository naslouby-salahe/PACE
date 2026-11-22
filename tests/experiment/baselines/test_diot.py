import numpy as np
import pytest

from pace.config import DiotSettings
from pace.experiment.baselines.diot import (
    DiotDetector,
    SymbolAlphabet,
    context_windows,
    fine_tune_detector,
    fit_alphabet,
    fit_federated_profile,
)

SETTINGS = DiotSettings(
    symbol_count=3,
    symbol_fit_row_limit=100,
    unseen_quantile=0.99,
    lookback=3,
    hidden_width=8,
    hidden_layers=1,
    federated_rounds=2,
    local_epochs=1,
    fine_tune_epochs=30,
    batch_size=32,
    learning_rate=0.05,
    peer_row_limit=120,
)
CENTERS = np.asarray([[0.0, 0.0], [5.0, 5.0], [-5.0, 5.0]])
RADII = np.full(3, 1.0)


def _periodic_rows(length: int, jitter: int) -> np.ndarray:
    generator = np.random.default_rng(jitter)
    cycle = CENTERS[np.arange(length) % 3]
    return cycle + 0.1 * generator.normal(size=cycle.shape)


def test_alphabet_assigns_each_row_to_its_nearest_centre() -> None:
    alphabet = SymbolAlphabet(CENTERS, RADII)
    probes = CENTERS + 0.2
    np.testing.assert_array_equal(alphabet.symbols(probes), np.arange(3))
    assert alphabet.size == 4


def test_rows_beyond_their_cluster_radius_map_to_the_unseen_symbol() -> None:
    alphabet = SymbolAlphabet(CENTERS, RADII)
    far = np.asarray([[0.0, 0.0], [40.0, -40.0], [5.0, 5.5]])
    np.testing.assert_array_equal(alphabet.symbols(far), [0, 3, 1])


def test_learned_radii_cover_the_requested_share_of_the_fitting_rows() -> None:
    pooled = _periodic_rows(300, 8)
    alphabet = fit_alphabet(pooled, SETTINGS, 4)
    seen = alphabet.symbols(pooled) != 3
    assert np.mean(seen) >= 0.98
    assert np.all(alphabet.radii >= 0.0)


def test_alphabet_is_learned_from_pooled_rows_and_capped_by_their_count() -> None:
    pooled = _periodic_rows(60, 1)
    assert fit_alphabet(pooled, SETTINGS, 4).size == 4
    assert fit_alphabet(pooled[:2], SETTINGS, 4).size == 3


def test_context_windows_are_left_padded_with_the_pad_symbol() -> None:
    symbols = np.asarray([0, 1, 2, 1], dtype=np.int64)
    windows = context_windows(symbols, 3, 3)
    assert windows.shape == (4, 3)
    np.testing.assert_array_equal(windows[0], [3, 3, 3])
    np.testing.assert_array_equal(windows[3], [0, 1, 2])
    np.testing.assert_array_equal(windows[1], [3, 3, 0])


def test_federated_profile_requires_streams_longer_than_the_lookback() -> None:
    alphabet = SymbolAlphabet(CENTERS, RADII)
    streams = (_periodic_rows(2, 1),)
    with pytest.raises(ValueError, match="longer than the lookback"):
        fit_federated_profile(streams, alphabet, SETTINGS, 1)


def test_federated_profile_is_deterministic() -> None:
    alphabet = SymbolAlphabet(CENTERS, RADII)
    streams = (_periodic_rows(90, 1), _periodic_rows(80, 2))
    first = fit_federated_profile(streams, alphabet, SETTINGS, 3)
    second = fit_federated_profile(streams, alphabet, SETTINGS, 3)
    assert bool((first.parameters == second.parameters).all())


def test_detector_scores_a_broken_sequence_as_more_surprising_than_the_learned_cycle() -> None:
    alphabet = SymbolAlphabet(CENTERS, RADII)
    profile = fit_federated_profile(
        (_periodic_rows(120, 1), _periodic_rows(120, 2)), alphabet, SETTINGS, 3
    )
    detector = fine_tune_detector(profile, _periodic_rows(120, 3), SETTINGS, 3)
    normal = detector.surprisal(_periodic_rows(60, 4))
    shuffled = _periodic_rows(60, 5)[np.random.default_rng(0).permutation(60)]
    broken = detector.surprisal(shuffled)
    assert normal.shape == (60,)
    assert np.isfinite(normal).all()
    assert np.all(normal >= 0.0)
    assert np.mean(broken[3:]) > np.mean(normal[3:])


def test_fine_tuning_does_not_mutate_the_shared_federated_profile() -> None:
    alphabet = SymbolAlphabet(CENTERS, RADII)
    profile = fit_federated_profile((_periodic_rows(100, 1),), alphabet, SETTINGS, 2)
    before = profile.parameters.clone()
    fine_tune_detector(profile, _periodic_rows(100, 6), SETTINGS, 2)
    assert bool((profile.parameters == before).all())


def _trained_detector() -> DiotDetector:
    alphabet = SymbolAlphabet(CENTERS, RADII)
    profile = fit_federated_profile(
        (_periodic_rows(120, 1), _periodic_rows(120, 2)), alphabet, SETTINGS, 3
    )
    return fine_tune_detector(profile, _periodic_rows(120, 3), SETTINGS, 3)


def test_unseen_flows_score_above_the_probability_floor_and_grow_with_distance() -> None:
    detector = _trained_detector()
    floor = -np.log(1e-16)
    near = detector.surprisal(np.vstack((CENTERS[:1] + 0.1, CENTERS[:1] + 3.0)))
    far = detector.surprisal(np.vstack((CENTERS[:1] + 0.1, CENTERS[:1] + 30.0)))
    assert near[0] < floor
    assert near[1] >= floor
    assert far[1] > near[1]


def test_unseen_symbols_never_become_training_targets() -> None:
    alphabet = SymbolAlphabet(CENTERS, RADII)
    rows = np.vstack((_periodic_rows(60, 1), np.full((60, 2), 80.0)))
    profile = fit_federated_profile((rows,), alphabet, SETTINGS, 2)
    assert bool(np.isfinite(profile.parameters.numpy()).all())


def test_excess_is_zero_inside_a_cluster_and_positive_outside() -> None:
    alphabet = SymbolAlphabet(CENTERS, RADII)
    inside = alphabet.excess(CENTERS[:1] + 0.1)
    outside = alphabet.excess(CENTERS[:1] + 9.0)
    assert inside[0] == 0.0
    assert outside[0] > 0.0
