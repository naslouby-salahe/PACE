import numpy as np
import pytest

from pace.core.decision import evaluate_peer_union, fit_peer_union
from pace.core.pooled import fit_pooled_peer_union, rank_threshold
from pace.types import CalibrationMode


def _realized(
    mode: CalibrationMode, rows: int, budget: float, experts: int = 1, replicates: int = 6
) -> float:
    generator = np.random.default_rng(rows + int(budget * 1e5))
    rates: list[float] = []
    for _ in range(replicates):
        calibration = generator.exponential(size=(experts, rows))
        fresh = generator.exponential(size=(experts, 200_000))
        _, alerts = evaluate_peer_union(calibration, fresh, 250, budget, 64.0, 20, None, mode)
        rates.append(np.mean(alerts).item())
    return float(np.mean(rates))


def test_rank_threshold_uses_the_finite_sample_rank() -> None:
    ordered = np.arange(1.0, 101.0)
    assert rank_threshold(ordered, 0.05) == 96.0
    assert rank_threshold(ordered, 0.0001) == 100.0
    assert rank_threshold(ordered, 0.999999) == 1.0


def test_unresolvable_budgets_fall_back_to_the_calibration_maximum() -> None:
    evidence = np.random.default_rng(3).exponential(size=(1, 300))
    fitted = fit_pooled_peer_union(evidence, 0.001, 64.0, 20)
    assert fitted.thresholds[0] == evidence.max()


def test_pooled_thresholds_do_not_collapse_with_more_calibration_rows() -> None:
    small = _realized(CalibrationMode.POOLED_CONFORMAL, 2000, 0.0025)
    large = _realized(CalibrationMode.POOLED_CONFORMAL, 40000, 0.0025)
    assert 0.5 * 0.0025 <= large <= 1.5 * 0.0025
    assert small <= 3.0 * 0.0025


def test_block_maximum_calibration_saturates_on_large_calibration_sets() -> None:
    saturated = _realized(CalibrationMode.IN_SAMPLE, 40000, 0.0025)
    assert saturated < 0.1 * 0.0025


def test_pooled_calibration_tracks_the_budget_across_sample_sizes() -> None:
    for rows in (2000, 8000, 40000):
        for budget in (0.005, 0.0125):
            ratio = _realized(CalibrationMode.POOLED_CONFORMAL, rows, budget) / budget
            assert 0.5 <= ratio <= 1.5


def test_pooled_union_respects_the_union_budget_in_sample() -> None:
    evidence = np.random.default_rng(5).exponential(size=(8, 5000))
    fitted = fit_pooled_peer_union(evidence, 0.02, 64.0, 24)
    assert fitted.block_alert_rates[0] <= 0.02


def test_pooled_thresholds_are_monotone_in_the_budget() -> None:
    evidence = np.random.default_rng(7).exponential(size=(4, 4000))
    thresholds = [
        fit_pooled_peer_union(evidence, budget, 64.0, 24).thresholds for budget in (0.01, 0.03, 0.1)
    ]
    for tighter, looser in zip(thresholds, thresholds[1:], strict=False):
        assert all(a >= b for a, b in zip(tighter, looser, strict=True))


def test_pooled_calibration_is_deterministic_and_validates_shares() -> None:
    evidence = np.random.default_rng(9).exponential(size=(3, 800))
    first = fit_pooled_peer_union(evidence, 0.05, 64.0, 20)
    second = fit_pooled_peer_union(evidence, 0.05, 64.0, 20)
    assert first == second
    with pytest.raises(ValueError, match="one budget share"):
        fit_pooled_peer_union(evidence, 0.05, 64.0, 20, (0.01,))


def test_resolution_blocks_avoid_the_large_sample_saturation() -> None:
    saturated = _realized(CalibrationMode.IN_SAMPLE, 8000, 0.0025)
    resolved = _realized(CalibrationMode.RESOLUTION_BLOCKS, 8000, 0.0025)
    assert resolved > 3.0 * saturated


def test_resolution_blocks_equal_the_block_maximum_when_blocks_already_resolve() -> None:
    evidence = np.random.default_rng(11).exponential(size=(4, 3000))
    fresh = np.random.default_rng(12).exponential(size=(4, 500))
    plain, _ = evaluate_peer_union(evidence, fresh, 250, 0.0125, 64.0, 20)
    resolved, _ = evaluate_peer_union(
        evidence, fresh, 250, 0.0125, 64.0, 20, None, CalibrationMode.RESOLUTION_BLOCKS, 5, 2
    )
    assert plain == resolved == fit_peer_union(evidence, 250, 0.0125, 64.0, 20)
