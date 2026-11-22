import numpy as np

from pace.core.cross_fit import fit_cross_fitted_peer_union, fit_forward_chained_peer_union
from pace.core.decision import evaluate_peer_union
from pace.types import CalibrationMode


def _drifting(rows: int, slope: float, seed: int) -> np.ndarray:
    generator = np.random.default_rng(seed)
    scale = 1.0 + slope * np.linspace(0.0, 1.0, rows)
    return generator.exponential(size=(1, rows)) * scale


def test_forward_chaining_is_more_conservative_than_cross_fitting_under_upward_drift() -> None:
    evidence = _drifting(4000, 2.0, 1)
    standard = fit_cross_fitted_peer_union(evidence, 5, 0.01, 64.0, 20)
    forward = fit_forward_chained_peer_union(evidence, 5, 0.01, 64.0, 20)
    assert forward.joint_multiplier < standard.joint_multiplier
    assert forward.thresholds[0] > standard.thresholds[0]


def test_forward_chaining_matches_cross_fitting_without_drift() -> None:
    evidence = np.random.default_rng(2).exponential(size=(1, 8000))
    standard = fit_cross_fitted_peer_union(evidence, 5, 0.01, 64.0, 20)
    forward = fit_forward_chained_peer_union(evidence, 5, 0.01, 64.0, 20)
    assert np.isclose(forward.thresholds[0], standard.thresholds[0], rtol=0.25)


def test_forward_chained_mode_keeps_the_in_time_false_alert_rate_near_the_budget() -> None:
    rates: list[float] = []
    for seed in range(6):
        generator = np.random.default_rng(10 + seed)
        calibration = generator.exponential(size=(4, 6000))
        fresh = generator.exponential(size=(4, 60_000))
        _, alerts = evaluate_peer_union(
            calibration,
            fresh,
            250,
            0.01,
            64.0,
            20,
            None,
            CalibrationMode.TEMPORAL_CROSS_FITTED,
        )
        rates.append(np.mean(alerts).item())
    assert 0.5 * 0.01 <= float(np.mean(rates)) <= 1.5 * 0.01


def test_forward_chaining_reports_one_held_out_rate_per_later_fold() -> None:
    evidence = np.random.default_rng(3).exponential(size=(2, 3000))
    fitted = fit_forward_chained_peer_union(evidence, 5, 0.02, 64.0, 20)
    assert len(fitted.block_alert_rates) == 4
