import numpy as np

from pace.config import ScientificSettings
from pace.experiment.curves import curve_from_alerts, score_threshold_curve
from pace.experiment.inputs import TargetFrame
from pace.types import AttackType, AttackTypeName, CalibrationMode, OperatingMethod
from tests.support import campaign_configuration, campaign_devices, run_single_seed_campaign

GRID = (0.01, 0.05, 0.1, 0.2)
ATTACK = AttackType(AttackTypeName("ATTACK/ONE"))


def _settings() -> ScientificSettings:
    scientific = campaign_configuration().scientific
    return scientific.model_copy(update={"operating_grid": GRID, "calibration_block_size": 50})


def _scored_frame() -> tuple[TargetFrame, np.ndarray, np.ndarray]:
    generator = np.random.default_rng(3)
    benign = generator.normal(size=200)
    attack = generator.normal(loc=3.0, size=100)
    query = np.concatenate((benign[:100], attack))
    frame = TargetFrame(
        np.zeros((100, 1)),
        np.zeros((100, 1)),
        np.zeros((200, 1)),
        100,
        ((ATTACK, np.zeros((100, 1))),),
    )
    return frame, benign[100:], query


def test_score_threshold_curve_is_monotone_in_the_budget() -> None:
    frame, calibration, query = _scored_frame()
    curve = score_threshold_curve(OperatingMethod.KITNET, frame, calibration, query, _settings())
    rates = [point.false_alert_rate for point in curve.points]
    detections = [point.detection_rates[0] for point in curve.points]
    assert rates == sorted(rates)
    assert detections == sorted(detections)
    assert [point.nominal_budget for point in curve.points] == list(GRID)
    assert curve.method is OperatingMethod.KITNET


def test_score_threshold_curve_separates_a_shifted_attack() -> None:
    frame, calibration, query = _scored_frame()
    curve = score_threshold_curve(OperatingMethod.MCDE, frame, calibration, query, _settings())
    assert curve.points[-1].detection_rates[0] > 0.9
    assert curve.points[0].false_alert_rate <= curve.points[-1].false_alert_rate


def test_curve_from_alerts_records_one_point_per_grid_budget() -> None:
    frame, _, query = _scored_frame()
    curve = curve_from_alerts(
        OperatingMethod.PACE, frame, GRID, lambda budget: query > (1.0 / budget)
    )
    assert len(curve.points) == len(GRID)


def test_synthetic_campaign_records_every_curve_and_an_allocation_audit() -> None:
    configuration = campaign_configuration()
    results = run_single_seed_campaign(campaign_devices(), configuration)
    methods = {curve.method for curve in results[0].curves}
    expected = {
        OperatingMethod.PACE,
        OperatingMethod.PACE_IN_SAMPLE_CALIBRATION,
        OperatingMethod.PACE_RESOLUTION_AWARE_BLOCKS,
        OperatingMethod.PACE_POOLED_CONFORMAL_RANK,
        OperatingMethod.PACE_ANCHOR_COLLAPSE,
        OperatingMethod.PACE_FORWARD_CHAINING_CALIBRATION,
        OperatingMethod.PACE_WITHOUT_GAUSSIAN_EXPERT,
        OperatingMethod.PACE_FAMILY_BUDGETS,
        OperatingMethod.PACE_SECOND_ORDER_AGGREGATION,
        OperatingMethod.MEAN_ENSEMBLE_BLOCK_MAXIMUM,
        OperatingMethod.MEAN_ENSEMBLE_SAME_CALIBRATION,
        OperatingMethod.KITNET,
        OperatingMethod.DIOT,
        OperatingMethod.MCDE,
    }
    assert methods == expected
    for result in results:
        assert result.allocation is not None
        assert np.isclose(
            result.allocation.effective_nominal_budget, configuration.scientific.alpha.fraction
        )
        assert result.allocation.resolution_floor > 0.0
        for curve in result.curves:
            budgets = [point.nominal_budget for point in curve.points]
            assert tuple(budgets) == configuration.scientific.operating_grid


def test_baseline_thresholds_do_not_saturate_on_a_single_extreme_calibration_row() -> None:
    generator = np.random.default_rng(8)
    calibration = generator.exponential(size=20000)
    calibration[7] = 60.0
    benign = generator.exponential(size=2000)
    attack = generator.normal(loc=25.0, scale=2.0, size=500)
    query = np.concatenate((benign, attack))
    frame = TargetFrame(
        np.zeros((10, 1)),
        np.zeros((20000, 1)),
        np.zeros((2500, 1)),
        2000,
        ((ATTACK, np.zeros((500, 1))),),
    )
    settings = _settings().model_copy(update={"operating_grid": (0.005, 0.01, 0.05, 0.1)})
    saturated = score_threshold_curve(
        OperatingMethod.KITNET,
        frame,
        calibration,
        query,
        settings.model_copy(update={"calibration_mode": CalibrationMode.IN_SAMPLE}),
    )
    corrected = score_threshold_curve(
        OperatingMethod.KITNET,
        frame,
        calibration,
        query,
        settings.model_copy(update={"calibration_mode": CalibrationMode.CROSS_FITTED}),
    )
    assert saturated.points[0].detection_rates[0] == 0.0
    assert corrected.points[0].detection_rates[0] > 0.99


def test_matched_comparator_uses_the_same_calibration_as_pace_and_the_original_does_not() -> None:
    configuration = campaign_configuration()
    results = run_single_seed_campaign(campaign_devices(), configuration)
    for result in results:
        curves = {curve.method: curve for curve in result.curves}
        matched = curves[OperatingMethod.MEAN_ENSEMBLE_SAME_CALIBRATION]
        original = curves[OperatingMethod.MEAN_ENSEMBLE_BLOCK_MAXIMUM]
        assert [point.nominal_budget for point in matched.points] == [
            point.nominal_budget for point in original.points
        ]
        rates = [point.false_alert_rate for point in matched.points]
        assert all(0.0 <= rate <= 1.0 for rate in rates)
