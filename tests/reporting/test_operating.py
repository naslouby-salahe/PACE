import numpy as np
import pytest

from pace.reporting.operating import (
    CurveReading,
    PairReading,
    TargetComparison,
    point_at,
    point_detection,
    read_curve,
)
from pace.types import (
    ClientName,
    DatasetClientId,
    DatasetName,
    MatchClass,
    MatchDirection,
    OperatingCurve,
    OperatingMethod,
    OperatingPoint,
    StudyStratum,
)


def _present(value: float | None) -> float:
    assert value is not None
    return value


TARGET = DatasetClientId(DatasetName.N_BAIOT, ClientName("client-0"))


def _curve(*points: tuple[float, float, float]) -> OperatingCurve:
    return OperatingCurve(
        method=OperatingMethod.PACE,
        points=tuple(
            OperatingPoint(nominal_budget=budget, false_alert_rate=rate, detection_rates=(tpr,))
            for budget, rate, tpr in points
        ),
    )


CURVE = _curve((0.01, 0.01, 0.2), (0.05, 0.05, 0.5), (0.1, 0.1, 0.8))


def test_a_reading_within_half_a_test_row_is_exact() -> None:
    reading = read_curve(CURVE, 0.0502, 0.001)
    assert reading.match is MatchClass.EXACT
    assert np.isclose(_present(reading.detection), 0.5)
    assert np.isclose(_present(reading.error), 0.0002)


def test_a_reading_inside_the_observed_range_is_interpolated() -> None:
    reading = read_curve(CURVE, 0.075, 0.001)
    assert reading.match is MatchClass.INTERPOLATED
    assert np.isclose(_present(reading.detection), 0.65)
    assert reading.error == 0.0


@pytest.mark.parametrize("rate", [0.0, 0.5])
def test_a_reading_outside_the_observed_range_is_unavailable_not_clamped(rate: float) -> None:
    reading = read_curve(CURVE, rate, 0.001)
    assert reading.match is MatchClass.UNAVAILABLE
    assert reading.detection is None
    assert reading.error is None


def test_the_envelope_never_lets_detection_decrease_with_the_false_alert_rate() -> None:
    curve = _curve((0.01, 0.01, 0.6), (0.05, 0.05, 0.4), (0.1, 0.1, 0.7))
    assert np.isclose(_present(read_curve(curve, 0.05, 0.001).detection), 0.6)
    assert np.isclose(_present(read_curve(curve, 0.08, 0.001).detection), 0.64, atol=0.02)


def test_tied_false_alert_rates_read_the_best_detection_among_them() -> None:
    curve = _curve((0.001, 0.0, 0.3), (0.002, 0.0, 0.55), (0.1, 0.1, 0.8))
    exact = read_curve(curve, 0.0, 0.001)
    assert exact.match is MatchClass.EXACT
    assert np.isclose(_present(exact.detection), 0.55)
    assert np.isclose(_present(read_curve(curve, 0.05, 0.001).detection), 0.675)


def test_points_expose_their_mean_detection_and_nominal_lookup() -> None:
    curve = _curve((0.01, 0.01, 0.3), (0.05, 0.04, 0.5))
    assert point_at(curve, 0.05).false_alert_rate == 0.04
    two = OperatingPoint(nominal_budget=0.1, false_alert_rate=0.1, detection_rates=(0.2, 0.6))
    assert np.isclose(point_detection(two), 0.4)


def _curve_reading(match: MatchClass, detection: float) -> CurveReading:
    if match is MatchClass.UNAVAILABLE:
        return CurveReading(match, None, None)
    return CurveReading(match, detection, 0.0)


def _reading(
    subject: float, reference: float, forward: MatchClass, backward: MatchClass
) -> PairReading:
    return PairReading(
        subject_detection=subject,
        subject_false_alert=0.06,
        reference_detection=reference,
        reference_false_alert=0.05,
        subject_at_reference=_curve_reading(forward, 0.5),
        reference_at_subject=_curve_reading(backward, 0.4),
    )


def test_matched_differences_follow_the_reading_direction() -> None:
    reading = _reading(0.7, 0.3, MatchClass.EXACT, MatchClass.INTERPOLATED)
    assert np.isclose(
        _present(reading.matched_difference(MatchDirection.SUBJECT_AT_REFERENCE)), 0.2
    )
    assert np.isclose(
        _present(reading.matched_difference(MatchDirection.REFERENCE_AT_SUBJECT)), 0.3
    )


def test_an_unavailable_reading_has_no_matched_difference() -> None:
    reading = _reading(0.7, 0.3, MatchClass.UNAVAILABLE, MatchClass.EXACT)
    assert reading.matched_difference(MatchDirection.SUBJECT_AT_REFERENCE) is None
    assert reading.matched_difference(MatchDirection.REFERENCE_AT_SUBJECT) is not None


def test_a_target_is_matched_only_when_every_seed_is_matched() -> None:
    complete = TargetComparison(
        TARGET,
        StudyStratum.N_BAIOT,
        (
            _reading(0.7, 0.3, MatchClass.EXACT, MatchClass.EXACT),
            _reading(0.5, 0.3, MatchClass.INTERPOLATED, MatchClass.EXACT),
        ),
    )
    partial = TargetComparison(
        TARGET,
        StudyStratum.N_BAIOT,
        (
            _reading(0.7, 0.3, MatchClass.EXACT, MatchClass.EXACT),
            _reading(0.5, 0.3, MatchClass.UNAVAILABLE, MatchClass.EXACT),
        ),
    )
    direction = MatchDirection.SUBJECT_AT_REFERENCE
    assert np.isclose(_present(complete.matched_difference(direction)), 0.2)
    assert complete.matching_error(direction) == 0.0
    assert partial.matched_difference(direction) is None
    assert partial.matching_error(direction) is None
    assert partial.classes(direction) == (MatchClass.EXACT, MatchClass.UNAVAILABLE)
    assert np.isclose(_present(partial.nominal_difference()), 0.3)
    assert np.isclose(_present(partial.false_alert_difference()), 0.01)
    assert np.isclose(_present(partial.reference_false_alert()), 0.05)
