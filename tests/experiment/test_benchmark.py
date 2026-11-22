import pytest

from pace.experiment.benchmark import benchmark_devices
from pace.experiment.runner import prepare_clients
from pace.types import ComputeMeasurement, ComputePhase, ComputeSubject, StudyStratum
from tests.support import campaign_configuration, campaign_devices


@pytest.fixture(scope="module")
def measurements() -> tuple[ComputeMeasurement, ...]:
    configuration = campaign_configuration()
    prepared = prepare_clients(campaign_devices(), configuration)
    return benchmark_devices(StudyStratum.N_BAIOT, prepared, configuration, 1, 1)


def test_every_method_reports_its_phases(measurements: tuple[ComputeMeasurement, ...]) -> None:
    covered = {(item.subject, item.phase) for item in measurements}
    assert covered == {
        (ComputeSubject.PACE, ComputePhase.OFFLINE_TRAINING),
        (ComputeSubject.PACE, ComputePhase.ONBOARDING_FIT),
        (ComputeSubject.PACE, ComputePhase.CALIBRATION),
        (ComputeSubject.PACE, ComputePhase.INFERENCE),
        (ComputeSubject.KITNET, ComputePhase.ONBOARDING_FIT),
        (ComputeSubject.KITNET, ComputePhase.CALIBRATION),
        (ComputeSubject.KITNET, ComputePhase.INFERENCE),
        (ComputeSubject.MCDE, ComputePhase.OFFLINE_TRAINING),
        (ComputeSubject.MCDE, ComputePhase.ONBOARDING_FIT),
        (ComputeSubject.MCDE, ComputePhase.INFERENCE),
        (ComputeSubject.DIOT, ComputePhase.OFFLINE_TRAINING),
        (ComputeSubject.DIOT, ComputePhase.ONBOARDING_FIT),
        (ComputeSubject.DIOT, ComputePhase.INFERENCE),
    }


def test_measurements_are_physical(measurements: tuple[ComputeMeasurement, ...]) -> None:
    for item in measurements:
        assert item.seconds >= 0.0
        assert item.seconds_sd >= 0.0
        assert item.peak_megabytes >= 0.0
        assert item.model_bytes > 0
        assert item.processed_rows > 0


def test_a_single_target_is_benchmarked_when_requested(
    measurements: tuple[ComputeMeasurement, ...],
) -> None:
    assert len({item.target for item in measurements}) == 1
