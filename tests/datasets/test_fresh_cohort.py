import pytest

from pace.datasets.common import iot23_fresh_cohort_role, target_cohort_role
from pace.datasets.registry import registered_strata
from pace.types import (
    ClientName,
    CommandTarget,
    DatasetClientId,
    DatasetName,
    Iot23Capture,
    PopulationScope,
    StudyStratum,
    TargetCohortRole,
)


def _client(name: str) -> DatasetClientId:
    return DatasetClientId(DatasetName.IOT_23, ClientName(name))


@pytest.mark.parametrize("name", ["17-1", "33-1", "39-1", "43-1", "52-1"])
def test_untouched_captures_are_targets_only_in_the_confirmation_stratum(name: str) -> None:
    assert target_cohort_role(_client(name)) is TargetCohortRole.OUTSIDE_PEER_POOL
    assert iot23_fresh_cohort_role(_client(name)) is TargetCohortRole.REPORTED_TARGET


def test_the_primary_population_becomes_the_frozen_peer_pool_of_the_confirmation_stratum() -> None:
    for capture in Iot23Capture:
        if capture.cohort_role is TargetCohortRole.OUTSIDE_PEER_POOL:
            continue
        assert capture.fresh_cohort_role is TargetCohortRole.PEER_ONLY
        assert iot23_fresh_cohort_role(_client(capture)) is TargetCohortRole.PEER_ONLY


def test_a_capture_too_small_to_calibrate_stays_outside_the_pool() -> None:
    assert iot23_fresh_cohort_role(_client("44-1")) is TargetCohortRole.OUTSIDE_PEER_POOL
    assert iot23_fresh_cohort_role(_client("never-seen")) is TargetCohortRole.OUTSIDE_PEER_POOL


def test_no_capture_is_a_reported_target_in_both_populations() -> None:
    for capture in Iot23Capture:
        both = (
            capture.cohort_role is TargetCohortRole.REPORTED_TARGET
            and capture.fresh_cohort_role is TargetCohortRole.REPORTED_TARGET
        )
        assert not both


def test_the_confirmation_stratum_is_registered_but_excluded_from_the_primary_population() -> None:
    assert StudyStratum.IOT_23_FRESH in registered_strata(DatasetName.IOT_23)
    assert StudyStratum.IOT_23_FRESH.dataset is DatasetName.IOT_23
    assert StudyStratum.IOT_23_FRESH not in CommandTarget.ALL.strata
    assert CommandTarget.IOT_23_FRESH.strata == (StudyStratum.IOT_23_FRESH,)
    assert not PopulationScope.PRIMARY.has_stratum(StudyStratum.IOT_23_FRESH)
    assert PopulationScope.CONFIRMATION.has_stratum(StudyStratum.IOT_23_FRESH)
    assert PopulationScope.PRIMARY.has_stratum(StudyStratum.IOT_23)
    assert not PopulationScope.CONFIRMATION.has_stratum(StudyStratum.IOT_23)
