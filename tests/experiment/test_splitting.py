from pathlib import Path
from typing import cast

import numpy as np
import pytest

from pace.config import SplitSettings, load_configuration
from pace.experiment.splitting import (
    assess_target_eligibility,
    chronological_indices,
    digest_split_plan,
    fit_pooled_standardizer,
    plan_client_split,
    split_attack_indices,
)
from pace.types import (
    AttackCapture,
    AttackType,
    AttackTypeName,
    ClientName,
    DatasetClientId,
    DatasetName,
    RowIndexArray,
    StudyStratum,
    TargetEligibilityIssue,
)
from tests.support import REPOSITORY_ROOT


def _split_settings() -> SplitSettings:
    configuration = load_configuration(REPOSITORY_ROOT / "configs" / "default.yaml", Path.cwd())
    return configuration.splitting


def _timestamps(row_count: int) -> np.ndarray:
    epoch = np.datetime64("1970-01-01", "s")
    return epoch + np.arange(row_count).astype("timedelta64[s]")


def test_benign_rows_keep_chronological_40_20_20_20_roles() -> None:
    timestamps = _timestamps(1000)[::-1]
    roles = chronological_indices(timestamps, _split_settings())
    assert tuple(
        len(indices) for indices in (roles.training, roles.reference, roles.calibration, roles.test)
    ) == (400, 200, 200, 200)
    for role in (roles.training, roles.reference, roles.calibration, roles.test):
        assert np.all(np.diff(timestamps[role]) > np.timedelta64(0, "s"))
    assert timestamps[roles.training[-1]] < timestamps[roles.reference[0]]
    assert timestamps[roles.reference[-1]] < timestamps[roles.calibration[0]]
    assert timestamps[roles.calibration[-1]] < timestamps[roles.test[0]]


def test_source_row_order_uses_the_same_chronological_roles() -> None:
    source_order = np.arange(1000, dtype=np.int64)[::-1]
    roles = chronological_indices(source_order, _split_settings())
    assert tuple(
        len(indices) for indices in (roles.training, roles.reference, roles.calibration, roles.test)
    ) == (400, 200, 200, 200)
    np.testing.assert_array_equal(source_order[roles.training], np.arange(400))
    np.testing.assert_array_equal(source_order[roles.reference], np.arange(400, 600))
    np.testing.assert_array_equal(source_order[roles.calibration], np.arange(600, 800))
    np.testing.assert_array_equal(source_order[roles.test], np.arange(800, 1000))


def test_benign_splitting_rejects_invalid_or_too_small_chronologies() -> None:
    array = np.asarray(["NaT"], dtype="datetime64[s]")
    split_settings = _split_settings()
    with pytest.raises(ValueError):
        chronological_indices(array, split_settings)
    timestamps_input = _timestamps(3)
    split_settings = _split_settings()
    with pytest.raises(ValueError):
        chronological_indices(timestamps_input, split_settings)
    array = np.asarray([-1, 0, 1, 2], dtype=np.int64)
    split_settings = _split_settings()
    with pytest.raises(ValueError):
        chronological_indices(array, split_settings)
    typed_array = cast(RowIndexArray, np.asarray([0.0, 1.0, 2.0, 3.0]))
    split_settings = _split_settings()
    with pytest.raises(ValueError):
        chronological_indices(typed_array, split_settings)


def test_benign_splitting_rejects_empty_roles() -> None:
    settings = _split_settings().model_copy(
        update={
            "training_fraction": 0.75,
            "reference_fraction": 0.1,
            "calibration_fraction": 0.1,
            "test_fraction": 0.05,
        }
    )
    timestamps_input = _timestamps(4)
    with pytest.raises(ValueError, match="Every benign chronology role"):
        chronological_indices(timestamps_input, settings)


def test_attack_rows_are_split_by_time_before_each_half_is_capped() -> None:
    timestamps = _timestamps(700)[::-1]
    split = split_attack_indices(timestamps, 600, 250)
    assert len(split.peer_development) == 250
    assert len(split.held_out) == 250
    np.testing.assert_array_equal(timestamps[split.peer_development], _timestamps(250))
    np.testing.assert_array_equal(timestamps[split.held_out], _timestamps(600)[350:])


def test_attack_source_order_is_split_before_each_half_is_capped() -> None:
    source_order = np.arange(700, dtype=np.int64)[::-1]
    split = split_attack_indices(source_order, 600, 250)
    np.testing.assert_array_equal(source_order[split.peer_development], np.arange(250))
    np.testing.assert_array_equal(source_order[split.held_out], np.arange(350, 600))


def test_client_split_plan_applies_benign_roles_and_attack_eligibility() -> None:
    client = DatasetClientId(DatasetName.N_BAIOT, ClientName("camera"))
    benign_source_order = np.arange(1000, dtype=np.int64)[::-1]
    attack_sizes = (599, 600, 25_001)
    attack_captures = tuple(
        AttackCapture(
            AttackType(AttackTypeName(f"GAFGYT/{variation}")),
            Path(f"{variation}.csv"),
            np.zeros((row_count, 1), dtype=np.float64),
            np.arange(row_count, dtype=np.int64),
        )
        for variation, row_count in zip(
            ("COMBO", "JUNK", "SCAN"),
            attack_sizes,
            strict=True,
        )
    )

    plan = plan_client_split(
        client, StudyStratum.N_BAIOT, benign_source_order, attack_captures, _split_settings()
    )

    np.testing.assert_array_equal(benign_source_order[plan.benign_roles.training], np.arange(400))
    np.testing.assert_array_equal(
        benign_source_order[plan.benign_roles.reference], np.arange(400, 600)
    )
    np.testing.assert_array_equal(
        benign_source_order[plan.benign_roles.calibration], np.arange(600, 800)
    )
    np.testing.assert_array_equal(benign_source_order[plan.benign_roles.test], np.arange(800, 1000))
    assert plan.excluded_attack_types == (AttackType(AttackTypeName("GAFGYT/COMBO")),)
    assert tuple(split.attack_type.identifier.split("/")[1] for split in plan.attack_splits) == (
        "JUNK",
        "SCAN",
    )
    np.testing.assert_array_equal(
        attack_captures[1].source_order[plan.attack_splits[0].indices.peer_development],
        np.arange(300),
    )
    np.testing.assert_array_equal(
        attack_captures[1].source_order[plan.attack_splits[0].indices.held_out],
        np.arange(300, 600),
    )
    assert len(plan.attack_splits[1].indices.peer_development) == 10_000
    assert len(plan.attack_splits[1].indices.held_out) == 10_000
    changed_cap_settings = _split_settings().model_copy(update={"attack_rows_per_half_cap": 20})
    changed_plan = plan_client_split(
        client,
        StudyStratum.N_BAIOT,
        benign_source_order,
        attack_captures,
        changed_cap_settings,
    )
    assert digest_split_plan(plan) != digest_split_plan(changed_plan)


def test_attack_minimum_overrides_are_applied_by_study_stratum() -> None:
    settings = _split_settings()
    source_order = np.arange(1000, dtype=np.int64)
    attack = AttackCapture(
        AttackType(AttackTypeName("TCP_SYN_DEVICE")),
        Path("attack.csv"),
        np.zeros((30, 1), dtype=np.float64),
        np.arange(30, dtype=np.int64),
    )
    unsw_client = DatasetClientId(DatasetName.UNSW_IOT_ATTACK_FLOWS, ClientName("ec1a59832811"))
    nbaiot_client = DatasetClientId(DatasetName.N_BAIOT, ClientName("camera"))

    unsw_plan = plan_client_split(
        unsw_client,
        StudyStratum.UNSW_IOT_ATTACK_FLOWS_G1,
        source_order,
        (attack,),
        settings,
    )
    nbaiot_plan = plan_client_split(
        nbaiot_client, StudyStratum.N_BAIOT, source_order, (attack,), settings
    )

    assert len(unsw_plan.attack_splits) == 1
    assert nbaiot_plan.attack_splits == ()
    assert nbaiot_plan.excluded_attack_types == (attack.attack_type,)


def test_split_digest_is_stable_and_changes_when_role_boundaries_change() -> None:
    client = DatasetClientId(DatasetName.N_BAIOT, ClientName("camera"))
    benign_source_order = np.arange(1000, dtype=np.int64)
    settings = _split_settings()
    first_plan = plan_client_split(client, StudyStratum.N_BAIOT, benign_source_order, (), settings)
    repeated_plan = plan_client_split(
        client, StudyStratum.N_BAIOT, benign_source_order, (), settings
    )
    changed_settings = settings.model_copy(
        update={
            "training_fraction": 0.5,
            "reference_fraction": 0.1,
            "calibration_fraction": 0.2,
            "test_fraction": 0.2,
        }
    )
    changed_plan = plan_client_split(
        client, StudyStratum.N_BAIOT, benign_source_order, (), changed_settings
    )

    assert digest_split_plan(first_plan) == digest_split_plan(repeated_plan)
    assert digest_split_plan(first_plan) != digest_split_plan(changed_plan)


def test_attack_splitting_rejects_too_few_rows_and_invalid_timestamps() -> None:
    with pytest.raises(ValueError):
        split_attack_indices(
            np.arange(10, dtype="datetime64[s]"),
            20,
            10,
        )
    with pytest.raises(ValueError):
        split_attack_indices(
            np.asarray(["NaT", "2020-01-01"], dtype="datetime64[s]"),
            2,
            10,
        )
    typed_array_input = cast(RowIndexArray, np.asarray([1.0, 2.0]))
    with pytest.raises(ValueError):
        split_attack_indices(
            typed_array_input,
            2,
            10,
        )
    with pytest.raises(ValueError):
        split_attack_indices(
            np.asarray([[0, 1], [2, 3]], dtype=np.int64),
            2,
            10,
        )


def test_target_eligibility_reports_each_protocol_constraint() -> None:
    settings = _split_settings()
    ineligible = assess_target_eligibility(
        0,
        349,
        3,
        settings.minimum_calibration_rows,
        4,
    )
    assert ineligible.issues == (
        TargetEligibilityIssue.NO_ATTACK_TYPES,
        TargetEligibilityIssue.INSUFFICIENT_CALIBRATION_ROWS,
        TargetEligibilityIssue.INSUFFICIENT_PEERS,
    )
    eligible = assess_target_eligibility(
        1,
        350,
        4,
        settings.minimum_calibration_rows,
        4,
    )
    assert eligible.is_eligible


def test_standardizer_uses_only_pooled_training_rows_and_handles_constants() -> None:
    target_training = np.asarray([[0.0, 5.0], [2.0, 5.0]])
    peer_training = np.asarray([[4.0, 5.0], [6.0, 5.0]])
    standardizer = fit_pooled_standardizer((target_training, peer_training))
    np.testing.assert_allclose(standardizer.mean, np.asarray([3.0, 5.0]))
    np.testing.assert_allclose(standardizer.scale, np.asarray([np.sqrt(5.0), 1.0]))
    transformed = standardizer.transform(np.asarray([[3.0, 500.0]]))
    np.testing.assert_allclose(transformed, np.asarray([[0.0, 495.0]]))
    with pytest.raises(ValueError):
        standardizer.transform(np.empty((0, 2)))
    with pytest.raises(ValueError):
        standardizer.transform(np.asarray([[np.inf, 0.0]]))
    with pytest.raises(ValueError):
        standardizer.transform(np.ones((1, 1)))


def test_standardizer_rejects_empty_misaligned_and_non_finite_training() -> None:
    with pytest.raises(ValueError):
        fit_pooled_standardizer(())
    with pytest.raises(ValueError):
        fit_pooled_standardizer((np.empty((0, 2)),))
    with pytest.raises(ValueError):
        fit_pooled_standardizer((np.ones((2, 1)), np.ones((2, 2))))
    with pytest.raises(ValueError):
        fit_pooled_standardizer((np.asarray([[np.nan]]),))
