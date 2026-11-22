import ipaddress

import numpy as np
import pytest

from pace.datasets.common import (
    parse_decimal,
    parse_integer,
    parse_numeric_literal,
    signed_log_feature_transform,
    stable_timestamp_order,
    target_cohort_role,
)
from pace.types import (
    ChronologicalOrder,
    ClientName,
    DatasetClientId,
    DatasetName,
    DatasetValidationError,
    FeatureValue,
    IntegerBase,
    SourceCell,
    StudyStratum,
    TargetCohortRole,
    UnswDevice,
)

LAB_NETWORK = ipaddress.IPv4Network("192.168.1.0/24")


def test_non_unsw_recovered_target_cohort_has_forty_eight_typed_identities() -> None:
    targets = (
        *(
            DatasetClientId(DatasetName.IOT_23, ClientName(identity))
            for identity in (
                "1-1",
                "3-1",
                "7-1",
                "8-1",
                "9-1",
                "34-1",
                "35-1",
                "36-1",
                "48-1",
                "49-1",
                "60-1",
            )
        ),
        *(
            DatasetClientId(DatasetName.TON_IOT, ClientName(identity))
            for identity in (
                "192.168.1.1",
                "192.168.1.152",
                "192.168.1.180",
                "192.168.1.184",
                "192.168.1.190",
                "192.168.1.193",
                "192.168.1.194",
                "192.168.1.195",
                "192.168.1.79",
            )
        ),
        *(
            DatasetClientId(DatasetName.GOTHAM_2025, ClientName(identity))
            for identity in (
                "iotsim-air-quality-1",
                "iotsim-building-monitor-1",
                "iotsim-city-power-1",
                "iotsim-combined-cycle-1",
                "iotsim-domotic-monitor-1",
                "iotsim-ip-camera-museum-1",
                "iotsim-ip-camera-street-1",
            )
        ),
        *(
            DatasetClientId(DatasetName.CTU_13, ClientName(f"CTU13-scenario-{number:02d}"))
            for number in (*range(1, 7), *range(8, 14))
        ),
        *(
            DatasetClientId(DatasetName.N_BAIOT, ClientName(identity))
            for identity in (
                "Danmini_Doorbell",
                "Ecobee_Thermostat",
                "Ennio_Doorbell",
                "Philips_B120N10_Baby_Monitor",
                "Provision_PT_737E_Security_Camera",
                "Provision_PT_838_Security_Camera",
                "Samsung_SNH_1011_N_Webcam",
                "SimpleHome_XCS7_1002_WHT_Security_Camera",
                "SimpleHome_XCS7_1003_WHT_Security_Camera",
            )
        ),
    )

    assert len(targets) == 48
    assert len(set(targets)) == len(targets)
    assert all(
        target_cohort_role(target, LAB_NETWORK) is TargetCohortRole.REPORTED_TARGET
        for target in targets
    )
    assert (
        target_cohort_role(DatasetClientId(DatasetName.IOT_23, ClientName("17-1")))
        is TargetCohortRole.OUTSIDE_PEER_POOL
    )
    iot23_peer_only = ("4-1", "5-1", "20-1", "21-1", "42-1")
    assert all(
        target_cohort_role(DatasetClientId(DatasetName.IOT_23, ClientName(identity)))
        is TargetCohortRole.PEER_ONLY
        for identity in iot23_peer_only
    )
    assert (
        target_cohort_role(DatasetClientId(DatasetName.CTU_13, ClientName("CTU13-scenario-07")))
        is TargetCohortRole.PEER_ONLY
    )
    assert (
        target_cohort_role(
            DatasetClientId(DatasetName.GOTHAM_2025, ClientName("iotsim-combined-cycle-10"))
        )
        is TargetCohortRole.PEER_ONLY
    )


def test_unsw_devices_are_historically_selected_targets() -> None:
    UNSW_DEVICES = tuple(UnswDevice)

    assert len(UNSW_DEVICES) == 10
    assert {device.identifier for device in UNSW_DEVICES} == {
        ClientName(identity)
        for identity in (
            "00166cab6b88",
            "50c7bf005639",
            "70ee50183443",
            "ec1a5979f489",
            "ec1a59832811",
            "0017882b9a25",
            "44650d56ccd3",
            "74c63b29d71d",
            "d073d5018308",
            "f4f5d88f0a3c",
        )
    }
    assert {device.group for device in UNSW_DEVICES} == {
        StudyStratum.UNSW_IOT_ATTACK_FLOWS_G1,
        StudyStratum.UNSW_IOT_ATTACK_FLOWS_G2,
    }
    assert len(UNSW_DEVICES) + 48 == 58
    assert (
        target_cohort_role(
            DatasetClientId(DatasetName.UNSW_IOT_ATTACK_FLOWS, ClientName("unknown-device"))
        )
        is TargetCohortRole.PEER_ONLY
    )


def test_stable_timestamp_order_uses_source_row_for_equal_timestamps() -> None:
    timestamps = np.array([3.0, 3.0, 1.0, 3.0], dtype=np.float64)
    source_rows = np.array([2, 0, 9, 1], dtype=np.int64)

    order, chronology = stable_timestamp_order(timestamps, source_rows)

    np.testing.assert_array_equal(order, [2, 1, 3, 0])
    assert chronology is ChronologicalOrder.REORDERED


def test_stable_timestamp_order_keeps_already_ordered_ties() -> None:
    timestamps = np.array([1, 1, 2], dtype=np.int64)
    source_rows = np.array([2, 4, 8], dtype=np.int64)

    order, chronology = stable_timestamp_order(timestamps, source_rows)

    np.testing.assert_array_equal(order, [0, 1, 2])
    assert chronology is ChronologicalOrder.ALREADY_ORDERED


def test_stable_timestamp_order_rejects_misaligned_arrays() -> None:
    with pytest.raises(DatasetValidationError, match="must align"):
        stable_timestamp_order(np.array([1, 2], dtype=np.int64), np.array([0], dtype=np.int64))


def test_stable_timestamp_order_rejects_multidimensional_arrays() -> None:
    with pytest.raises(DatasetValidationError, match="one-dimensional"):
        stable_timestamp_order(np.array([[1, 2]], dtype=np.int64), np.array([0], dtype=np.int64))


def test_signed_log_transform_matches_float32_reference_without_mutating_input() -> None:
    features = np.array([[-1000.0, -1.5, -0.0, 0.0, 0.25, 1.5, 1000.0]], dtype=np.float64)
    original = features.copy()
    float32_values = features.astype(np.float32)
    expected = (
        (np.sign(float32_values) * np.log1p(np.abs(float32_values)))
        .astype(np.float32)
        .astype(np.float64)
    )

    actual = signed_log_feature_transform(features)

    assert actual.dtype == np.float64
    assert np.array_equal(actual, expected)
    assert np.array_equal(features, original)


def test_signed_log_transform_rejects_non_finite_source_values() -> None:
    with pytest.raises(DatasetValidationError, match="must be finite before"):
        signed_log_feature_transform(np.array([[np.inf]], dtype=np.float64))


def test_signed_log_transform_rejects_values_that_overflow_float32() -> None:
    overflowing = np.array([[np.finfo(np.float64).max]], dtype=np.float64)
    with (
        np.errstate(over="ignore", invalid="ignore"),
        pytest.raises(DatasetValidationError, match="must remain finite"),
    ):
        signed_log_feature_transform(overflowing)


def test_vectorized_reservoir_extension_matches_sequential_appends() -> None:
    from array import array

    from pace.datasets.common import extend_reservoirs
    from pace.types import FloatRowReservoir

    generator = np.random.default_rng(3)
    row_count = 600
    groups = generator.integers(0, 3, size=row_count)
    timestamps = generator.normal(size=row_count)
    source_rows = np.arange(row_count, dtype=np.int64)
    features = generator.normal(size=(row_count, 4)).astype(np.float32)
    capacities = (20, 50, 400)

    def buckets() -> list[FloatRowReservoir]:
        return [FloatRowReservoir(capacity, 4) for capacity in capacities]

    sequential, vectorized = buckets(), buckets()
    sequential_generator = np.random.default_rng(11)
    for index in range(row_count):
        bucket = sequential[int(groups[index])]
        position = bucket.sampling_population_rows
        bucket.sampling_population_rows += 1
        bucket.source_population_rows += 1
        if position < bucket.capacity:
            bucket.timestamps.append(float(timestamps[index]))
            bucket.source_rows.append(int(source_rows[index]))
            bucket.features.extend(features[index].tolist())
            continue
        slot = int(sequential_generator.integers(0, position + 1))
        if slot < bucket.capacity:
            bucket.timestamps[slot] = float(timestamps[index])
            bucket.source_rows[slot] = int(source_rows[index])
            bucket.features[slot * 4 : slot * 4 + 4] = array("f", features[index].tolist())
    vector_generator = np.random.default_rng(11)
    for start in range(0, row_count, 250):
        chunk = slice(start, start + 250)
        extend_reservoirs(
            vectorized,
            groups[chunk],
            timestamps[chunk],
            source_rows[chunk],
            features[chunk],
            vector_generator,
        )
    for left, right in zip(sequential, vectorized, strict=True):
        assert left.timestamps == right.timestamps
        assert left.source_rows == right.source_rows
        assert left.features == right.features
        assert left.source_population_rows == right.source_population_rows
        assert left.sampling_population_rows == right.sampling_population_rows


def test_record_blocks_skip_comments_and_blank_lines_and_respect_block_size() -> None:
    import io

    from pace.datasets.common import record_blocks

    payload = b"#header\r\n1,a\r\n\r\n2,b\n#note\n3,c\n4,d\n5,e"
    for chunk_bytes in (3, 7, 1 << 20):
        blocks = list(record_blocks(io.BytesIO(payload), 2, chunk_bytes=chunk_bytes))
        assert [[line.rstrip(b"\r") for line in block] for block in blocks] == [
            [b"1,a", b"2,b"],
            [b"3,c", b"4,d"],
            [b"5,e"],
        ]
    assert list(record_blocks(io.BytesIO(b""), 2)) == []
    assert list(record_blocks(io.BytesIO(b"1\n2\n"), 2)) == [[b"1", b"2"]]


def test_read_ahead_streams_bytes_in_order_and_propagates_errors() -> None:
    import io

    from pace.datasets.common import read_ahead

    payload = bytes(range(256)) * 400 + b"tail\nsecond line\n"
    with read_ahead(io.BytesIO(payload), chunk_bytes=1000, depth=2) as stream:
        assert stream.read() == payload

    class Failing(io.BytesIO):
        def read(self, size: int | None = -1) -> bytes:
            raise OSError(f"broken source {size}")

    with read_ahead(Failing(), chunk_bytes=10) as stream, pytest.raises(OSError, match="broken"):
        stream.read()


def test_decimal_text_is_parsed_at_the_boundary() -> None:
    assert parse_decimal(SourceCell("2.5")) == 2.5
    cell = SourceCell("not-a-number")
    with pytest.raises(ValueError, match="could not convert"):
        parse_decimal(cell)


def test_integer_text_is_parsed_at_the_boundary() -> None:
    assert parse_integer(SourceCell("42")) == 42
    assert parse_integer(SourceCell("0x1f"), IntegerBase.HEXADECIMAL) == 31
    assert parse_integer(SourceCell("0b11"), IntegerBase.AUTO) == 3
    cell = SourceCell("4.2")
    with pytest.raises(ValueError, match="invalid literal"):
        parse_integer(cell)


def test_integer_literals_widen_to_feature_values() -> None:
    assert parse_numeric_literal(SourceCell("0x10")) == FeatureValue(16.0)
    assert parse_numeric_literal(SourceCell(" 7 ")) == FeatureValue(7.0)
    assert parse_numeric_literal(SourceCell("-")) == FeatureValue(0.0)
    assert parse_numeric_literal(SourceCell("")) == FeatureValue(0.0)
    assert parse_numeric_literal(SourceCell("seven")) is None
