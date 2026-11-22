import csv
from pathlib import Path
from typing import NoReturn

import numpy as np
import pytest

from pace.datasets.adapters.gotham import (
    gotham_feature_names,
    gotham_source_files,
    load_gotham_clients,
)
from pace.types import CorpusPath, DatasetValidationError, ReservoirCaps
from tests.support import load_gotham_capture

HEADER = (
    "frame.time",
    "frame.len",
    "frame.protocols",
    "eth.src",
    "eth.dst",
    "ip.dst",
    "ip.src",
    "ip.flags",
    "ip.ttl",
    "ip.proto",
    "ip.checksum",
    "ip.tos",
    "tcp.srcport",
    "tcp.dstport",
    "tcp.flags",
    "tcp.window_size_value",
    "tcp.window_size_scalefactor",
    "tcp.checksum",
    "tcp.options",
    "tcp.pdu.size",
    "udp.srcport",
    "udp.dstport",
    "label",
)


def _write_capture(path: Path, *, out_of_order: bool = False) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    timestamps = [
        "Jan 14, 2025 18:44:53.420758000 GMT",
        "Jan 14, 2025 18:44:54.000000001 GMT",
        "Jan 14, 2025 18:44:55.000000002 GMT",
        "Jan 14, 2025 18:44:56.000000003 GMT",
    ]
    if out_of_order:
        timestamps[2] = "Jan 14, 2025 18:44:52.000000000 GMT"
    rows = [
        (
            timestamps[0],
            "55",
            "eth:ip:udp",
            "aa",
            "bb",
            "192.0.2.1",
            "192.0.2.2",
            "0x02",
            "64",
            "17",
            "0x1111",
            "0",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "5683",
            "41000",
            "Benign",
        ),
        (
            timestamps[1],
            "78",
            "eth:ip:tcp",
            "aa",
            "bb",
            "192.0.2.1",
            "192.0.2.2",
            "0x02",
            "61",
            "6",
            "0x2222",
            "0",
            "40000",
            "23",
            "0x0018",
            "8192",
            "",
            "0x3333",
            "MSS",
            "24",
            "",
            "",
            "Mirai DoS",
        ),
        (
            timestamps[2],
            "60",
            "eth:ip:tcp",
            "aa",
            "bb",
            "192.0.2.1",
            "192.0.2.2",
            "0x02",
            "61",
            "6",
            "0x4444",
            "0",
            "40001",
            "23",
            "0x0002",
            "8192",
            "",
            "0x5555",
            "",
            "0",
            "",
            "",
            "Mirai DoS",
        ),
        (
            timestamps[3],
            "60",
            "eth:ip:tcp",
            "aa",
            "bb",
            "192.0.2.1",
            "192.0.2.2",
            "0x02",
            "61",
            "6",
            "0x6666",
            "0",
            "40002",
            "23",
            "0x0002",
            "8192",
            "",
            "0x7777",
            "",
            "0",
            "",
            "",
            "Unknown",
        ),
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(HEADER)
        writer.writerows(rows)
    return path.read_bytes()


def test_gotham_rejects_attack_label_without_alphanumeric_characters(tmp_path: Path) -> None:
    source = tmp_path / "processed" / "invalid-label.csv"
    _write_capture(source)
    with source.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.reader(stream))
    rows[2][-1] = "!!!"
    with source.open("w", newline="", encoding="utf-8") as stream:
        csv.writer(stream).writerows(rows)

    with pytest.raises(DatasetValidationError, match="empty packet label"):
        load_gotham_capture(source)


def test_gotham_adapter_maps_packets_without_identity_or_time_features(tmp_path: Path) -> None:
    source = tmp_path / "processed" / "iotsim-city-power-1.csv"
    original = _write_capture(source)

    client = load_gotham_capture(source)

    assert client.client.name == "iotsim-city-power-1"
    assert client.study_stratum.name == "GOTHAM_2025"
    assert client.feature_schema.name == "GOTHAM_TSHARK_ENGINEERED_V1"
    assert client.feature_names == gotham_feature_names()
    assert len(gotham_feature_names()) == 37
    assert client.benign_features.shape == (1, 37)
    assert np.isclose(client.benign_features[0, 4], np.log1p(2.0))
    assert client.benign_source_order.tolist() == [0]
    assert client.attack_captures[0].attack_type.identifier == "MIRAI_DOS"
    assert client.attack_captures[0].source_order.tolist() == [0, 1]
    assert np.isclose(client.attack_captures[0].features[0, 12], np.log1p(1.0))
    assert np.allclose(
        client.attack_captures[0].features[:, 0],
        [np.log1p(78.0), np.log1p(60.0)],
        rtol=1e-6,
    )
    assert client.attack_captures[0].features[0, -1] == 0.0
    assert np.isclose(client.attack_captures[0].features[1, -1], np.log1p(1.0))
    assert "ip_source" not in client.feature_names
    assert "timestamp" not in client.feature_names
    assert all(not capture.features.flags.writeable for capture in client.attack_captures)
    assert not client.benign_features.flags.writeable
    assert source.read_bytes() == original


def test_gotham_adapter_sorts_capture_timestamps_and_rejects_invalid_values(
    tmp_path: Path,
) -> None:
    source = tmp_path / "out-of-order.csv"
    _write_capture(source, out_of_order=True)
    client = load_gotham_capture(source)
    capture = client.attack_captures[0]
    assert np.allclose(capture.features[:, 0], [np.log1p(60.0), np.log1p(78.0)], rtol=1e-6)
    assert capture.source_order.tolist() == [0, 1]

    _write_capture(source)
    contents = source.read_text(encoding="utf-8")
    source.write_text(contents.replace(",55,", ",nan,"), encoding="utf-8")
    with pytest.raises(DatasetValidationError, match="Non-finite"):
        load_gotham_capture(source)


def test_gotham_aggregates_include_unknown_packets_and_sampling_stays_bounded(
    tmp_path: Path,
) -> None:
    caps = ReservoirCaps(benign=1, attack=40_000)
    source = tmp_path / "same-second.csv"
    _write_capture(source)
    contents = source.read_text(encoding="utf-8")
    contents = contents.replace("18:44:54.", "18:44:53.")
    contents = contents.replace("18:44:55.", "18:44:53.")
    contents = contents.replace("18:44:56.", "18:44:53.")
    contents = contents.replace("Mirai DoS", "Benign", 1)
    source.write_text(contents, encoding="utf-8")

    first = load_gotham_capture(source, caps)
    second = load_gotham_capture(source, caps)

    assert first.benign_features.shape == (1, len(gotham_feature_names()))
    assert first.benign_source_population_rows is not None
    assert first.benign_source_population_rows == 2
    assert np.array_equal(first.benign_features, second.benign_features)
    assert np.isclose(first.benign_features[0, 32], np.log1p(4.0))
    assert np.isclose(first.benign_features[0, 33], np.log1p(253.0))
    assert np.isclose(first.benign_features[0, 34], np.log1p(2.0))
    assert np.isclose(first.benign_features[0, 35], np.log1p(1.0))
    assert np.isclose(first.benign_features[0, 36], np.log1p(0.5))


def test_gotham_adapter_discovers_and_loads_capture_clients(tmp_path: Path) -> None:
    source = tmp_path / "Gotham2025" / "extracted" / "processed" / "iotsim-city-power-1.csv"
    _write_capture(source)

    clients = load_gotham_clients(tmp_path)

    assert gotham_source_files(tmp_path) == (source,)
    assert tuple(client.client.name for client in clients) == ("iotsim-city-power-1",)


def test_gotham_adapter_rejects_missing_schema_and_invalid_timestamp(tmp_path: Path) -> None:
    source = tmp_path / "missing.csv"
    source.write_text("frame.time,label\ninvalid,Benign\n", encoding="utf-8")
    with pytest.raises(DatasetValidationError, match="missing Gotham columns"):
        load_gotham_capture(source)

    source = tmp_path / "invalid-time.csv"
    _write_capture(source)
    contents = source.read_text(encoding="utf-8").replace(
        "Jan 14, 2025 18:44:53.420758000 GMT", "not-a-time"
    )
    source.write_text(contents, encoding="utf-8")
    with pytest.raises(DatasetValidationError, match="Invalid Gotham timestamp"):
        load_gotham_capture(source)


def test_gotham_adapter_rejects_capture_without_attack_rows(tmp_path: Path) -> None:
    source = tmp_path / "benign-only.csv"
    _write_capture(source)
    contents = source.read_text(encoding="utf-8").replace("Mirai DoS", "Unknown")
    source.write_text(contents, encoding="utf-8")

    with pytest.raises(DatasetValidationError, match="no clients with benign and attack"):
        load_gotham_capture(source)


STAMP = "Jan 14, 2025 18:44:{second:02d}.000000000 GMT"


def _row(second: int = 1, label: str = "Benign", **overrides: str) -> tuple[str, ...]:
    values = {
        "frame.time": STAMP.format(second=second),
        "frame.len": "60",
        "frame.protocols": "eth:ip:tcp",
        "ip.dst": "192.0.2.1",
        "ip.src": "192.0.2.2",
        "ip.flags": "0x02",
        "ip.ttl": "64",
        "ip.proto": "6",
        "ip.tos": "0",
        "tcp.srcport": "1234",
        "tcp.dstport": "80",
        "tcp.flags": "0x02",
        "tcp.window_size_value": "100",
        "tcp.pdu.size": "0",
        "label": label,
    }
    values.update(overrides)
    return tuple(values.get(column, "") for column in HEADER)


def _write(path: Path, rows: tuple[tuple[str, ...], ...], header: tuple[str, ...] = HEADER) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)
    return path


def _valid_rows() -> tuple[tuple[str, ...], ...]:
    return (_row(1, "Benign"), _row(2, "Mirai"))


def test_valid_rows_skip_blank_and_unlabelled_labels(tmp_path: Path) -> None:
    source = _write(
        tmp_path / "device.csv",
        (*_valid_rows(), _row(3, ""), _row(4, "unknown"), _row(5, "Normal")),
    )
    client = load_gotham_capture(source)
    assert client.benign_features.shape[0] == 2
    assert client.attack_captures[0].attack_type.identifier == "MIRAI"


def test_header_and_record_shape_errors_are_rejected(tmp_path: Path) -> None:
    source_input = _write(tmp_path / "a.csv", (), HEADER[:-1])
    with pytest.raises(DatasetValidationError, match="missing Gotham columns"):
        load_gotham_capture(source_input)
    source_input = _write(tmp_path / "b.csv", ((*_row(1), "extra"),))
    with pytest.raises(DatasetValidationError, match="Malformed Gotham record"):
        load_gotham_capture(source_input)


@pytest.mark.parametrize(
    "stamp",
    [
        "not a time",
        "Foo 14, 2025 18:44:01.000000000 GMT",
        "Feb 31, 2025 18:44:01.000000000 GMT",
    ],
)
def test_invalid_timestamps_are_rejected(tmp_path: Path, stamp: str) -> None:
    source = _write(tmp_path / "t.csv", (_row(1, **{"frame.time": stamp}),))
    with pytest.raises(DatasetValidationError, match="Invalid Gotham timestamp"):
        load_gotham_capture(source)


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"frame.len": "abc"}, "Invalid numeric frame.len"),
        ({"frame.len": "inf"}, "Non-finite frame.len"),
        ({"tcp.dstport": "abc"}, "Invalid numeric tcp.dstport"),
        ({"tcp.dstport": "inf"}, "Non-finite tcp.dstport"),
    ],
)
def test_invalid_numeric_fields_are_rejected(
    tmp_path: Path, override: dict[str, str], message: str
) -> None:
    source = _write(tmp_path / "n.csv", (_row(1, **override),))
    with pytest.raises(DatasetValidationError, match=message):
        load_gotham_capture(source)


def test_numeric_placeholders_hex_and_udp_ports_are_accepted(tmp_path: Path) -> None:
    rows = (
        _row(
            1, "Benign", **{"frame.len": "-", "ip.ttl": "", "ip.flags": "0xzz", "tcp.dstport": "-"}
        ),
        _row(
            2,
            "Mirai",
            **{"tcp.srcport": "", "tcp.dstport": "", "udp.dstport": "0x35", "frame.len": "0x40"},
        ),
    )
    assert load_gotham_capture(_write(tmp_path / "p.csv", rows)).attack_captures


def test_labels_must_name_an_attack_and_populations_must_be_complete(tmp_path: Path) -> None:
    source_input = _write(tmp_path / "e.csv", (_row(1, "Benign"), _row(2, "!!!")))
    with pytest.raises(DatasetValidationError, match="empty packet label"):
        load_gotham_capture(source_input)
    source_input = _write(tmp_path / "b.csv", (_row(1, "Benign"),))
    with pytest.raises(DatasetValidationError, match="no clients with benign and attack"):
        load_gotham_capture(source_input)
    source_input = _write(tmp_path / "a.csv", (_row(1, "Mirai"),))
    with pytest.raises(DatasetValidationError, match="no clients with benign and attack"):
        load_gotham_capture(source_input)


def test_reservoir_replacement_keeps_retained_rows_at_capacity(tmp_path: Path) -> None:
    rows = tuple(_row(second, "Benign") for second in range(1, 21)) + tuple(
        _row(second, "Mirai") for second in range(21, 41)
    )
    client = load_gotham_capture(
        _write(tmp_path / "r.csv", rows), ReservoirCaps(benign=2, attack=2)
    )
    assert client.benign_features.shape[0] == 2
    assert client.benign_source_population_rows == 20
    assert client.attack_captures[0].features.shape[0] == 2


def test_corpus_discovery_and_loading(tmp_path: Path) -> None:
    with pytest.raises(DatasetValidationError, match="processed data is unavailable"):
        gotham_source_files(tmp_path)
    processed = tmp_path / CorpusPath.GOTHAM_PROCESSED
    processed.mkdir(parents=True)
    with pytest.raises(DatasetValidationError, match="No Gotham processed CSV files"):
        gotham_source_files(tmp_path)
    _write(processed / "only_benign.csv", (_row(1, "Benign"),))
    with pytest.raises(DatasetValidationError, match="no clients with benign and attack"):
        load_gotham_clients(tmp_path)
    _write(processed / "good.csv", _valid_rows())
    assert len(load_gotham_clients(tmp_path)) == 1


def test_unreadable_capture_file_is_reported_as_validation_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / CorpusPath.GOTHAM_PROCESSED / "device.csv"
    _write_capture(source)

    def refuse(self: Path, mode: str = "r") -> NoReturn:
        raise PermissionError(f"{self} cannot be opened with mode {mode}")

    monkeypatch.setattr(Path, "open", refuse)
    with pytest.raises(DatasetValidationError, match="Could not read Gotham capture"):
        load_gotham_clients(tmp_path)
