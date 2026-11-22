import csv
import io
import tarfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import structlog

from pace.datasets.adapters.iot23 import (
    iot23_feature_names,
    iot23_source_files,
    load_iot23_archive,
    load_iot23_capture,
    load_iot23_clients,
    normalize_iot23_attack_type,
)
from pace.types import DatasetValidationError, Iot23Sampling, TargetCohortRole

FIELDS = (
    "ts",
    "uid",
    "id.orig_h",
    "id.orig_p",
    "id.resp_h",
    "id.resp_p",
    "proto",
    "service",
    "duration",
    "orig_bytes",
    "resp_bytes",
    "conn_state",
    "local_orig",
    "local_resp",
    "missed_bytes",
    "history",
    "orig_pkts",
    "orig_ip_bytes",
    "resp_pkts",
    "resp_ip_bytes",
    "label",
    "detailed-label",
)


ZEEK_MARKER = chr(35)


def write_iot23_log(path: Path, records: tuple[tuple[str, ...], ...]) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        stream.write(ZEEK_MARKER + "separator \\x09\n")
        stream.write(ZEEK_MARKER + "fields\t" + "\t".join(FIELDS) + "\n")
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        writer.writerows(records)
    return path.read_bytes()


def iot23_row(
    ts: str, label: str, family: str, duration: str = "1.5", orig_bytes: str = "20"
) -> tuple[str, ...]:
    return (
        ts,
        "C1",
        "192.168.1.10",
        "123",
        "8.8.8.8",
        "443",
        "tcp",
        "-",
        duration,
        orig_bytes,
        "40",
        "SF",
        "T",
        "F",
        "-",
        "ShADf",
        "2",
        "100",
        "3",
        "120",
        label,
        family,
    )


def test_iot23_adapter_maps_zeek_features_and_orders_rows_chronologically(
    tmp_path: Path,
) -> None:
    path = tmp_path / "CTU-IoT-Malware-Capture-7-1" / "conn.log.labeled"
    source = write_iot23_log(
        path,
        (
            iot23_row("3", "Malicious", "Okiru", "3", "30"),
            iot23_row("1", "Benign", "-", "1", "10"),
            iot23_row("2", "Malicious", "PartOfAHorizontalPortScan", "2", "20"),
            iot23_row("1", "Benign", "-", "1.1", "11"),
        ),
    )

    client = load_iot23_capture(path)

    assert client.client.name == "7-1"
    assert client.feature_names == iot23_feature_names()
    assert len(client.feature_names) == 36
    assert client.feature_names[:8] == (
        "duration",
        "orig_bytes",
        "resp_bytes",
        "missed_bytes",
        "orig_pkts",
        "orig_ip_bytes",
        "resp_pkts",
        "resp_ip_bytes",
    )
    assert np.allclose(
        client.benign_features[:, :2],
        [[np.log1p(1.0), np.log1p(10.0)], [np.log1p(1.1), np.log1p(11.0)]],
        rtol=1e-6,
    )
    assert np.isclose(client.benign_features[0, 10], np.log1p(1.0))  # state_SF
    assert np.isclose(client.benign_features[0, 21], np.log1p(1.0))  # proto_tcp
    assert np.isclose(client.benign_features[0, 24], np.log1p(1.0))  # orig_p_privileged
    assert client.benign_features[0, 28] == 0.0  # service_known
    assert np.isclose(client.benign_features[0, 29], np.log1p(5.0))
    assert np.allclose(
        client.benign_features[0, 30:36],
        [np.log1p(1.0), np.log1p(1.0), 0.0, 0.0, np.log1p(1.0), 0.0],
    )
    assert client.benign_source_population_rows is not None
    assert client.benign_source_population_rows == 2
    assert [capture.attack_type.identifier for capture in client.attack_captures] == [
        "HORIZONTAL_PORT_SCAN",
        "OKIRU",
    ]
    assert np.allclose(
        client.attack_captures[0].features[:, :2],
        [[np.log1p(2.0), np.log1p(20.0)]],
        rtol=1e-6,
    )
    assert [capture.source_order.tolist() for capture in client.attack_captures] == [[0], [0]]
    assert not client.benign_features.flags.writeable
    assert path.read_bytes() == source


def test_iot23_adapter_rejects_missing_schema_and_empty_labelled_population(
    tmp_path: Path,
) -> None:
    path = tmp_path / "Capture-1-1" / "conn.log.labeled"
    write_iot23_log(path, (iot23_row("1", "Benign", "-"),))
    with pytest.raises(DatasetValidationError, match="both benign and labelled malicious"):
        load_iot23_capture(path)


def test_iot23_sampling_is_bounded_reproducible_and_reports_population(
    tmp_path: Path,
) -> None:
    sampling = replace(Iot23Sampling.standard(), benign_cap=1)
    path = tmp_path / "CTU-IoT-Malware-Capture-7-1" / "conn.log.labeled"
    write_iot23_log(
        path,
        (
            iot23_row("1", "Benign", "-", "1", "10"),
            iot23_row("2", "Benign", "-", "2", "20"),
            iot23_row("3", "Benign", "-", "3", "30"),
            iot23_row("4", "Benign", "-", "4", "40"),
            iot23_row("5", "Malicious", "Okiru"),
        ),
    )

    first = load_iot23_capture(path, sampling)
    second = load_iot23_capture(path, sampling)

    assert first.benign_features.shape == (1, len(iot23_feature_names()))
    assert np.array_equal(first.benign_features, second.benign_features)
    assert first.benign_source_population_rows is not None
    assert first.benign_source_population_rows == 4
    attack_capture = first.attack_captures[0]
    assert attack_capture.source_population_rows is not None
    assert attack_capture.source_population_rows == 1


def test_iot23_reservoir_rows_keep_source_order_when_timestamps_tie(
    tmp_path: Path,
) -> None:
    path = tmp_path / "CTU-IoT-Malware-Capture-7-1" / "conn.log.labeled"
    records = tuple(
        iot23_row("1", "Benign", "-", orig_bytes=str(index * 10)) for index in range(14)
    ) + (iot23_row("2", "Malicious", "Okiru"),)
    write_iot23_log(path, records)

    client = load_iot23_capture(path, replace(Iot23Sampling.standard(), benign_cap=2))

    assert np.all(np.diff(client.benign_features[:, 1]) >= 0.0)


def test_iot23_historical_large_file_thinning_precedes_final_sampling(
    tmp_path: Path,
) -> None:
    sampling = replace(
        Iot23Sampling.standard(),
        chunk_rows=4,
        thin_group_min_rows=2,
        thin_fraction=0.5,
        large_file_bytes=0,
    )
    path = tmp_path / "CTU-IoT-Malware-Capture-7-1" / "conn.log.labeled"
    source = write_iot23_log(
        path,
        (
            iot23_row("1", "Benign", "-"),
            iot23_row("2", "Malicious", "Okiru", orig_bytes="10"),
            iot23_row("3", "Malicious", "Okiru", orig_bytes="20"),
            iot23_row("4", "Malicious", "Okiru", orig_bytes="30"),
            iot23_row("5", "Malicious", "Okiru", orig_bytes="40"),
            iot23_row("6", "Malicious", "Okiru", orig_bytes="50"),
        ),
    )

    client = load_iot23_capture(path, sampling)
    repeated_client = load_iot23_capture(path, sampling)

    attack_capture = client.attack_captures[0]
    assert attack_capture.source_population_rows is not None
    assert attack_capture.source_population_rows == 5
    assert attack_capture.features.shape == (3, len(iot23_feature_names()))
    assert np.array_equal(attack_capture.features, repeated_client.attack_captures[0].features)
    assert client.benign_source_population_rows is not None
    assert client.benign_source_population_rows == 1
    assert path.read_bytes() == source


def test_iot23_benign_only_peer_capture_is_retained_for_standardization(
    tmp_path: Path,
) -> None:
    path = tmp_path / "CTU-IoT-Malware-Capture-4-1" / "conn.log.labeled"
    write_iot23_log(path, (iot23_row("1", "Benign", "-"),))

    client = load_iot23_capture(path)

    assert client.client.name == "4-1"
    assert client.target_cohort_role is TargetCohortRole.PEER_ONLY
    assert client.benign_features.shape == (1, len(iot23_feature_names()))
    assert client.attack_captures == ()


def test_iot23_benign_only_capture_outside_pool_is_skipped_in_archive(
    tmp_path: Path,
) -> None:
    archive_path = tmp_path / "iot_23_datasets_small.tar.gz"
    content = io.BytesIO()
    with tarfile.open(fileobj=content, mode="w:gz") as archive:
        for name, records in (
            (
                "CTU-IoT-Malware-Capture-4-1/conn.log.labeled",
                (iot23_row("1", "Benign", "-"),),
            ),
            (
                "CTU-IoT-Malware-Capture-7-1/conn.log.labeled",
                (iot23_row("1", "Benign", "-"),),
            ),
            (
                "CTU-IoT-Malware-Capture-8-1/conn.log.labeled",
                (iot23_row("1", "Benign", "-"), iot23_row("2", "Malicious", "DDoS")),
            ),
        ):
            log = io.StringIO()
            log.write(ZEEK_MARKER + "separator \\x09\n")
            log.write(ZEEK_MARKER + "fields\t" + "\t".join(FIELDS) + "\n")
            writer = csv.writer(log, delimiter="\t", lineterminator="\n")
            writer.writerows(records)
            payload = log.getvalue().encode()
            member = tarfile.TarInfo(name)
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))
    archive_path.write_bytes(content.getvalue())

    clients = load_iot23_archive(archive_path)

    assert tuple(client.client.name for client in clients) == ("4-1", "8-1")


def test_iot23_adapter_expands_embedded_terminal_schema_fields(tmp_path: Path) -> None:
    path = tmp_path / "Capture-1-1" / "conn.log.labeled"
    path.parent.mkdir(parents=True)
    source_fields = (*FIELDS[:-2], "tunnel_parents   label   detailed-label")
    benign_row = (*iot23_row("1", "Benign", "-")[:-2], "(empty)   Benign   -")
    attack_row = (*iot23_row("2", "Malicious", "Okiru")[:-2], "(empty)   Malicious   Okiru")
    with path.open("w", encoding="utf-8") as stream:
        stream.write(ZEEK_MARKER + "fields\t" + "\t".join(source_fields) + "\n")
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        writer.writerow(benign_row)
        writer.writerow(attack_row)

    client = load_iot23_capture(path)

    assert client.benign_features.shape == (1, len(iot23_feature_names()))
    assert client.attack_captures[0].attack_type.identifier == "OKIRU"

    path.write_text(
        ZEEK_MARKER + "separator \\x09\n" + ZEEK_MARKER + "fields\tts\tlabel\n",
        encoding="utf-8",
    )
    with pytest.raises(DatasetValidationError, match="missing IoT-23 fields"):
        load_iot23_capture(path)


def test_iot23_source_discovery_only_finds_labelled_logs(tmp_path: Path) -> None:
    root = tmp_path / "IoT-23-v2"
    labelled = root / "Capture-1-1" / "conn.log.labeled"
    labelled.parent.mkdir(parents=True)
    labelled.touch()
    (labelled.parent / "conn.log").touch()

    assert iot23_source_files(tmp_path) == (labelled,)
    assert normalize_iot23_attack_type("C&C-HeartBeat").identifier == (
        "COMMAND_AND_CONTROL_HEARTBEAT"
    )


def test_iot23_loader_rejects_missing_source_and_malformed_record(tmp_path: Path) -> None:
    with pytest.raises(DatasetValidationError, match="No extracted IoT-23"):
        iot23_source_files(tmp_path)

    path = tmp_path / "Capture-2-1" / "conn.log.labeled"
    path.parent.mkdir(parents=True)
    path.write_text(
        ZEEK_MARKER + "fields\t" + "\t".join(FIELDS) + "\n" + "1\ttoo-short\n",
        encoding="utf-8",
    )
    with pytest.raises(DatasetValidationError, match="malformed Zeek record"):
        load_iot23_capture(path)


def test_iot23_archive_is_streamed_once_without_extracting_raw_members(tmp_path: Path) -> None:
    archive_path = tmp_path / "iot_23_datasets_small.tar.gz"
    content = io.BytesIO()
    with tarfile.open(fileobj=content, mode="w:gz") as archive:
        for name, family in (
            ("CTU-IoT-Malware-Capture-7-1/conn.log.labeled", "Okiru"),
            ("CTU-IoT-Malware-Capture-8-1/conn.log.labeled", "DDoS"),
            ("CTU-IoT-Malware-Capture-1-1/conn.log.labeled", "BenignOnly"),
        ):
            log = io.StringIO()
            log.write(ZEEK_MARKER + "separator \\x09\n")
            log.write(ZEEK_MARKER + "fields\t" + "\t".join(FIELDS) + "\n")
            writer = csv.writer(log, delimiter="\t", lineterminator="\n")
            writer.writerow(iot23_row("1", "Benign", "-"))
            if family != "BenignOnly":
                writer.writerow(iot23_row("2", "Malicious", family))
            payload = log.getvalue().encode()
            member = tarfile.TarInfo(name)
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))
    archive_path.write_bytes(content.getvalue())

    with structlog.testing.capture_logs() as logs:
        clients = load_iot23_archive(archive_path)

    assert tuple(client.client.name for client in clients) == ("7-1", "8-1")
    assert any(entry.get("client") == "1-1" and entry.get("usable") is False for entry in logs)
    assert all(client.benign_features.shape == (1, 36) for client in clients)
    assert not (tmp_path / "CTU-IoT-Malware-Capture-7-1").exists()
    assert iot23_source_files(tmp_path) == (archive_path,)


def test_iot23_archive_rejects_archives_without_labelled_logs(tmp_path: Path) -> None:
    archive_path = tmp_path / "iot_23_datasets_small.tar.gz"
    with tarfile.open(archive_path, mode="w:gz") as archive:
        member = tarfile.TarInfo("readme.txt")
        payload = b"No flow logs"
        member.size = len(payload)
        archive.addfile(member, io.BytesIO(payload))

    with pytest.raises(DatasetValidationError, match="No mixed benign-and-malicious"):
        load_iot23_archive(archive_path)


CAPTURE = "CTU-IoT-Malware-Capture-7-1"


def _log(tmp_path: Path, records: tuple[tuple[str, ...], ...]) -> Path:
    path = tmp_path / CAPTURE / "conn.log.labeled"
    write_iot23_log(path, records)
    return path


def _with(row: tuple[str, ...], field: str, value: str) -> tuple[str, ...]:
    values = list(row)
    values[FIELDS.index(field)] = value
    return tuple(values)


def test_attack_names_normalize_aliases_and_placeholders() -> None:
    assert normalize_iot23_attack_type("-").identifier == "UNSPECIFIED"
    assert normalize_iot23_attack_type("  ").identifier == "UNSPECIFIED"
    assert normalize_iot23_attack_type("C&C-HeartBeat").identifier == (
        "COMMAND_AND_CONTROL_HEARTBEAT"
    )
    assert normalize_iot23_attack_type("PartOfAHorizontalPortScan").identifier == (
        "HORIZONTAL_PORT_SCAN"
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("duration", "abc"),
        ("duration", "inf"),
        ("id.resp_p", "abc"),
        ("id.resp_p", "inf"),
        ("ts", "abc"),
        ("ts", "inf"),
    ],
)
def test_rows_with_unusable_numbers_are_dropped(tmp_path: Path, field: str, value: str) -> None:
    path = _log(
        tmp_path,
        (
            iot23_row("1", "Benign", "-"),
            iot23_row("2", "Malicious", "Okiru"),
            _with(iot23_row("3", "Malicious", "Okiru"), field, value),
        ),
    )
    assert load_iot23_capture(path).attack_captures[0].features.shape[0] == 1


def test_placeholder_numbers_and_missing_services_are_zeroed(tmp_path: Path) -> None:
    benign = _with(_with(iot23_row("1", "Benign", "-"), "orig_bytes", "(empty)"), "history", "")
    path = _log(tmp_path, (benign, iot23_row("2", "Malicious", "Okiru")))
    assert load_iot23_capture(path).benign_features.shape[0] == 1


def test_header_errors_are_rejected(tmp_path: Path) -> None:
    no_header = tmp_path / CAPTURE / "conn.log.labeled"
    no_header.parent.mkdir(parents=True)
    no_header.write_text(ZEEK_MARKER + "separator x\n", encoding="utf-8")
    with pytest.raises(DatasetValidationError, match="no Zeek fields header"):
        load_iot23_capture(no_header)
    missing = tmp_path / "other" / CAPTURE / "conn.log.labeled"
    missing.parent.mkdir(parents=True)
    missing.write_text(ZEEK_MARKER + "fields\tts\tlabel\n", encoding="utf-8")
    with pytest.raises(DatasetValidationError, match="missing IoT-23 fields"):
        load_iot23_capture(missing)


def test_capture_identity_falls_back_to_the_parent_directory(tmp_path: Path) -> None:
    path = tmp_path / "custom-name" / "conn.log.labeled"
    write_iot23_log(path, (iot23_row("1", "Benign", "-"), iot23_row("2", "Malicious", "Okiru")))
    assert load_iot23_capture(path).client.name == "custom-name"


def test_loading_a_directory_of_captures_and_discovery_failures(tmp_path: Path) -> None:
    with pytest.raises(DatasetValidationError, match="No extracted IoT-23"):
        iot23_source_files(tmp_path)
    _log(
        tmp_path / "IoT-23-v2",
        (iot23_row("1", "Benign", "-"), iot23_row("2", "Malicious", "Okiru")),
    )
    assert len(load_iot23_clients(tmp_path)) == 1


def test_unreadable_sources_and_archives_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(DatasetValidationError, match="Could not read IoT-23 source"):
        load_iot23_capture(tmp_path / "missing" / "conn.log.labeled")
    broken = tmp_path / "iot_23_datasets_small.tar.gz"
    broken.write_bytes(b"not an archive")
    with pytest.raises(DatasetValidationError, match="Could not read IoT-23 archive"):
        load_iot23_archive(broken)
