import csv
from math import isclose, log1p
from pathlib import Path

import pytest

from pace.datasets.adapters.unsw import (
    load_unsw_clients,
    load_unsw_device,
    unsw_source_files,
)
from pace.types import (
    DatasetValidationError,
    SourceCell,
    StudyStratum,
    TargetCohortRole,
    UnswAttackFamily,
    UnswColumn,
    UnswDevice,
)

UNSW_DEVICES = tuple(UnswDevice)
FEATURE_NAMES = tuple(
    column.feature for column in UnswColumn.counters(StudyStratum.UNSW_IOT_ATTACK_FLOWS_G1)
)


def _header() -> tuple[str, ...]:
    return (
        "Timestamp",
        "FromLocalArpPortAllPacket",
        "FromLocalArpPortAllByte",
        "FromLocalUdpPort53IP192.168.1.1Packet",
        "FromLocalUdpPort53IP192.168.1.1Byte",
        "FromLocalUdpPort67IP192.168.1.1Packet",
        "FromLocalUdpPort67IP192.168.1.1Byte",
        "ToLocalArpPortAllPacket",
        "ToLocalArpPortAllByte",
        "ToLocalUdpPort53IP192.168.1.1Packet",
        "ToLocalUdpPort53IP192.168.1.1Byte",
        "ToLocalUdpPort67IP192.168.1.1Packet",
        "ToLocalUdpPort67IP192.168.1.1Byte",
        "ToLocalUdpPort67IP255.255.255.255/32Packet",
        "ToLocalUdpPort67IP255.255.255.255/32Byte",
        "ToLocal0x888ePortAllPacket",
        "ToLocal0x888ePortAllByte",
        " NoOfFlows",
        "CaptureSpecificTcpPort80Packet",
    )


def _fixture_data(data_root: Path, device_id: str) -> tuple[Path, ...]:
    dataset_root = data_root / "UNSW-IoT-Attack-Flows"
    flow_directory = dataset_root / "flowdata" / "flowdata"
    annotation_directory = dataset_root / "annotations" / "annotations"
    flow_directory.mkdir(parents=True)
    annotation_directory.mkdir(parents=True)
    flow_file = flow_directory / f"{device_id}_flowstats.csv"
    with flow_file.open("w", newline="", encoding="utf-8") as target:
        writer = csv.writer(target)
        header = _header()
        positions = {name.strip(): index for index, name in enumerate(header)}
        writer.writerow(header)
        for timestamp, overrides in (
            (90_000, ()),
            (
                150_000,
                (
                    ("FromLocalUdpPort53IP192.168.1.1Packet", "3"),
                    ("FromLocalUdpPort53IP192.168.1.1Byte", "30"),
                    ("ToLocalUdpPort67IP192.168.1.1Packet", "7"),
                    ("ToLocalUdpPort67IP255.255.255.255/32Packet", "8"),
                    ("ToLocal0x888ePortAllPacket", "-4"),
                    (" NoOfFlows", "9"),
                    ("CaptureSpecificTcpPort80Packet", "999"),
                ),
            ),
            (210_000, ()),
        ):
            row = ["0"] * len(header)
            row[positions["Timestamp"]] = str(timestamp)
            for name, value in overrides:
                row[positions[name.strip()]] = value
            writer.writerow(row)
    annotation_file = annotation_directory / f"{device_id}.csv"
    annotation_file.write_text("100,200,SomeFeature,TcpSynDevice10L2D\n", encoding="utf-8")
    return flow_file, annotation_file


def test_unsw_adapter_maps_common_counters_and_uses_annotation_windows(tmp_path: Path) -> None:
    device = UNSW_DEVICES[0]
    source_files = _fixture_data(tmp_path, device.identifier)
    source_contents = tuple(path.read_bytes() for path in source_files)

    client = load_unsw_device(tmp_path, device)

    assert client.feature_names == FEATURE_NAMES
    assert client.feature_names[-1] == "flow_count"
    assert client.client.name == device.identifier
    assert client.study_stratum is StudyStratum.UNSW_IOT_ATTACK_FLOWS_G1
    assert client.target_cohort_role is TargetCohortRole.REPORTED_TARGET
    assert client.benign_source_order.tolist() == [0, 2]
    assert client.benign_features[:, 0].tolist() == [0.0, 0.0]
    capture = client.attack_captures[0]
    assert capture.attack_type.identifier == "TCP_SYN_DEVICE"
    assert capture.source_order.tolist() == [1]
    assert capture.features.shape == (1, len(FEATURE_NAMES))
    assert len(FEATURE_NAMES) == 15
    assert isclose(capture.features[0, 0], log1p(3), rel_tol=1e-6)
    assert isclose(capture.features[0, 6], log1p(8), rel_tol=1e-6)
    assert isclose(capture.features[0, 10], -log1p(4), rel_tol=1e-6)
    assert isclose(capture.features[0, 14], log1p(9), rel_tol=1e-6)
    assert tuple(path.read_bytes() for path in source_files) == source_contents


def test_unsw_g2_uses_its_frozen_common_columns(tmp_path: Path) -> None:
    device = next(
        item for item in UNSW_DEVICES if item.group is StudyStratum.UNSW_IOT_ATTACK_FLOWS_G2
    )
    _fixture_data(tmp_path, device.identifier)

    client = load_unsw_device(tmp_path, device)

    assert client.feature_names == tuple(
        column.feature for column in UnswColumn.counters(StudyStratum.UNSW_IOT_ATTACK_FLOWS_G2)
    )
    assert len(client.feature_names) == 13
    assert isclose(client.attack_captures[0].features[0, 6], log1p(7), rel_tol=1e-6)


def test_unsw_family_normalization_collapses_load_variants() -> None:
    assert (
        UnswAttackFamily.from_annotation(SourceCell("TcpSynDevice1W2D"))
        is UnswAttackFamily.TCP_SYN_DEVICE
    )
    assert (
        UnswAttackFamily.from_annotation(SourceCell("TcpSynDevice100L2D"))
        is UnswAttackFamily.TCP_SYN_DEVICE
    )
    assert (
        UnswAttackFamily.from_annotation(SourceCell("Ssdp10L2D2W"))
        is UnswAttackFamily.SSDP_REFLECTION
    )
    cell = SourceCell("Unknown")
    with pytest.raises(DatasetValidationError, match="Unknown UNSW attack annotation"):
        UnswAttackFamily.from_annotation(cell)


def test_unsw_manifest_has_two_five_device_strata() -> None:
    assert (
        sum(device.group is StudyStratum.UNSW_IOT_ATTACK_FLOWS_G1 for device in UNSW_DEVICES) == 5
    )
    assert (
        sum(device.group is StudyStratum.UNSW_IOT_ATTACK_FLOWS_G2 for device in UNSW_DEVICES) == 5
    )
    assert len({device.identifier for device in UNSW_DEVICES}) == 10


def test_unsw_loader_rejects_nonchronological_flow_rows(tmp_path: Path) -> None:
    device = UNSW_DEVICES[0]
    flow_file, _ = _fixture_data(tmp_path, device.identifier)
    with flow_file.open("w", newline="", encoding="utf-8") as target:
        writer = csv.writer(target)
        writer.writerow(_header())
        writer.writerow((200_000, *(0 for _ in _header()[1:-1]), 0))
        writer.writerow((100_000, *(0 for _ in _header()[1:-1]), 0))
    with pytest.raises(DatasetValidationError, match="chronologically ordered"):
        load_unsw_device(tmp_path, device)


def test_unsw_loader_rejects_overlapping_families_and_nonfinite_counters(
    tmp_path: Path,
) -> None:
    device = UNSW_DEVICES[0]
    flow_file, annotation_file = _fixture_data(tmp_path, device.identifier)
    annotation_file.write_text(
        "100,200,Feature,TcpSynDevice10L2D\n150,160,Feature,Ssdp10L2D2W\n",
        encoding="utf-8",
    )
    with pytest.raises(DatasetValidationError, match="overlapping attack types"):
        load_unsw_device(tmp_path, device)

    annotation_file.write_text("100,200,Feature,TcpSynDevice10L2D\n", encoding="utf-8")
    contents = flow_file.read_text(encoding="utf-8").replace(",3,30,", ",3,nan,", 1)
    flow_file.write_text(contents, encoding="utf-8")
    with pytest.raises(DatasetValidationError, match="non-finite feature values"):
        load_unsw_device(tmp_path, device)


def test_unsw_sources_are_dataset_scoped_and_group_specific(tmp_path: Path) -> None:
    expected = _fixture_data(tmp_path, UNSW_DEVICES[0].identifier)
    with pytest.raises(DatasetValidationError, match="incomplete"):
        unsw_source_files(tmp_path, StudyStratum.UNSW_IOT_ATTACK_FLOWS_G1)
    assert expected[0].is_file()


DEVICE = UNSW_DEVICES[0]


COLUMNS = (
    "Timestamp",
    *UnswColumn.counters(StudyStratum.UNSW_IOT_ATTACK_FLOWS_G1),
)


def _write_flows(
    data_root: Path,
    header: tuple[str, ...] = COLUMNS,
    rows: tuple[tuple[object, ...], ...] | None = None,
) -> Path:
    flow_directory = data_root / "UNSW-IoT-Attack-Flows" / "flowdata" / "flowdata"
    flow_directory.mkdir(parents=True, exist_ok=True)
    flow_file = flow_directory / f"{DEVICE.identifier}_flowstats.csv"
    default_rows = tuple(
        (timestamp, *(1 for _ in header[1:])) for timestamp in (90_000, 150_000, 210_000)
    )
    with flow_file.open("w", newline="", encoding="utf-8") as target:
        writer = csv.writer(target)
        writer.writerow(header)
        writer.writerows(default_rows if rows is None else rows)
    return flow_file


def _write_annotations(data_root: Path, text: str) -> Path:
    annotation_directory = data_root / "UNSW-IoT-Attack-Flows" / "annotations" / "annotations"
    annotation_directory.mkdir(parents=True, exist_ok=True)
    annotation_file = annotation_directory / f"{DEVICE.identifier}.csv"
    annotation_file.write_text(text, encoding="utf-8")
    return annotation_file


VALID_ANNOTATION = "100,200,Feature,TcpSynDevice10L2D\n"


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("1,2,Feature\n", "fewer than four annotation fields"),
        ("9,2,Feature,TcpSynDevice10L2D\n", "invalid attack interval"),
        ("1,2,Feature,\n", "invalid attack interval"),
        ("x,2,Feature,TcpSynDevice10L2D\n", "Could not read attack annotations"),
        ("", "No attack intervals"),
    ],
)
def test_annotation_errors_are_rejected(tmp_path: Path, text: str, message: str) -> None:
    _write_flows(tmp_path)
    _write_annotations(tmp_path, text)
    with pytest.raises(DatasetValidationError, match=message):
        load_unsw_device(tmp_path, DEVICE)


def test_missing_annotation_file_is_rejected(tmp_path: Path) -> None:
    _write_flows(tmp_path)
    with pytest.raises(DatasetValidationError, match="Could not read attack annotations"):
        load_unsw_device(tmp_path, DEVICE)


@pytest.mark.parametrize(
    ("header", "message"),
    [
        (COLUMNS[1:], "exactly one Timestamp column"),
        ((*COLUMNS, "Timestamp"), "exactly one Timestamp column"),
        ((*COLUMNS, COLUMNS[1]), "repeats source column"),
        ((COLUMNS[0], *COLUMNS[2:]), "lacks shared counter"),
    ],
)
def test_flow_header_errors_are_rejected(
    tmp_path: Path, header: tuple[str, ...], message: str
) -> None:
    _write_annotations(tmp_path, VALID_ANNOTATION)
    _write_flows(tmp_path, header, ((1, *(1 for _ in header[1:])),))
    with pytest.raises(DatasetValidationError, match=message):
        load_unsw_device(tmp_path, DEVICE)


def test_flow_rows_must_match_header_width_and_be_numeric(tmp_path: Path) -> None:
    _write_annotations(tmp_path, VALID_ANNOTATION)
    _write_flows(tmp_path, rows=((90_000, 1),))
    with pytest.raises(DatasetValidationError, match="wrong column count"):
        load_unsw_device(tmp_path, DEVICE)
    _write_flows(tmp_path, rows=(("later", *(1 for _ in COLUMNS[1:])),))
    with pytest.raises(DatasetValidationError, match="Could not read flow counters"):
        load_unsw_device(tmp_path, DEVICE)


def test_empty_and_missing_flow_files_are_rejected(tmp_path: Path) -> None:
    _write_annotations(tmp_path, VALID_ANNOTATION)
    _write_flows(tmp_path, rows=())
    with pytest.raises(DatasetValidationError, match="no flow-stat records"):
        load_unsw_device(tmp_path, DEVICE)
    flow_file = _write_flows(tmp_path)
    flow_file.write_text("", encoding="utf-8")
    with pytest.raises(DatasetValidationError, match="Could not read flow counters"):
        load_unsw_device(tmp_path, DEVICE)
    flow_file.unlink()
    with pytest.raises(DatasetValidationError, match="Could not read flow counters"):
        load_unsw_device(tmp_path, DEVICE)


def test_device_needs_both_benign_and_attack_rows(tmp_path: Path) -> None:
    _write_flows(tmp_path)
    _write_annotations(tmp_path, "1,2,Feature,TcpSynDevice10L2D\n")
    with pytest.raises(DatasetValidationError, match="needs benign and annotated attack"):
        load_unsw_device(tmp_path, DEVICE)
    _write_annotations(tmp_path, "0,1000,Feature,TcpSynDevice10L2D\n")
    with pytest.raises(DatasetValidationError, match="needs benign and annotated attack"):
        load_unsw_device(tmp_path, DEVICE)


def test_group_helpers_reject_strata_without_unsw_devices(tmp_path: Path) -> None:
    with pytest.raises(DatasetValidationError, match="No UNSW IoT devices"):
        load_unsw_clients(tmp_path, StudyStratum.N_BAIOT)
    with pytest.raises(DatasetValidationError, match="No UNSW IoT devices"):
        unsw_source_files(tmp_path, StudyStratum.N_BAIOT)


def test_repeated_unused_source_columns_are_ignored(tmp_path: Path) -> None:
    _write_annotations(tmp_path, VALID_ANNOTATION)
    header = (*COLUMNS, "UnusedCounter", "UnusedCounter")
    _write_flows(tmp_path, header)
    assert load_unsw_device(tmp_path, DEVICE).feature_names == FEATURE_NAMES
