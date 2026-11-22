import csv
import ipaddress
from pathlib import Path

import numpy as np
import pytest

from pace.datasets.adapters.ton_iot import (
    load_ton_iot_clients,
    normalize_ton_iot_attack_type,
    ton_iot_feature_names,
    ton_iot_source_files,
)
from pace.types import (
    DatasetValidationError,
    IPv4AddressText,
    TonIotSampling,
    TonIoTStudyHost,
)

LAB_NETWORK = ipaddress.IPv4Network("192.168.1.0/24")
STUDY_HOSTS = frozenset(host.host_ip(LAB_NETWORK) for host in TonIoTStudyHost)
FIELDS = (
    "ts",
    "src_ip",
    "src_port",
    "dst_ip",
    "dst_port",
    "proto",
    "service",
    "duration",
    "src_bytes",
    "dst_bytes",
    "conn_state",
    "missed_bytes",
    "src_pkts",
    "src_ip_bytes",
    "dst_pkts",
    "dst_ip_bytes",
    "label",
    "type",
)


def _row(
    ts: str,
    source: str,
    destination: str,
    label: str,
    family: str,
    duration: str = "1.5",
) -> tuple[str, ...]:
    return (
        ts,
        source,
        "1234",
        destination,
        "443",
        "tcp",
        "-",
        duration,
        "20",
        "40",
        "SF",
        "-",
        "2",
        "100",
        "3",
        "120",
        label,
        family,
    )


def _write_csv(path: Path, rows: tuple[tuple[str, ...], ...]) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(FIELDS)
        writer.writerows(rows)
    return path.read_bytes()


def test_ton_adapter_groups_lan_victims_maps_features_and_sorts_timestamps(
    tmp_path: Path,
) -> None:
    root = tmp_path / "Processed_Network_dataset"
    first = root / "Network_dataset_1.csv"
    second = root / "Network_dataset_2.csv"
    original = (
        _write_csv(
            first,
            (
                _row("3", "192.168.1.30", "192.168.1.152", "1", "injection", "-"),
                _row("2", "8.8.8.8", "192.168.1.152", "0", "normal"),
                _row("1", "192.168.1.31", "192.168.1.152", "1", "ddos"),
            ),
        ),
        _write_csv(
            second,
            (_row("4", "192.168.1.32", "192.168.1.152", "1", "ddos"),),
        ),
    )

    clients = load_ton_iot_clients(root, LAB_NETWORK, {IPv4AddressText("192.168.1.152")})

    assert len(clients) == 1
    client = clients[0]
    assert client.client.name == "192.168.1.152"
    assert client.feature_names == ton_iot_feature_names()
    assert len(client.feature_names) == 29
    assert np.allclose(client.benign_features[:, 0], [np.log1p(1.5)], rtol=1e-6)
    assert client.benign_features[0, 3] == 0.0
    assert np.isclose(client.benign_features[0, 10], np.log1p(1.0))  # state_SF
    assert np.isclose(client.benign_features[0, 21], np.log1p(1.0))  # proto_tcp
    assert client.benign_features[0, 24] == 0.0  # service_known
    assert client.benign_features[0, 25] == 0.0  # src_port_privileged
    assert np.isclose(client.benign_features[0, 27], np.log1p(1.0))  # dst_port_privileged
    assert [capture.attack_type.identifier for capture in client.attack_captures] == [
        "DDOS",
        "INJECTION",
    ]
    assert np.allclose(
        client.attack_captures[0].features[:, 0], [np.log1p(1.5), np.log1p(1.5)], rtol=1e-6
    )
    assert client.attack_captures[0].source_order.tolist() == [0, 1]
    assert client.attack_captures[1].source_order.tolist() == [0]
    assert all(not capture.features.flags.writeable for capture in client.attack_captures)
    assert first.read_bytes() == original[0]
    assert second.read_bytes() == original[1]


def test_ton_adapter_rejects_invalid_manifest_and_unmatched_clients(tmp_path: Path) -> None:
    root = tmp_path / "Processed_Network_dataset"
    path = root / "Network_dataset_1.csv"
    _write_csv(path, (_row("1", "192.168.1.152", "8.8.8.8", "0", "normal"),))

    identities = {IPv4AddressText("192.168.2.1")}
    with pytest.raises(DatasetValidationError, match="outside the lab LAN"):
        load_ton_iot_clients(root, LAB_NETWORK, identities)
    identities = {IPv4AddressText("192.168.1.152")}
    with pytest.raises(DatasetValidationError, match="no clients"):
        load_ton_iot_clients(root, LAB_NETWORK, identities)


def test_ton_adapter_discovers_files_and_normalizes_type(tmp_path: Path) -> None:
    root = tmp_path / "TON-IoT" / "Processed_datasets" / "Processed_Network_dataset"
    first = root / "Network_dataset_1.csv"
    first.parent.mkdir(parents=True)
    first.touch()
    (root / "ignore.csv").touch()

    assert ton_iot_source_files(root) == (first,)
    assert normalize_ton_iot_attack_type("mitm").identifier == "MAN_IN_THE_MIDDLE"
    with pytest.raises(DatasetValidationError, match="benign"):
        normalize_ton_iot_attack_type("normal")


def test_ton_adapter_discovers_network_release_from_shared_raw_root(tmp_path: Path) -> None:
    data_root = tmp_path / "raw"
    source = (
        data_root
        / "TON-IoT"
        / "Processed_datasets"
        / "Processed_Network_dataset"
        / "Network_dataset_1.csv"
    )
    _write_csv(
        source,
        (
            _row("1", "192.168.1.30", "192.168.1.152", "0", "normal"),
            _row("2", "192.168.1.30", "192.168.1.152", "1", "ddos"),
        ),
    )

    clients = load_ton_iot_clients(
        data_root,
        LAB_NETWORK,
        (TonIoTStudyHost.MIDDLEWARE_SERVER,),
    )

    assert ton_iot_source_files(data_root) == (source,)
    assert len(clients) == 1
    assert clients[0].client.name == "192.168.1.152"


def test_ton_adapter_rejects_label_type_mismatch_and_malformed_selected_features(
    tmp_path: Path,
) -> None:
    root = tmp_path / "Processed_Network_dataset"
    source = root / "Network_dataset_1.csv"
    _write_csv(
        source,
        (_row("1", "192.168.1.30", "192.168.1.152", "1", "normal"),),
    )
    identities = {IPv4AddressText("192.168.1.152")}
    with pytest.raises(DatasetValidationError, match="inconsistent label/type"):
        load_ton_iot_clients(root, LAB_NETWORK, identities)

    _write_csv(
        source,
        (
            _row("1", "8.8.8.8", "192.168.1.152", "0", "normal"),
            _row("2", "192.168.1.30", "192.168.1.152", "1", "ddos", "nan"),
        ),
    )
    identities = {IPv4AddressText("192.168.1.152")}
    with pytest.raises(DatasetValidationError, match="malformed selected numeric"):
        load_ton_iot_clients(root, LAB_NETWORK, identities)


def test_ton_source_discovery_rejects_missing_directory(tmp_path: Path) -> None:
    with pytest.raises(DatasetValidationError, match="No TON-IoT network CSV files"):
        ton_iot_source_files(tmp_path)
    with pytest.raises(DatasetValidationError, match="directory is unavailable"):
        ton_iot_source_files(tmp_path / "missing")


def test_ton_target_manifest_and_ambiguous_local_flow_policy() -> None:
    assert tuple(tuple(TonIoTStudyHost)) == (
        TonIoTStudyHost.ROUTER,
        TonIoTStudyHost.MIDDLEWARE_SERVER,
        TonIoTStudyHost.SECURITY_ONION,
        TonIoTStudyHost.OWASP_SHEPHERD,
        TonIoTStudyHost.ORCHESTRATED_SERVER,
        TonIoTStudyHost.WINDOWS_7,
        TonIoTStudyHost.METASPLOITABLE_3,
        TonIoTStudyHost.WINDOWS_10,
        TonIoTStudyHost.HOST_79,
    )
    assert tuple(host.host_ip(LAB_NETWORK) for host in TonIoTStudyHost) == (
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


HOST = IPv4AddressText("192.168.1.152")


def _destination_row(
    ts: str, destination: str, label: str, family: str, duration: str = "1.5"
) -> tuple[str, ...]:
    return (
        ts,
        "192.168.1.30",
        "1234",
        destination,
        "443",
        "tcp",
        "-",
        duration,
        "20",
        "40",
        "SF",
        "-",
        "2",
        "100",
        "3",
        "120",
        label,
        family,
    )


def _write(path: Path, rows: tuple[tuple[str, ...], ...], fields: tuple[str, ...] = FIELDS) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(fields)
        writer.writerows(rows)


def _load(root: Path) -> None:
    load_ton_iot_clients(root, LAB_NETWORK, {HOST})


def test_missing_fields_are_rejected(tmp_path: Path) -> None:
    _write(tmp_path / "Network_dataset_1.csv", (), FIELDS[:-1])
    with pytest.raises(DatasetValidationError, match="missing TON-IoT fields"):
        _load(tmp_path)


def test_malformed_selected_rows_are_rejected_and_unselected_ones_skipped(tmp_path: Path) -> None:
    source = tmp_path / "Network_dataset_1.csv"
    selected_long = (*_destination_row("1", HOST, "1", "ddos"), "extra")
    _write(source, (selected_long,))
    with pytest.raises(DatasetValidationError, match="malformed selected TON-IoT row"):
        _load(tmp_path)
    unselected_short = ("2", "8.8.8.8", "1", "9.9.9.9")
    _write(
        source,
        (
            unselected_short,
            _destination_row("3", HOST, "0", "normal"),
            _destination_row("4", HOST, "1", "ddos"),
        ),
    )
    _load(tmp_path)


def test_unsupported_type_and_non_finite_values_are_rejected(tmp_path: Path) -> None:
    source = tmp_path / "Network_dataset_1.csv"
    _write(source, (_destination_row("1", HOST, "1", "mystery"),))
    with pytest.raises(DatasetValidationError, match="unsupported selected TON-IoT type"):
        _load(tmp_path)
    _write(source, (_destination_row("nan", HOST, "1", "ddos"),))
    with pytest.raises(DatasetValidationError, match="malformed selected numeric values"):
        _load(tmp_path)
    _write(source, (_destination_row("1", HOST, "1", "ddos", "inf"),))
    with pytest.raises(DatasetValidationError, match="malformed selected numeric values"):
        _load(tmp_path)
    broken = _destination_row("1", HOST, "1", "ddos")
    _write(source, ((*broken[:4], "inf", *broken[5:]),))
    with pytest.raises(DatasetValidationError, match="malformed selected numeric values"):
        _load(tmp_path)


def test_identities_must_be_lan_addresses(tmp_path: Path) -> None:
    _write(tmp_path / "Network_dataset_1.csv", ())
    identities = {IPv4AddressText("not-an-ip")}
    with pytest.raises(DatasetValidationError, match="must be LAN IPv4"):
        load_ton_iot_clients(tmp_path, LAB_NETWORK, identities)
    identities = {IPv4AddressText("8.8.8.8")}
    with pytest.raises(DatasetValidationError, match="outside the lab LAN"):
        load_ton_iot_clients(tmp_path, LAB_NETWORK, identities)


def test_unreadable_sources_and_empty_selections_are_rejected(tmp_path: Path) -> None:
    (tmp_path / "Network_dataset_1.csv").mkdir()
    with pytest.raises(DatasetValidationError, match="Could not read TON-IoT source"):
        _load(tmp_path)
    (tmp_path / "Network_dataset_1.csv").rmdir()
    _write(tmp_path / "Network_dataset_1.csv", (_destination_row("1", "8.8.8.8", "0", "normal"),))
    with pytest.raises(DatasetValidationError, match="produced no clients"):
        _load(tmp_path)


def test_default_manifest_is_used_and_discovery_prefers_nested_layouts(tmp_path: Path) -> None:
    nested = tmp_path / "Processed_datasets" / "Processed_Network_dataset"
    _write(nested / "Network_dataset_1.csv", ())
    assert len(ton_iot_source_files(tmp_path)) == 1
    with pytest.raises(DatasetValidationError, match="produced no clients"):
        load_ton_iot_clients(tmp_path, LAB_NETWORK, tuple(TonIoTStudyHost))
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(DatasetValidationError, match="No TON-IoT network CSV files"):
        ton_iot_source_files(empty)


def test_chunk_flushing_thinning_and_reservoir_replacement(tmp_path: Path) -> None:
    sampling = TonIotSampling(
        benign_cap=2,
        attack_cap_per_label=2,
        source_chunk_rows=4,
        chunk_group_threshold=1,
        chunk_keep_fraction=0.75,
        chunk_reservoir_cap=3,
    )
    rows = tuple(
        _destination_row(
            str(index), HOST, "0" if index % 2 else "1", "normal" if index % 2 else "ddos"
        )
        for index in range(1, 25)
    ) + tuple(_destination_row(str(100 + index), "8.8.8.8", "0", "normal") for index in range(8))
    _write(tmp_path / "Network_dataset_1.csv", rows)
    clients = load_ton_iot_clients(tmp_path, LAB_NETWORK, {HOST}, sampling)
    assert len(clients) == 1
    client = clients[0]
    assert client.benign_features.shape[0] <= 2
    assert client.attack_captures[0].features.shape[0] <= 2
    assert client.benign_source_population_rows == 12
