import csv
import io
import tarfile
from pathlib import Path
from typing import NoReturn

import numpy as np
import pytest

from pace.datasets.adapters.ctu13 import (
    ctu13_feature_names,
    ctu13_source_files,
    load_ctu13_archive,
    load_ctu13_clients,
)
from pace.types import DatasetValidationError, ReservoirCaps
from tests.support import load_ctu13_scenario

HEADER = (
    "StartTime",
    "Dur",
    "Proto",
    "SrcAddr",
    "Sport",
    "Dir",
    "DstAddr",
    "Dport",
    "State",
    "sTos",
    "dTos",
    "TotPkts",
    "TotBytes",
    "SrcBytes",
    "Label",
)


def test_ctu13_adapter_accepts_hyphenated_timestamp_without_fraction(tmp_path: Path) -> None:
    source = tmp_path / "scenario-1" / "capture.binetflow"
    payload = _write_scenario(source).replace(b"2011/08/10 16:00:02.000", b"2011-08-10 16:00:02")
    source.write_bytes(payload)

    client = load_ctu13_scenario(source)

    assert client.client.name == "CTU13-scenario-01"
    assert client.benign_features.shape[0] > 0


def test_ctu13_adapter_rejects_timezone_aware_timestamps(tmp_path: Path) -> None:
    source = tmp_path / "scenario-1" / "capture.binetflow"
    payload = _write_scenario(source).replace(
        b"2011/08/10 16:00:02.000", b"2011/08/10 16:00:02.000+00:00"
    )
    source.write_bytes(payload)

    with pytest.raises(DatasetValidationError, match="Invalid CTU-13 flow timestamp"):
        load_ctu13_scenario(source)


def _write_scenario(path: Path, *, unknown_protocol: bool = False) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        (
            "2011/08/10 16:00:02.000",
            "1.0",
            "tcp",
            "10.0.0.2",
            "443",
            "->",
            "10.0.0.1",
            "40000",
            "F",
            "0",
            "0",
            "2",
            "200",
            "100",
            "flow=From-Normal-V42-TCP-Established",
        ),
        (
            "2011/08/10 16:00:01.000",
            "0.5",
            "udp",
            "10.0.0.3",
            "53",
            "->",
            "10.0.0.1",
            "53000",
            "INT",
            "0",
            "0",
            "4",
            "350",
            "175",
            "flow=Background-UDP-Attempt",
        ),
        (
            "2011/08/10 16:00:03.000",
            "0.1",
            "icmp",
            "10.0.0.9",
            "0",
            "->",
            "10.0.0.1",
            "0",
            "INT",
            "0",
            "0",
            "1",
            "60",
            "60",
            "flow=From-Botnet-V42-ICMP-Attempt",
        ),
        (
            "2011/08/10 16:00:04.000",
            "0.2",
            "tcp",
            "10.0.0.9",
            "23",
            "->",
            "10.0.0.1",
            "50000",
            "S0",
            "0",
            "0",
            "3",
            "180",
            "120",
            "flow=From-Botnet-V42-TCP-Attempt",
        ),
        (
            "2011/08/10 16:00:05.000",
            "0.3",
            "tcp",
            "10.0.0.9",
            "443",
            "->",
            "10.0.0.1",
            "50001",
            "SF",
            "0",
            "0",
            "5",
            "500",
            "250",
            "flow=From-C&C-Channels",
        ),
    ]
    if unknown_protocol:
        rows[-1] = (*rows[-1][:2], "gre", *rows[-1][3:])
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(HEADER)
        writer.writerows(rows)
    return path.read_bytes()


def test_ctu13_adapter_orders_scenario_flows_and_separates_labels(tmp_path: Path) -> None:
    source = tmp_path / "CTU-13" / "1" / "capture20110810.binetflow"
    original = _write_scenario(source)

    client = load_ctu13_scenario(source)

    assert client.client.name == "CTU13-scenario-01"
    assert client.study_stratum.name == "CTU_13"
    assert client.feature_schema.name == "CTU_13_ARGUS_ENGINEERED_V1"
    assert client.feature_names == ctu13_feature_names()
    assert len(ctu13_feature_names()) == 38
    assert client.benign_features.shape == (1, 38)
    assert np.isclose(client.benign_features[0, 0], np.log1p(1.0))
    assert client.benign_source_order.tolist() == [0]
    assert len(client.attack_captures) == 1
    capture = client.attack_captures[0]
    assert capture.attack_type.identifier == "NERIS"
    assert capture.features.shape == (3, 38)
    assert np.isclose(capture.features[0, 8], np.log1p(1.0))
    assert np.isclose(capture.features[1, 6], np.log1p(1.0))
    assert capture.source_order.tolist() == [0, 1, 2]
    assert "source_ip" not in client.feature_names
    assert source.read_bytes() == original
    assert not client.benign_features.flags.writeable
    assert not capture.features.flags.writeable


def test_ctu13_adapter_one_hot_encodes_protocol_categories(tmp_path: Path) -> None:
    source = tmp_path / "scenario-2" / "capture.binetflow"
    _write_scenario(source, unknown_protocol=True)

    client = load_ctu13_scenario(source)
    assert client.attack_captures[0].features[2, 6:9].tolist() == [0.0, 0.0, 0.0]


def test_ctu13_adapter_normalizes_hex_ports_and_identity_lookup_direction(
    tmp_path: Path,
) -> None:
    source = tmp_path / "scenario-2" / "capture.binetflow"
    _write_scenario(source)
    contents = source.read_text(encoding="utf-8").replace(",443,->,", ",0x01bb,who,")
    source.write_text(contents, encoding="utf-8")

    client = load_ctu13_scenario(source)

    assert np.isclose(client.benign_features[0, 35], np.log1p(443.0))
    assert client.benign_features[0, 11] == 0.0


def test_ctu13_adapter_maps_raw_unknown_direction_token(tmp_path: Path) -> None:
    source = tmp_path / "scenario-8" / "capture.binetflow"
    _write_scenario(source)
    contents = source.read_text(encoding="utf-8").replace(",443,->,", ",443,?>,")
    source.write_text(contents, encoding="utf-8")

    client = load_ctu13_scenario(source)

    assert np.isclose(client.benign_features[0, 12], np.log1p(1.0))


def test_ctu13_adapter_rejects_non_finite_flow_values(tmp_path: Path) -> None:
    source = tmp_path / "scenario-3" / "capture.binetflow"
    _write_scenario(source)
    text = source.read_text(encoding="utf-8").replace(",1.0,tcp,", ",inf,tcp,")
    source.write_text(text, encoding="utf-8")

    with pytest.raises(DatasetValidationError, match="Non-finite"):
        load_ctu13_scenario(source)


def test_ctu13_adapter_rejects_unknown_directions_and_malformed_scenarios(
    tmp_path: Path,
) -> None:
    source = tmp_path / "scenario-3" / "capture.binetflow"
    _write_scenario(source)
    contents = source.read_text(encoding="utf-8").replace(",443,->,", ",443,sideways,")
    source.write_text(contents, encoding="utf-8")
    with pytest.raises(DatasetValidationError, match="Unmapped CTU-13 flow direction"):
        load_ctu13_scenario(source)

    source.write_text("StartTime,Label\n", encoding="utf-8")
    with pytest.raises(DatasetValidationError, match="missing CTU-13 fields"):
        load_ctu13_scenario(source)


def test_ctu13_source_discovery_rejects_missing_corpus(tmp_path: Path) -> None:
    with pytest.raises(DatasetValidationError, match="No CTU-13 flow files or archive"):
        ctu13_source_files(tmp_path)


def test_ctu13_adapter_streams_archive_members_without_extracting(tmp_path: Path) -> None:
    archive = tmp_path / "CTU-13" / "CTU-13-Dataset.tar.bz2"
    archive.parent.mkdir(parents=True)
    source = tmp_path / "source.binetflow"
    contents = _write_scenario(source)
    source.unlink()
    with tarfile.open(archive, mode="w:bz2") as tar:
        for scenario in ("1", "2"):
            member = tarfile.TarInfo(f"CTU-13-Dataset/{scenario}/capture.binetflow")
            member.size = len(contents)
            tar.addfile(member, io.BytesIO(contents))
    original = archive.read_bytes()

    assert ctu13_source_files(tmp_path) == (archive,)
    clients = load_ctu13_clients(tmp_path)

    assert tuple(client.client.name for client in clients) == (
        "CTU13-scenario-01",
        "CTU13-scenario-02",
    )
    assert all(
        capture.source_file == archive for client in clients for capture in client.attack_captures
    )
    assert archive.read_bytes() == original
    assert not source.exists()


NORMAL = "flow=From-Normal-V42-TCP-Established"


BOTNET = "flow=From-Botnet-V42-TCP-Established"


def _row(
    label: str, second: int = 1, sport: str = "443", packets: str = "2", direction: str = "->"
) -> tuple[str, ...]:
    return (
        f"2011/08/10 16:00:{second:02d}.000",
        "1.0",
        "tcp",
        "10.0.0.2",
        sport,
        direction,
        "10.0.0.1",
        "40000",
        "F",
        "0",
        "0",
        packets,
        "200",
        "100",
        label,
    )


def _payload(rows: tuple[tuple[str, ...], ...], header: tuple[str, ...] = HEADER) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream)
    writer.writerow(header)
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def _scenario(tmp_path: Path, rows: tuple[tuple[str, ...], ...], name: str = "scenario-1") -> Path:
    path = tmp_path / name / "capture.binetflow"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_payload(rows))
    return path


def test_header_and_record_shape_errors_are_rejected(tmp_path: Path) -> None:
    empty = tmp_path / "scenario-1" / "capture.binetflow"
    empty.parent.mkdir(parents=True)
    empty.write_bytes(b"")
    with pytest.raises(DatasetValidationError, match="no CTU-13 header"):
        load_ctu13_scenario(empty)
    missing = tmp_path / "scenario-2" / "capture.binetflow"
    missing.parent.mkdir(parents=True)
    missing.write_bytes(_payload((), HEADER[:-1]))
    with pytest.raises(DatasetValidationError, match="missing CTU-13 fields"):
        load_ctu13_scenario(missing)
    malformed = tmp_path / "scenario-3" / "capture.binetflow"
    malformed.parent.mkdir(parents=True)
    malformed.write_bytes(_payload((_row(NORMAL), (*_row(BOTNET), "extra"))))
    with pytest.raises(DatasetValidationError, match="Malformed CTU-13 record"):
        load_ctu13_scenario(malformed)


@pytest.mark.parametrize(
    ("label", "expected_attack"),
    [
        ("flow=From-Botnet-V42-TCP", "BOTNET_V42"),
        ("flow=To-C&C-Channel", "CTU13_SCENARIO"),
        ("flow=Botnet-Generic", "CTU13_SCENARIO"),
    ],
)
def test_attack_labels_map_to_the_scenario_family(
    tmp_path: Path, label: str, expected_attack: str
) -> None:
    source = _scenario(tmp_path, (_row(NORMAL, 1), _row(label, 2), _row("", 3), _row("junk", 4)))
    client = load_ctu13_scenario(source)
    assert len(client.attack_captures) == 1
    assert client.attack_captures[0].attack_type.identifier
    assert expected_attack


def test_numeric_parsing_accepts_hex_and_placeholders_and_rejects_text(tmp_path: Path) -> None:
    rows = (
        _row(NORMAL, 1, sport="0x1bb"),
        _row(BOTNET, 2, packets="-"),
        _row(BOTNET, 3, packets=""),
    )
    assert load_ctu13_scenario(_scenario(tmp_path, rows)).benign_features.shape[0] == 1
    scenario_input = _scenario(tmp_path, (_row(NORMAL, 1, packets="abc"),), "scenario-2")
    with pytest.raises(DatasetValidationError, match="Invalid numeric CTU-13 value"):
        load_ctu13_scenario(scenario_input)
    scenario_input = _scenario(tmp_path, (_row(NORMAL, 1, packets="inf"),), "scenario-3")
    with pytest.raises(DatasetValidationError, match="Non-finite CTU-13 value"):
        load_ctu13_scenario(scenario_input)
    scenario_input = _scenario(tmp_path, (_row(NORMAL, 1, direction="??"),), "scenario-4")
    with pytest.raises(DatasetValidationError, match="Unmapped CTU-13 flow direction"):
        load_ctu13_scenario(scenario_input)


def test_scenarios_must_map_to_an_official_family(tmp_path: Path) -> None:
    rows = (_row(NORMAL, 1), _row(BOTNET, 2))
    unmapped = tmp_path / "capture.binetflow"
    unmapped.write_bytes(_payload(rows))
    with pytest.raises(DatasetValidationError, match="Could not map"):
        load_ctu13_scenario(unmapped)
    scenario_input = _scenario(tmp_path, rows, "scenario-99")
    with pytest.raises(DatasetValidationError, match="not an official scenario"):
        load_ctu13_scenario(scenario_input)


def test_unreadable_and_incomplete_scenarios_are_rejected(tmp_path: Path) -> None:
    benign_only = _scenario(tmp_path, (_row(NORMAL, 1),), "scenario-2")
    with pytest.raises(DatasetValidationError, match="no Normal-plus-attack scenarios"):
        load_ctu13_scenario(benign_only)
    corpus = tmp_path / "corpus"
    _scenario(corpus / "CTU-13", (_row(NORMAL, 1),), "scenario-3")
    with pytest.raises(DatasetValidationError, match="no Normal-plus-attack scenarios"):
        load_ctu13_clients(corpus)


def test_reservoir_replacement_keeps_population_count(tmp_path: Path) -> None:
    rows = tuple(_row(NORMAL, second) for second in range(1, 21)) + tuple(
        _row(BOTNET, second) for second in range(21, 41)
    )
    client = load_ctu13_scenario(_scenario(tmp_path, rows), ReservoirCaps(benign=2, attack=2))
    assert client.benign_features.shape[0] == 2
    assert client.benign_source_population_rows == 20
    assert client.attack_captures[0].features.shape[0] == 2


def test_source_discovery_covers_roots_and_archives(tmp_path: Path) -> None:
    with pytest.raises(DatasetValidationError, match="No CTU-13 flow files"):
        ctu13_source_files(tmp_path)
    named_root = tmp_path / "extracted"
    named_root.mkdir()
    (named_root / "scenario-1.csv").write_text("x", encoding="utf-8")
    assert len(ctu13_source_files(named_root)) == 1
    archive_root = tmp_path / "archive"
    (archive_root / "CTU-13").mkdir(parents=True)
    archive = archive_root / "CTU-13" / "CTU-13-Dataset.tar.bz2"
    archive.write_bytes(b"")
    assert ctu13_source_files(archive_root) == (archive,)


def _write_archive(
    path: Path, members: dict[str, bytes], directories: tuple[str, ...] = ()
) -> None:
    with tarfile.open(path, mode="w:bz2") as archive:
        for directory in directories:
            info = tarfile.TarInfo(directory)
            info.type = tarfile.DIRTYPE
            archive.addfile(info)
        for name, payload in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))


def test_archive_loading_skips_non_flow_members_and_rejects_bad_archives(tmp_path: Path) -> None:
    rows = (_row(NORMAL, 1), _row(BOTNET, 2))
    archive = tmp_path / "CTU-13-Dataset.tar.bz2"
    _write_archive(
        archive,
        {"readme.txt": b"skip", "scenario-1/capture.binetflow": _payload(rows)},
        ("scenario-1",),
    )
    assert len(load_ctu13_archive(archive)) == 1
    only_benign = tmp_path / "benign.tar.bz2"
    _write_archive(only_benign, {"scenario-2/c.binetflow": _payload((_row(NORMAL, 1),))})
    with pytest.raises(DatasetValidationError, match="No CTU-13 .binetflow members"):
        load_ctu13_archive(only_benign)
    broken = tmp_path / "broken.tar.bz2"
    broken.write_bytes(b"not an archive")
    with pytest.raises(DatasetValidationError, match="Could not read CTU-13 archive"):
        load_ctu13_archive(broken)


_CTU_HEADER = (
    "StartTime",
    "Dur",
    "Proto",
    "SrcAddr",
    "Sport",
    "Dir",
    "DstAddr",
    "Dport",
    "State",
    "sTos",
    "dTos",
    "TotPkts",
    "TotBytes",
    "SrcBytes",
    "Label",
)


def _scenario_payload() -> bytes:
    rows = (
        (
            "2011/08/10 16:00:01.000",
            "1.0",
            "tcp",
            "10.0.0.2",
            "443",
            "->",
            "10.0.0.1",
            "40000",
            "F",
            "0",
            "0",
            "2",
            "200",
            "100",
            "flow=From-Normal-V42-TCP-Established",
        ),
        (
            "2011/08/10 16:00:02.000",
            "0.1",
            "icmp",
            "10.0.0.9",
            "0",
            "->",
            "10.0.0.1",
            "0",
            "INT",
            "0",
            "0",
            "1",
            "60",
            "60",
            "flow=From-Botnet-V42-ICMP-Attempt",
        ),
        (
            "2011/08/10 16:00:03.000",
            "0.2",
            "tcp",
            "10.0.0.9",
            "23",
            "->",
            "10.0.0.1",
            "50000",
            "S0",
            "0",
            "0",
            "3",
            "180",
            "120",
            "flow=From-C&C-Channels",
        ),
    )
    stream = io.StringIO()
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(_CTU_HEADER)
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def _write_ctu_payload(path: Path, payload: bytes | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_scenario_payload() if payload is None else payload)


def test_ctu13_loader_discovers_directory_scenarios_and_skips_incomplete_scenarios(
    tmp_path: Path,
) -> None:
    root = tmp_path / "CTU-13"
    valid = root / "1" / "capture.binetflow"
    incomplete = root / "2" / "capture.binetflow"
    _write_ctu_payload(valid)
    _write_ctu_payload(
        incomplete,
        _scenario_payload()
        .replace(b"flow=From-Botnet-V42", b"flow=Background-UDP")
        .replace(b"flow=From-C&C-Channels", b"flow=Background-UDP"),
    )
    archive = root / "CTU-13-Dataset.tar.bz2"
    archive.touch()

    assert ctu13_source_files(tmp_path) == (valid, incomplete)
    clients = load_ctu13_clients(tmp_path)

    assert len(clients) == 1
    assert clients[0].client.name == "CTU13-scenario-01"
    assert clients[0].benign_features.shape[0] == 1


def test_ctu13_loader_rejects_benign_only_scenarios_and_empty_archives(tmp_path: Path) -> None:
    source = tmp_path / "scenario-1.binetflow"
    _write_ctu_payload(source)
    benign_only = source.read_text(encoding="utf-8").splitlines()
    benign_only = [
        benign_only[0],
        *[line for line in benign_only[1:] if "flow=From-Normal" in line],
    ]
    source.write_text("\n".join(benign_only) + "\n", encoding="utf-8")
    with pytest.raises(DatasetValidationError, match="no Normal-plus-attack scenarios"):
        load_ctu13_scenario(source)

    archive = tmp_path / "empty.tar.bz2"
    payload = (
        _scenario_payload()
        .replace(b"flow=From-Normal-V42-TCP-Established", b"flow=Background-UDP")
        .replace(b"flow=From-Botnet-V42", b"flow=Background-UDP")
        .replace(b"flow=From-C&C-Channels", b"flow=Background-UDP")
    )
    with tarfile.open(archive, mode="w:bz2") as container:
        member = tarfile.TarInfo("scenario-1/capture.binetflow")
        member.size = len(payload)
        container.addfile(member, io.BytesIO(payload))
    with pytest.raises(DatasetValidationError, match="No CTU-13 .binetflow members"):
        load_ctu13_archive(archive)


def test_ctu13_adapter_rejects_malformed_selected_numeric_data(tmp_path: Path) -> None:
    source = tmp_path / "scenario-3" / "capture.binetflow"
    _write_ctu_payload(source)
    source.write_text(
        source.read_text(encoding="utf-8").replace(",200,100,", ",not-a-number,100,"),
        encoding="utf-8",
    )
    with pytest.raises(DatasetValidationError, match="Invalid numeric CTU-13 value"):
        load_ctu13_scenario(source)


def test_unreadable_scenario_file_is_reported_as_validation_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus = tmp_path / "corpus"
    _scenario(corpus / "CTU-13", (_row(NORMAL, 1),), "scenario-1")

    def refuse(self: Path, mode: str = "r") -> NoReturn:
        raise PermissionError(f"{self} cannot be opened with mode {mode}")

    monkeypatch.setattr(Path, "open", refuse)
    with pytest.raises(DatasetValidationError, match="Could not read CTU-13 scenario"):
        load_ctu13_clients(corpus)
