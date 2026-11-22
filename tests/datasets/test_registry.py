import ipaddress
import os
from pathlib import Path

import numpy as np
import pytest

import pace.datasets.registry as registry
from pace.datasets.registry import (
    DatasetAdapter,
    DatasetLoad,
    dataset_adapter,
    dataset_source_files,
    inspect_dataset_sources,
    load_dataset,
    resolve_study_stratum,
)
from pace.experiment.artifacts import campaign_artifact_path
from pace.types import (
    AttackCapture,
    AttackType,
    AttackTypeName,
    ClientName,
    DatasetAvailability,
    DatasetClientData,
    DatasetClientId,
    DatasetClientLoader,
    DatasetInventoryReport,
    DatasetLocationKind,
    DatasetName,
    DatasetSourceDiscovery,
    DatasetSourceInventory,
    DatasetValidationError,
    FeatureName,
    FeatureSchemaId,
    LabNetwork,
    StudyStratum,
    default_study_stratum,
)


def _client(
    name: str = "client",
    dataset: DatasetName = DatasetName.IOT_23,
    stratum: StudyStratum = StudyStratum.IOT_23,
    schema: FeatureSchemaId = FeatureSchemaId.IOT_23_ZEEK_ENGINEERED_V1,
) -> DatasetClientData:
    features = np.asarray([[1.0]], dtype=np.float64)
    order = np.asarray([0], dtype=np.int64)
    return DatasetClientData(
        DatasetClientId(dataset, ClientName(name)),
        stratum,
        schema,
        (FeatureName("duration_seconds"),),
        features,
        order,
        (
            AttackCapture(
                AttackType(AttackTypeName("ATTACK")),
                Path("attack.csv"),
                features.copy(),
                order.copy(),
            ),
        ),
    )


def _adapter_callbacks(
    client: DatasetClientData,
) -> tuple[DatasetClientLoader, DatasetSourceDiscovery]:
    def load(
        _root: Path, _stratum: StudyStratum, _network: LabNetwork
    ) -> tuple[DatasetClientData, ...]:
        return (client,)

    def sources(_root: Path, _stratum: StudyStratum) -> tuple[Path, ...]:
        return ()

    return load, sources


def test_registry_covers_every_frozen_dataset() -> None:
    registered = tuple(dataset_adapter(dataset).dataset for dataset in DatasetName)

    assert set(registered) == set(DatasetName)
    assert len(registered) == len(set(registered))


def test_dataset_load_and_adapter_validation_reject_invalid_identity_combinations(
    tmp_path: Path,
) -> None:
    client = _client()
    with pytest.raises(ValueError, match="stratum does not belong"):
        DatasetLoad(
            DatasetName.IOT_23,
            StudyStratum.TON_IOT,
            (client,),
            (tmp_path / "source.csv",),
        )
    with pytest.raises(ValueError, match="require clients and source files"):
        DatasetLoad(DatasetName.IOT_23, StudyStratum.IOT_23, (), ())
    unsw_client = _client(
        "unsw-client",
        DatasetName.UNSW_IOT_ATTACK_FLOWS,
        StudyStratum.UNSW_IOT_ATTACK_FLOWS_G2,
        FeatureSchemaId.UNSW_MUD_G2_MINUTE_COUNTERS,
    )
    with pytest.raises(ValueError, match="do not match"):
        DatasetLoad(
            DatasetName.UNSW_IOT_ATTACK_FLOWS,
            StudyStratum.UNSW_IOT_ATTACK_FLOWS_G1,
            (unsw_client,),
            (tmp_path / "source.csv",),
        )
    with pytest.raises(ValueError, match="identities must be unique"):
        DatasetLoad(
            DatasetName.IOT_23,
            StudyStratum.IOT_23,
            (client, client),
            (tmp_path / "source.csv",),
        )
    with pytest.raises(ValueError, match="source files must be unique"):
        DatasetLoad(
            DatasetName.IOT_23,
            StudyStratum.IOT_23,
            (client,),
            (tmp_path / "source.csv", tmp_path / "source.csv"),
        )
    adapter_callbacks = _adapter_callbacks(client)
    with pytest.raises(ValueError, match="declare valid strata"):
        DatasetAdapter(DatasetName.IOT_23, (), *adapter_callbacks)
    adapter_callbacks = _adapter_callbacks(client)
    with pytest.raises(ValueError, match="declare valid strata"):
        DatasetAdapter(
            DatasetName.IOT_23,
            (StudyStratum.TON_IOT,),
            *adapter_callbacks,
        )
    adapter_callbacks = _adapter_callbacks(client)
    with pytest.raises(ValueError, match="cannot repeat"):
        DatasetAdapter(
            DatasetName.IOT_23,
            (StudyStratum.IOT_23, StudyStratum.IOT_23),
            *adapter_callbacks,
        )


def test_single_stratum_wrapper_and_registry_loaders(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    client = _client()
    source = tmp_path / "source.csv"
    calls: list[str] = []

    def load(
        root: Path, _stratum: StudyStratum, _network: LabNetwork
    ) -> tuple[DatasetClientData, ...]:
        calls.append(root.as_posix())
        return (client,)

    def sources(root: Path, _stratum: StudyStratum) -> tuple[Path, ...]:
        calls.append(root.as_posix())
        return (source,)

    adapter = DatasetAdapter(DatasetName.IOT_23, (StudyStratum.IOT_23,), load, sources)
    monkeypatch.setattr(registry, "_registered_adapters", lambda: (adapter,))

    loaded = load_dataset(tmp_path, LAB_NETWORK, DatasetName.IOT_23)

    assert loaded.clients == (client,)
    assert loaded.source_files == (source,)
    assert dataset_adapter(DatasetName.IOT_23) is adapter
    assert dataset_source_files(tmp_path, DatasetName.IOT_23) == (source,)
    assert len(calls) == 3


def test_stratum_resolution_validates_dataset_membership() -> None:
    assert resolve_study_stratum(DatasetName.IOT_23) is StudyStratum.IOT_23
    assert default_study_stratum(DatasetName.TON_IOT) is StudyStratum.TON_IOT
    with pytest.raises(DatasetValidationError, match="not registered"):
        resolve_study_stratum(DatasetName.IOT_23, StudyStratum.TON_IOT)
    with pytest.raises(DatasetValidationError, match="two study strata"):
        default_study_stratum(DatasetName.UNSW_IOT_ATTACK_FLOWS)


def test_multistratum_dataset_requires_an_explicit_group(tmp_path: Path) -> None:
    with pytest.raises(DatasetValidationError, match="multiple study strata"):
        load_dataset(tmp_path, LAB_NETWORK, DatasetName.UNSW_IOT_ATTACK_FLOWS)


def test_campaign_paths_are_partitioned_by_study_stratum(tmp_path: Path) -> None:
    digest = "a" * 64
    first_group = campaign_artifact_path(
        tmp_path,
        StudyStratum.UNSW_IOT_ATTACK_FLOWS_G1,
        digest,
    )
    second_group = campaign_artifact_path(
        tmp_path,
        StudyStratum.UNSW_IOT_ATTACK_FLOWS_G2,
        digest,
    )

    assert first_group.path != second_group.path
    assert first_group.path.parent.parent.name == "unsw_iot_attack_flows_g1"
    assert second_group.path.parent.parent.name == "unsw_iot_attack_flows_g2"


def test_ambiguous_dataset_name_cannot_select_a_campaign_artifact(tmp_path: Path) -> None:
    with pytest.raises(DatasetValidationError, match="select G1 or G2"):
        campaign_artifact_path(
            tmp_path,
            DatasetName.UNSW_IOT_ATTACK_FLOWS,
            "b" * 64,
        )


LAB_NETWORK = ipaddress.IPv4Network("192.168.1.0/24")


def _stub_client() -> DatasetClientData:
    rows = np.asarray([[1.0]], dtype=np.float64)
    return DatasetClientData(
        DatasetClientId(DatasetName.N_BAIOT, ClientName("stub")),
        StudyStratum.N_BAIOT,
        FeatureSchemaId.N_BAIOT_KITSUNE_SIGNED_LOG_V1,
        (FeatureName("feature"),),
        rows,
        np.asarray([0], dtype=np.int64),
        (),
    )


@pytest.mark.parametrize(
    ("dataset", "loader_name"),
    [
        (DatasetName.IOT_23, "load_iot23_clients"),
        (DatasetName.TON_IOT, "load_ton_iot_clients"),
        (DatasetName.GOTHAM_2025, "load_gotham_clients"),
        (DatasetName.CTU_13, "load_ctu13_clients"),
    ],
)
def test_single_stratum_adapters_delegate_to_their_loaders(
    dataset: DatasetName,
    loader_name: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub = (_stub_client(),)

    def fake_loader(_root: Path, *_network: LabNetwork) -> tuple[DatasetClientData, ...]:
        return stub

    monkeypatch.setattr(registry, loader_name, fake_loader)
    adapter = dataset_adapter(dataset)
    assert adapter.load_clients(tmp_path, default_study_stratum(dataset), LAB_NETWORK) == stub


def test_unsw_adapter_forwards_the_requested_group(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[StudyStratum] = []

    def fake_clients(_root: Path, group: StudyStratum | None) -> tuple[DatasetClientData, ...]:
        assert group is not None
        seen.append(group)
        return (_stub_client(),)

    def fake_sources(_root: Path, group: StudyStratum | None) -> tuple[Path, ...]:
        assert group is not None
        seen.append(group)
        return (tmp_path / "source.csv",)

    monkeypatch.setattr(registry, "load_unsw_clients", fake_clients)
    monkeypatch.setattr(registry, "unsw_source_files", fake_sources)
    adapter = dataset_adapter(DatasetName.UNSW_IOT_ATTACK_FLOWS)
    group = StudyStratum.UNSW_IOT_ATTACK_FLOWS_G2
    adapter.load_clients(tmp_path, group, LAB_NETWORK)
    assert adapter.discover_sources(tmp_path, group) == (tmp_path / "source.csv",)
    assert seen == [group, group]


def test_nbaiot_adapter_loads_every_device_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "N-BaIoT"
    (root / "device_a").mkdir(parents=True)
    (root / "device_b").mkdir()
    (root / "notes.txt").write_text("ignored", encoding="utf-8")

    def fake_device(_path: Path) -> DatasetClientData:
        return _stub_client()

    monkeypatch.setattr(registry, "load_nbaiot_device", fake_device)
    adapter = dataset_adapter(DatasetName.N_BAIOT)
    assert len(adapter.load_clients(tmp_path, StudyStratum.N_BAIOT, LAB_NETWORK)) == 2


def test_nbaiot_adapter_rejects_missing_and_empty_directories(tmp_path: Path) -> None:
    adapter = dataset_adapter(DatasetName.N_BAIOT)
    with pytest.raises(DatasetValidationError, match="Could not inspect N-BaIoT"):
        adapter.load_clients(tmp_path, StudyStratum.N_BAIOT, LAB_NETWORK)
    (tmp_path / "N-BaIoT").mkdir()
    with pytest.raises(DatasetValidationError, match="No N-BaIoT device folders"):
        adapter.load_clients(tmp_path, StudyStratum.N_BAIOT, LAB_NETWORK)


def test_dataset_load_rejects_mismatched_inputs(tmp_path: Path) -> None:
    client = _stub_client()
    source = tmp_path / "source.csv"
    with pytest.raises(ValueError, match="does not belong"):
        DatasetLoad(DatasetName.IOT_23, StudyStratum.N_BAIOT, (client,), (source,))
    with pytest.raises(ValueError, match="require clients"):
        DatasetLoad(DatasetName.N_BAIOT, StudyStratum.N_BAIOT, (), (source,))
    with pytest.raises(ValueError, match="identities must be unique"):
        DatasetLoad(DatasetName.N_BAIOT, StudyStratum.N_BAIOT, (client, client), (source,))
    with pytest.raises(ValueError, match="source files must be unique"):
        DatasetLoad(DatasetName.N_BAIOT, StudyStratum.N_BAIOT, (client,), (source, source))


def _populate_dataset_sources(data_root: Path) -> tuple[tuple[Path, bytes], ...]:
    expected_contents: list[tuple[Path, bytes]] = []
    for index, source in enumerate(DatasetName):
        location = data_root / source.source_location
        contents = f"source-{index}".encode()
        if source.location_kind is DatasetLocationKind.FILE:
            location.parent.mkdir(parents=True, exist_ok=True)
            location.write_bytes(contents)
            expected_contents.append((location, contents))
        else:
            location.mkdir(parents=True, exist_ok=True)
            data_file = location / "sample.csv"
            data_file.write_bytes(contents)
            expected_contents.append((data_file, contents))
    return tuple(expected_contents)


def test_dataset_inventory_is_complete_and_read_only(tmp_path: Path) -> None:
    data_root = tmp_path / "raw"
    contents = _populate_dataset_sources(data_root)

    report = inspect_dataset_sources(data_root)

    assert report.data_root == data_root.resolve()
    assert report.is_complete
    assert tuple(source.dataset for source in report.sources) == tuple(DatasetName)
    assert all(source.file_count == 1 for source in report.sources)
    assert tuple((path, path.read_bytes()) for path, _ in contents) == contents


def test_dataset_inventory_reports_missing_empty_and_invalid_layouts(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "raw"
    file_source, directory_source, *_ = DatasetName
    wrong_file_location = data_root / file_source.source_location
    wrong_file_location.mkdir(parents=True)
    empty_directory_location = data_root / directory_source.source_location
    empty_directory_location.mkdir(parents=True)

    report = inspect_dataset_sources(data_root)

    assert not report.is_complete
    assert report.sources[0].availability is DatasetAvailability.INVALID_LAYOUT
    assert report.sources[1].availability is DatasetAvailability.EMPTY
    assert all(source.availability is DatasetAvailability.MISSING for source in report.sources[2:])


def test_dataset_inventory_report_rejects_duplicate_or_missing_sources(
    tmp_path: Path,
) -> None:
    inventory = DatasetSourceInventory(
        dataset=DatasetName.IOT_23,
        location=tmp_path,
        availability=DatasetAvailability.MISSING,
        file_count=0,
        byte_count=0,
    )
    with pytest.raises(ValueError):
        DatasetInventoryReport(tmp_path, (inventory,))


def test_dataset_inventory_distinguishes_unreadable_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root = tmp_path / "raw"
    _populate_dataset_sources(data_root)
    inaccessible_location = data_root / next(iter(DatasetName)).source_location
    original_stat = Path.stat

    def restricted_stat(path: Path, *, follow_symlinks: bool = True) -> os.stat_result:
        if path == inaccessible_location:
            raise PermissionError(path)
        return original_stat(path, follow_symlinks=follow_symlinks)

    monkeypatch.setattr(Path, "stat", restricted_stat)

    report = inspect_dataset_sources(data_root)

    assert report.sources[0].availability is DatasetAvailability.UNREADABLE
    assert all(source.is_available for source in report.sources[1:])
