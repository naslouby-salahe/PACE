import stat
from dataclasses import dataclass
from pathlib import Path

import structlog

from pace.datasets.adapters.ctu13 import ctu13_source_files, load_ctu13_clients
from pace.datasets.adapters.gotham import gotham_source_files, load_gotham_clients
from pace.datasets.adapters.iot23 import iot23_source_files, load_iot23_clients
from pace.datasets.adapters.nbaiot import load_nbaiot_device, nbaiot_source_files
from pace.datasets.adapters.ton_iot import load_ton_iot_clients, ton_iot_source_files
from pace.datasets.adapters.unsw import load_unsw_clients, unsw_source_files
from pace.types import (
    CorpusPath,
    DatasetAvailability,
    DatasetClientData,
    DatasetClientLoader,
    DatasetInventoryReport,
    DatasetLocationKind,
    DatasetName,
    DatasetSourceDiscovery,
    DatasetSourceInventory,
    DatasetValidationError,
    LabNetwork,
    LogEvent,
    StudyStratum,
)

logger = structlog.get_logger()


@dataclass(frozen=True, slots=True)
class DatasetLoad:
    dataset: DatasetName
    study_stratum: StudyStratum
    clients: tuple[DatasetClientData, ...]
    source_files: tuple[Path, ...]

    def __post_init__(self) -> None:
        if self.study_stratum.dataset is not self.dataset:
            raise ValueError("Dataset load stratum does not belong to its dataset")
        if not self.clients or not self.source_files:
            raise ValueError("Dataset loads require clients and source files")
        if any(
            client.client.dataset is not self.dataset
            or client.study_stratum is not self.study_stratum
            for client in self.clients
        ):
            raise ValueError("Dataset load clients do not match its dataset and study stratum")
        if len({client.client for client in self.clients}) != len(self.clients):
            raise ValueError("Dataset load client identities must be unique")
        if len(set(self.source_files)) != len(self.source_files):
            raise ValueError("Dataset load source files must be unique")


@dataclass(frozen=True, slots=True)
class DatasetAdapter:
    dataset: DatasetName
    strata: tuple[StudyStratum, ...]
    load_clients: DatasetClientLoader
    discover_sources: DatasetSourceDiscovery

    def __post_init__(self) -> None:
        if not self.strata or any(stratum.dataset is not self.dataset for stratum in self.strata):
            raise ValueError("Dataset adapter must declare valid strata for its dataset")
        if len(set(self.strata)) != len(self.strata):
            raise ValueError("Dataset adapter cannot repeat study strata")


def _load_nbaiot(
    data_root: Path, _stratum: StudyStratum, _lab_network: LabNetwork
) -> tuple[DatasetClientData, ...]:
    directory = data_root / CorpusPath.N_BAIOT
    try:
        device_directories = tuple(sorted(path for path in directory.iterdir() if path.is_dir()))
    except OSError as error:
        raise DatasetValidationError(f"Could not inspect N-BaIoT data in {directory}") from error
    if not device_directories:
        raise DatasetValidationError(f"No N-BaIoT device folders were found in {directory}")
    return tuple(load_nbaiot_device(path) for path in device_directories)


def _load_iot23(
    data_root: Path, _stratum: StudyStratum, _lab_network: LabNetwork
) -> tuple[DatasetClientData, ...]:
    return load_iot23_clients(data_root)


def _load_gotham(
    data_root: Path, _stratum: StudyStratum, _lab_network: LabNetwork
) -> tuple[DatasetClientData, ...]:
    return load_gotham_clients(data_root)


def _load_ton_iot(
    data_root: Path, _stratum: StudyStratum, lab_network: LabNetwork
) -> tuple[DatasetClientData, ...]:
    return load_ton_iot_clients(data_root, lab_network)


def _load_ctu13(
    data_root: Path, _stratum: StudyStratum, _lab_network: LabNetwork
) -> tuple[DatasetClientData, ...]:
    return load_ctu13_clients(data_root)


def _load_unsw(
    data_root: Path, stratum: StudyStratum, _lab_network: LabNetwork
) -> tuple[DatasetClientData, ...]:
    return load_unsw_clients(data_root, stratum)


def _sources_nbaiot(data_root: Path, _stratum: StudyStratum) -> tuple[Path, ...]:
    return nbaiot_source_files(data_root)


def _sources_iot23(data_root: Path, _stratum: StudyStratum) -> tuple[Path, ...]:
    return iot23_source_files(data_root)


def _sources_gotham(data_root: Path, _stratum: StudyStratum) -> tuple[Path, ...]:
    return gotham_source_files(data_root)


def _sources_ton_iot(data_root: Path, _stratum: StudyStratum) -> tuple[Path, ...]:
    return ton_iot_source_files(data_root)


def _sources_ctu13(data_root: Path, _stratum: StudyStratum) -> tuple[Path, ...]:
    return ctu13_source_files(data_root)


def _sources_unsw(data_root: Path, stratum: StudyStratum) -> tuple[Path, ...]:
    return unsw_source_files(data_root, stratum)


def _registered_adapters() -> tuple[DatasetAdapter, ...]:
    return (
        DatasetAdapter(DatasetName.IOT_23, (StudyStratum.IOT_23,), _load_iot23, _sources_iot23),
        DatasetAdapter(
            DatasetName.TON_IOT, (StudyStratum.TON_IOT,), _load_ton_iot, _sources_ton_iot
        ),
        DatasetAdapter(
            DatasetName.GOTHAM_2025,
            (StudyStratum.GOTHAM_2025,),
            _load_gotham,
            _sources_gotham,
        ),
        DatasetAdapter(DatasetName.CTU_13, (StudyStratum.CTU_13,), _load_ctu13, _sources_ctu13),
        DatasetAdapter(DatasetName.N_BAIOT, (StudyStratum.N_BAIOT,), _load_nbaiot, _sources_nbaiot),
        DatasetAdapter(
            DatasetName.UNSW_IOT_ATTACK_FLOWS,
            (
                StudyStratum.UNSW_IOT_ATTACK_FLOWS_G1,
                StudyStratum.UNSW_IOT_ATTACK_FLOWS_G2,
            ),
            _load_unsw,
            _sources_unsw,
        ),
    )


def registered_strata(dataset: DatasetName) -> tuple[StudyStratum, ...]:
    return dataset_adapter(dataset).strata


def dataset_adapter(dataset: DatasetName) -> DatasetAdapter:
    try:
        return next(adapter for adapter in _registered_adapters() if adapter.dataset is dataset)
    except StopIteration as error:
        raise DatasetValidationError(
            f"No dataset adapter is registered for {dataset.name}"
        ) from error


def load_dataset(
    data_root: Path,
    lab_network: LabNetwork,
    dataset: DatasetName,
    stratum: StudyStratum | None = None,
) -> DatasetLoad:
    selected_stratum = resolve_study_stratum(dataset, stratum)
    adapter = dataset_adapter(dataset)
    logger.info(LogEvent.DATASET_LOADED, dataset=dataset.name, study_stratum=selected_stratum.name)
    clients = adapter.load_clients(data_root, selected_stratum, lab_network)
    source_files = adapter.discover_sources(data_root, selected_stratum)
    return DatasetLoad(dataset, selected_stratum, clients, source_files)


def resolve_study_stratum(
    dataset: DatasetName,
    stratum: StudyStratum | None = None,
) -> StudyStratum:
    adapter = dataset_adapter(dataset)
    if stratum is None:
        if len(adapter.strata) != 1:
            raise DatasetValidationError(
                f"{dataset.name} has multiple study strata; select one explicitly"
            )
        selected_stratum = adapter.strata[0]
    else:
        selected_stratum = stratum
    if selected_stratum not in adapter.strata:
        raise DatasetValidationError(
            f"Study stratum {selected_stratum.name} is not registered for {dataset.name}"
        )
    return selected_stratum


def dataset_source_files(
    data_root: Path,
    dataset: DatasetName,
    stratum: StudyStratum | None = None,
) -> tuple[Path, ...]:
    selected_stratum = resolve_study_stratum(dataset, stratum)
    return dataset_adapter(dataset).discover_sources(data_root, selected_stratum)


def _inspect_location(dataset: DatasetName, data_root: Path) -> DatasetSourceInventory:
    location = data_root / dataset.source_location

    def inventory(
        availability: DatasetAvailability, files: tuple[Path, ...] = ()
    ) -> DatasetSourceInventory:
        return DatasetSourceInventory(
            dataset=dataset,
            location=location,
            availability=availability,
            file_count=len(files),
            byte_count=sum(path.stat().st_size for path in files),
        )

    try:
        location_mode = location.stat().st_mode
    except FileNotFoundError:
        return inventory(DatasetAvailability.MISSING)
    except OSError:
        return inventory(DatasetAvailability.UNREADABLE)
    try:
        if dataset.location_kind is DatasetLocationKind.FILE and stat.S_ISREG(location_mode):
            files = (location,)
        elif dataset.location_kind is DatasetLocationKind.DIRECTORY and stat.S_ISDIR(location_mode):
            files = tuple(sorted(path for path in location.rglob("*") if path.is_file()))
        else:
            return inventory(DatasetAvailability.INVALID_LAYOUT)
        return inventory(
            DatasetAvailability.AVAILABLE if files else DatasetAvailability.EMPTY, files
        )
    except OSError:
        return inventory(DatasetAvailability.UNREADABLE)


def inspect_dataset_sources(data_root: Path) -> DatasetInventoryReport:
    sources = tuple(_inspect_location(dataset, data_root) for dataset in DatasetName)
    return DatasetInventoryReport(data_root.resolve(), sources)
