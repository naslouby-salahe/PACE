import shutil
import tempfile
from collections.abc import Callable
from pathlib import Path

import numpy as np
import structlog
from pydantic import ValidationError
from structlog.stdlib import BoundLogger

import pace
from pace import types as pace_types
from pace.config import ResolvedConfiguration
from pace.types import (
    ArtifactPayload,
    AttackCapture,
    AttackType,
    AttackTypeName,
    CacheFileName,
    CacheSlot,
    ClientName,
    DatasetClientData,
    DatasetClientId,
    DatasetName,
    DatasetValidationError,
    FeatureName,
    FeatureSchemaId,
    FloatMatrix,
    FrozenStrictModel,
    LabNetwork,
    LengthPrefixedHasher,
    LogEvent,
    ObservationCount,
    OutputDirectory,
    RowIndexArray,
    ScientificDigest,
    StudyStratum,
    TargetCohortRole,
    TextEncoding,
)


class CachedCapture(FrozenStrictModel, frozen=True):
    attack_type: AttackTypeName
    source_file: Path
    source_population_rows: ObservationCount | None


class CachedClient(FrozenStrictModel, frozen=True):
    dataset: DatasetName
    client: ClientName
    study_stratum: StudyStratum
    feature_schema: FeatureSchemaId
    feature_names: tuple[FeatureName, ...]
    target_cohort_role: TargetCohortRole
    benign_source_population_rows: ObservationCount | None
    captures: tuple[CachedCapture, ...]


class CachedClients(FrozenStrictModel, frozen=True):
    clients: tuple[CachedClient, ...]


logger = structlog.get_logger()


def _benign_path(directory: Path, client: CacheSlot) -> Path:
    return directory / f"{client}.{CacheFileName.BENIGN_FEATURES}.npy"


def _order_path(directory: Path, client: CacheSlot) -> Path:
    return directory / f"{client}.{CacheFileName.SOURCE_ORDER}.npy"


def _capture_features_path(directory: Path, client: CacheSlot, capture: CacheSlot) -> Path:
    return directory / f"{client}.{capture}.{CacheFileName.CAPTURE_FEATURES}.npy"


def _capture_order_path(directory: Path, client: CacheSlot, capture: CacheSlot) -> Path:
    return directory / f"{client}.{capture}.{CacheFileName.SOURCE_ORDER}.npy"


def _load_matrix(path: Path) -> FloatMatrix:
    return np.load(path)


def _load_order(path: Path) -> RowIndexArray:
    return np.load(path)


def _digest_loader_code() -> ScientificDigest:
    package_root = Path(pace.__file__).resolve().parent
    hasher = LengthPrefixedHasher()
    for path in (
        *sorted(Path(__file__).resolve().parent.rglob("*.py")),
        Path(pace_types.__file__).resolve(),
    ):
        hasher.framed_path(path.relative_to(package_root))
        hasher.framed(ArtifactPayload(path.read_bytes()))
    return hasher.hexdigest()


def cache_key(
    dataset: DatasetName,
    stratum: StudyStratum,
    input_digest: ScientificDigest,
    lab_network: LabNetwork,
) -> ScientificDigest:
    hasher = LengthPrefixedHasher()
    hasher.framed_member(dataset)
    hasher.framed_member(stratum)
    hasher.framed_label(input_digest)
    hasher.framed(ArtifactPayload(lab_network.with_prefixlen.encode(TextEncoding.UTF8)))
    hasher.framed_label(_digest_loader_code())
    return hasher.hexdigest()


def store_clients(
    cache_root: Path, key: ScientificDigest, clients: tuple[DatasetClientData, ...]
) -> None:
    cache_root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(dir=cache_root, prefix=CacheFileName.STAGING))
    try:
        for index, client in enumerate(clients):
            slot = CacheSlot(index)
            np.save(_benign_path(staging, slot), client.benign_features)
            np.save(_order_path(staging, slot), client.benign_source_order)
            for capture_index, capture in enumerate(client.attack_captures):
                capture_slot = CacheSlot(capture_index)
                np.save(_capture_features_path(staging, slot, capture_slot), capture.features)
                np.save(_capture_order_path(staging, slot, capture_slot), capture.source_order)
        manifest = CachedClients(
            clients=tuple(
                CachedClient(
                    dataset=client.client.dataset,
                    client=client.client.name,
                    study_stratum=client.study_stratum,
                    feature_schema=client.feature_schema,
                    feature_names=client.feature_names,
                    target_cohort_role=client.target_cohort_role,
                    benign_source_population_rows=client.benign_source_population_rows,
                    captures=tuple(
                        CachedCapture(
                            attack_type=capture.attack_type.identifier,
                            source_file=capture.source_file,
                            source_population_rows=capture.source_population_rows,
                        )
                        for capture in client.attack_captures
                    ),
                )
                for client in clients
            )
        )
        (staging / CacheFileName.MANIFEST).write_text(
            manifest.model_dump_json(), encoding=TextEncoding.UTF8
        )
        destination = cache_root / key
        shutil.rmtree(destination, ignore_errors=True)
        staging.replace(destination)
        logger.info(LogEvent.CLIENT_CACHE_STORED, cache_key=key, client_count=len(clients))
        prune_superseded(cache_root, key, clients[0].study_stratum)
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _cached_stratum(directory: Path) -> StudyStratum | None:
    try:
        manifest = CachedClients.model_validate_json(
            (directory / CacheFileName.MANIFEST).read_bytes(), strict=True
        )
    except (OSError, ValueError, ValidationError):
        return None
    return manifest.clients[0].study_stratum if manifest.clients else None


def prune_superseded(cache_root: Path, key: ScientificDigest, stratum: StudyStratum) -> None:
    for entry in sorted(cache_root.iterdir()):
        if (
            entry.name == key
            or entry.name.startswith(CacheFileName.STAGING)
            or _cached_stratum(entry) is not stratum
        ):
            continue
        shutil.rmtree(entry, ignore_errors=True)
        logger.info(LogEvent.CLIENT_CACHE_PRUNED, cache_key=entry.name, stratum=stratum.name)


def load_cached_clients(
    cache_root: Path, key: ScientificDigest
) -> tuple[DatasetClientData, ...] | None:
    directory = cache_root / key

    try:
        manifest = CachedClients.model_validate_json(
            (directory / CacheFileName.MANIFEST).read_bytes(), strict=True
        )
        return tuple(
            DatasetClientData(
                DatasetClientId(entry.dataset, entry.client),
                entry.study_stratum,
                entry.feature_schema,
                entry.feature_names,
                _load_matrix(_benign_path(directory, CacheSlot(index))),
                _load_order(_order_path(directory, CacheSlot(index))),
                tuple(
                    AttackCapture(
                        AttackType(capture.attack_type),
                        capture.source_file,
                        _load_matrix(
                            _capture_features_path(
                                directory, CacheSlot(index), CacheSlot(capture_index)
                            )
                        ),
                        _load_order(
                            _capture_order_path(
                                directory, CacheSlot(index), CacheSlot(capture_index)
                            )
                        ),
                        capture.source_population_rows,
                    )
                    for capture_index, capture in enumerate(entry.captures)
                ),
                entry.target_cohort_role,
                entry.benign_source_population_rows,
            )
            for index, entry in enumerate(manifest.clients)
        )
    except (OSError, ValueError, ValidationError) as error:
        logger.warning(
            LogEvent.CLIENT_CACHE_REJECTED, cache_key=key, failure_type=type(error).__name__
        )
        return None


def validate_dataset_clients(clients: tuple[DatasetClientData, ...]) -> DatasetName:
    if not clients:
        raise DatasetValidationError("Dataset campaign requires at least one client")
    client_ids = tuple(client.client for client in clients)
    if len(set(client_ids)) != len(client_ids):
        raise DatasetValidationError("Dataset client identifiers must be unique")
    datasets = {client.dataset for client in client_ids}
    if len(datasets) != 1:
        raise DatasetValidationError("A campaign must use clients from exactly one dataset")
    strata = {client.study_stratum for client in clients}
    if len(strata) != 1:
        raise DatasetValidationError("A campaign must use clients from exactly one study stratum")
    schemas = {client.feature_schema for client in clients}
    if len(schemas) != 1:
        raise DatasetValidationError("A campaign must use exactly one feature schema")
    feature_schema = clients[0].feature_names
    if any(client.feature_names != feature_schema for client in clients):
        raise DatasetValidationError("Clients in one dataset must share a feature schema")
    return next(iter(datasets))


def resolve_clients(
    clients: tuple[DatasetClientData, ...] | Callable[[], tuple[DatasetClientData, ...]],
    configuration: ResolvedConfiguration,
    identity: tuple[DatasetName, StudyStratum],
    input_digest: ScientificDigest,
    logger: BoundLogger,
) -> tuple[DatasetClientData, ...]:
    dataset, stratum = identity
    cache_root: Path = (
        configuration.runtime.output_root / OutputDirectory.CACHE / OutputDirectory.CLIENTS
    )
    key = cache_key(dataset, stratum, input_digest, configuration.runtime.lab_network)
    cached = load_cached_clients(cache_root, key) if callable(clients) else None
    if callable(clients):
        logger.info(
            LogEvent.CLIENT_CACHE_MISS if cached is None else LogEvent.CLIENT_CACHE_HIT,
            cache_key=key,
        )
    if cached is not None:
        resolved = cached
    elif callable(clients):
        resolved = clients()
    else:
        resolved = clients
    if (
        validate_dataset_clients(resolved) is not dataset
        or resolved[0].study_stratum is not stratum
    ):
        raise DatasetValidationError("Loaded client data differs from the campaign identity")
    if callable(clients) and cached is None:
        store_clients(cache_root, key, resolved)
    return resolved
