import csv
import io
import ipaddress
import math
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from functools import lru_cache
from math import isfinite
from pathlib import Path

import numpy as np
import polars as pl
import structlog

from pace.datasets.common import (
    extend_reservoirs,
    feature_columns,
    flagged_positions,
    has_regular_field_layout,
    ordered_reservoir_matrix,
    parse_decimal,
    polars_text,
    polars_texts,
    privileged_port_indicator,
    record_blocks,
    reservoir_columns,
)
from pace.types import (
    AttackCapture,
    AttackType,
    AttackTypeName,
    ClientName,
    ColumnIndex,
    CorpusPath,
    DatasetClientData,
    DatasetClientId,
    DatasetName,
    DatasetValidationError,
    DerivedColumn,
    EpochSeconds,
    FeatureGroup,
    FeatureName,
    FeatureSchemaId,
    FeatureValue,
    FloatArray,
    FloatMatrix,
    FloatRowReservoir,
    IPv4AddressText,
    LabNetwork,
    LogEvent,
    MissingValueToken,
    NetworkProtocolCategory,
    ObservationCount,
    RecordCommentPolicy,
    RowIndexArray,
    SamplingSeed,
    SharedFeature,
    SourceCell,
    SourceColumn,
    SourceLine,
    SourceRecord,
    StudyStratum,
    TargetCohortRole,
    TonIotBinaryLabel,
    TonIotColumn,
    TonIotSampling,
    TonIoTStudyHost,
    TonIotType,
    ZeekConnectionState,
)

logger = structlog.get_logger()


@dataclass(frozen=True, slots=True)
class _GroupKey:
    host: IPv4AddressText
    attack_type: AttackType | None

    @property
    def sort_key(
        self,
    ) -> (
        tuple[IPv4AddressText, ObservationCount]
        | tuple[IPv4AddressText, ObservationCount, AttackTypeName]
    ):
        if self.attack_type is None:
            return self.host, 0
        return self.host, 1, self.attack_type.identifier


@lru_cache(maxsize=1)
def ton_iot_feature_names() -> tuple[FeatureName, ...]:
    return (
        *(FeatureName(column) for column in TonIotColumn.numeric()),
        *(FeatureGroup.STATE.feature(state) for state in ZeekConnectionState),
        *(FeatureGroup.PROTOCOL.feature(protocol) for protocol in NetworkProtocolCategory),
        FeatureName(SharedFeature.SERVICE_KNOWN),
        *(name for column in TonIotColumn.ports() for name in column.port_features),
    )


def ton_iot_study_targets() -> tuple[TonIoTStudyHost, ...]:
    return tuple(TonIoTStudyHost)


def _bucket(capacity: ObservationCount) -> FloatRowReservoir:
    return FloatRowReservoir(capacity, len(ton_iot_feature_names()))


@lru_cache(maxsize=16)
def normalize_ton_iot_attack_type(label: SourceCell) -> AttackType:
    normalized = SourceCell(label.strip().lower())
    known = TonIotType.known(normalized)
    if known is TonIotType.NORMAL:
        raise DatasetValidationError("The benign TON-IoT label is not an attack type")
    if known is not None and known.attack_type_name is not None:
        return AttackType(known.attack_type_name)
    return AttackType(AttackTypeName(normalized.upper()))


def ton_iot_source_files(data_root: Path) -> tuple[Path, ...]:
    if (data_root / CorpusPath.TON_IOT_PROCESSED).is_dir():
        root = data_root / CorpusPath.TON_IOT_PROCESSED
    elif (data_root / CorpusPath.TON_IOT_PROCESSED_DIRECT).is_dir():
        root = data_root / CorpusPath.TON_IOT_PROCESSED_DIRECT
    else:
        root = data_root
    if not root.is_dir():
        raise DatasetValidationError(f"TON-IoT network data directory is unavailable: {root}")
    files = tuple(sorted(root.glob("Network_dataset_*.csv")))
    if not files:
        raise DatasetValidationError(f"No TON-IoT network CSV files were found in {root}")
    return files


@lru_cache(maxsize=256)
def _state_vector(state: SourceCell) -> tuple[FeatureValue, ...]:
    return tuple(FeatureValue(state == item) for item in ZeekConnectionState)


@lru_cache(maxsize=256)
def _protocol_vector(protocol: SourceCell) -> tuple[FeatureValue, ...]:
    return tuple(FeatureValue(protocol == item) for item in NetworkProtocolCategory)


def _cell_number(record: SourceRecord, column: TonIotColumn) -> FeatureValue:
    raw = SourceCell(record.cell(column).strip())
    return FeatureValue(0.0) if raw in MissingValueToken.numeric() else parse_decimal(raw)


def _ton_iot_features(record: SourceRecord) -> list[FeatureValue]:
    values: list[FeatureValue] = []
    for column in TonIotColumn.numeric():
        value = _cell_number(record, column)
        if not isfinite(value):
            raise ValueError("non-finite numeric feature")
        values.append(value)
    values.extend(_state_vector(SourceCell(record.cell(TonIotColumn.CONNECTION_STATE).strip())))
    values.extend(_protocol_vector(SourceCell(record.cell(TonIotColumn.PROTOCOL).strip().lower())))
    values.append(
        FeatureValue(record.cell(TonIotColumn.SERVICE).strip() not in MissingValueToken.numeric())
    )
    for column in TonIotColumn.ports():
        port = _cell_number(record, column)
        if not isfinite(port):
            raise ValueError("non-finite port")
        values.extend((FeatureValue(0.0 < port < 1024.0), port))
    return values


def _selected_hosts(
    client_identities: Iterable[TonIoTStudyHost | IPv4AddressText], lab_network: LabNetwork
) -> frozenset[IPv4AddressText]:
    selected = frozenset(
        identity.host_ip(lab_network) if isinstance(identity, TonIoTStudyHost) else identity
        for identity in client_identities
    )
    try:
        invalid = tuple(
            identity for identity in selected if ipaddress.ip_address(identity) not in lab_network
        )
    except ValueError as error:
        raise DatasetValidationError(
            "TON-IoT client identities must be LAN IPv4 addresses"
        ) from error
    if invalid:
        raise DatasetValidationError(f"TON-IoT identities are outside the lab LAN: {invalid}")
    return selected


def _required_fields() -> set[SourceColumn]:
    return {SourceColumn(column) for column in TonIotColumn}


def _parse_selected_record(
    record: SourceRecord, endpoint: IPv4AddressText, source_file: Path
) -> tuple[EpochSeconds, list[FeatureValue], AttackType | None]:
    label = SourceCell(record.cell(TonIotColumn.TYPE).strip().lower())
    known = TonIotType.known(label)
    is_attack = known is not TonIotType.NORMAL
    try:
        if known is None:
            raise DatasetValidationError(
                f"{source_file} has unsupported selected TON-IoT type {label!r}"
            )
        if record.cell(TonIotColumn.LABEL).strip() != known.binary_label:
            raise DatasetValidationError(
                f"{source_file} has inconsistent label/type values for {endpoint}"
            )
        timestamp = parse_decimal(record.cell(TonIotColumn.TIMESTAMP))
        if not isfinite(timestamp):
            raise ValueError("non-finite timestamp")
        features = _ton_iot_features(record)
    except (TypeError, ValueError) as error:
        raise DatasetValidationError(
            f"{source_file} has malformed selected numeric values for {endpoint}"
        ) from error
    attack_type = normalize_ton_iot_attack_type(label) if is_attack else None
    return EpochSeconds(timestamp), features, attack_type


def _attack_type_or_none(label: SourceCell) -> AttackType | None:
    known = TonIotType.known(label)
    if known is None or known is TonIotType.NORMAL:
        return None
    return normalize_ton_iot_attack_type(label)


def _decimal(column: TonIotColumn) -> pl.Expr:
    text = pl.col(column).str.strip_chars()
    return (
        pl.when(text.is_in(polars_texts(MissingValueToken.numeric())))
        .then(pl.lit(0.0))
        .otherwise(text.cast(pl.Float64, strict=False))
    )


def _feature_expressions() -> list[pl.Expr]:
    state = pl.col(TonIotColumn.CONNECTION_STATE).str.strip_chars()
    protocol = pl.col(TonIotColumn.PROTOCOL).str.strip_chars().str.to_lowercase()
    ports = [_decimal(column) for column in TonIotColumn.ports()]
    return [
        *(_decimal(column) for column in TonIotColumn.numeric()),
        *((state == polars_text(item)).cast(pl.Float64) for item in ZeekConnectionState),
        *((protocol == polars_text(item)).cast(pl.Float64) for item in NetworkProtocolCategory),
        (
            ~pl.col(TonIotColumn.SERVICE)
            .str.strip_chars()
            .is_in(polars_texts(MissingValueToken.numeric()))
        ).cast(pl.Float64),
        *(expression for port in ports for expression in (privileged_port_indicator(port), port)),
    ]


def _parsed_block(
    subset: pl.DataFrame, hosts: list[IPv4AddressText], source_file: Path
) -> tuple[FloatArray, FloatMatrix, list[AttackType | None]]:
    label = pl.col(TonIotColumn.TYPE).str.strip_chars().str.to_lowercase()
    columns = feature_columns(len(ton_iot_feature_names()))
    parsed = subset.select(
        label.alias(DerivedColumn.LABEL),
        pl.col(TonIotColumn.TIMESTAMP)
        .str.strip_chars()
        .cast(pl.Float64, strict=False)
        .alias(DerivedColumn.TIMESTAMP),
        label.is_in(polars_texts(TonIotType)).alias(DerivedColumn.SUPPORTED),
        (
            pl.col(TonIotColumn.LABEL).str.strip_chars()
            == pl.when(label != polars_text(TonIotType.NORMAL))
            .then(pl.lit(polars_text(TonIotBinaryLabel.ATTACK)))
            .otherwise(pl.lit(polars_text(TonIotBinaryLabel.NORMAL)))
        ).alias(DerivedColumn.CONSISTENT),
        *(
            expression.alias(column)
            for expression, column in zip(_feature_expressions(), columns, strict=True)
        ),
    )
    usable = (
        parsed[DerivedColumn.SUPPORTED].to_numpy()
        & parsed[DerivedColumn.CONSISTENT].to_numpy()
        & parsed.select(
            pl.all_horizontal(
                pl.col(DerivedColumn.TIMESTAMP, *columns).is_finite().fill_null(False)
            )
        )
        .to_series()
        .to_numpy()
    )
    labels = parsed[DerivedColumn.LABEL].to_list()
    timestamps = parsed[DerivedColumn.TIMESTAMP].fill_null(0.0).to_numpy().copy()
    features = parsed.select(columns).to_numpy().copy()
    attack_types: list[AttackType | None] = [
        _attack_type_or_none(SourceCell(name)) for name in labels
    ]
    for index in flagged_positions(~usable):
        record = SourceRecord(
            {
                SourceColumn(column): SourceCell(cell)
                for column, cell in zip(subset.columns, subset.row(index), strict=True)
            }
        )
        timestamps[index], values, attack_types[index] = _parse_selected_record(
            record, hosts[index], source_file
        )
        features[index] = values
    return timestamps, features, attack_types


class _TonIoTAccumulator:
    def __init__(self, selected: frozenset[IPv4AddressText], sampling: TonIotSampling) -> None:
        self.selected = selected
        self.sampling = sampling
        self.random = np.random.default_rng(SamplingSeed.TON_IOT)
        self.benign_rows: dict[IPv4AddressText, FloatRowReservoir] = {}
        self.attack_rows: dict[IPv4AddressText, dict[AttackType, FloatRowReservoir]] = defaultdict(
            dict
        )
        self.benign_population: defaultdict[IPv4AddressText, ObservationCount] = defaultdict(
            lambda: 0
        )
        self.attack_population: defaultdict[
            IPv4AddressText, defaultdict[AttackType, ObservationCount]
        ] = defaultdict(lambda: defaultdict(lambda: 0))
        self.sequence = 0

    def _retain(self, key: _GroupKey, chunk: FloatRowReservoir) -> None:
        population = chunk.source_population_rows
        if key.attack_type is None:
            self.benign_population[key.host] = self.benign_population[key.host] + population
        else:
            self.attack_population[key.host][key.attack_type] = (
                self.attack_population[key.host][key.attack_type] + population
            )
        retained = (
            math.floor(population * self.sampling.chunk_keep_fraction)
            if population > self.sampling.chunk_group_threshold
            else population
        )
        if retained < len(chunk.timestamps):
            positions = np.sort(self.random.choice(len(chunk.timestamps), retained, replace=False))
        else:
            positions = np.arange(len(chunk.timestamps), dtype=np.int64)
        if positions.size == 0:
            return
        timestamps, source_rows, features = reservoir_columns((chunk,))
        extend_reservoirs(
            [self._destination(key)],
            np.zeros(positions.size, dtype=np.int64),
            timestamps[positions],
            source_rows[positions],
            features[positions],
            self.random,
        )

    def _destination(self, key: _GroupKey) -> FloatRowReservoir:
        if key.attack_type is None:
            if key.host not in self.benign_rows:
                self.benign_rows[key.host] = _bucket(self.sampling.benign_cap)
            return self.benign_rows[key.host]
        if key.attack_type not in self.attack_rows[key.host]:
            self.attack_rows[key.host][key.attack_type] = _bucket(
                self.sampling.attack_cap_per_label
            )
        return self.attack_rows[key.host][key.attack_type]

    def flush(self, chunk_groups: dict[_GroupKey, FloatRowReservoir]) -> None:
        for key, chunk in sorted(chunk_groups.items(), key=lambda item: item[0].sort_key):
            self._retain(key, chunk)
        chunk_groups.clear()

    def _frame(
        self,
        block: list[SourceLine],
        header: list[SourceColumn],
        columns: tuple[ColumnIndex, ColumnIndex],
        source_file: Path,
    ) -> tuple[pl.DataFrame, RowIndexArray]:
        sequence = self.sequence + np.arange(len(block), dtype=np.int64)
        if has_regular_field_layout(block, len(header)):
            try:
                frame = pl.read_csv(
                    io.BytesIO(b"\n".join(block)),
                    has_header=False,
                    new_columns=header,
                    infer_schema=False,
                ).fill_null("")
            except pl.exceptions.PolarsError:
                logger.warning(
                    LogEvent.ADAPTER_SOURCE_READ,
                    dataset=DatasetName.TON_IOT.name,
                    source_file=source_file.name,
                    fallback="record-wise parsing",
                )
            else:
                return frame, sequence
        source_column, destination_column = columns
        parsed = list(enumerate(csv.reader(line.decode() for line in block)))
        complete = [(position, row) for position, row in parsed if len(row) == len(header)]
        for _, row in parsed:
            if len(row) == len(header):
                continue
            addresses = (
                (row[column] if len(row) > column else "").strip()
                for column in (source_column, destination_column)
            )
            if any(address in self.selected for address in addresses):
                raise DatasetValidationError(f"{source_file} has a malformed selected TON-IoT row")
        frame = pl.DataFrame(
            [row for _, row in complete],
            schema=dict.fromkeys(header, pl.String),
            orient="row",
        )
        return frame, sequence[np.asarray([position for position, _ in complete], dtype=np.int64)]

    def _stage_block(
        self,
        frame: pl.DataFrame,
        sequence: RowIndexArray,
        source_file: Path,
        chunk_groups: dict[_GroupKey, FloatRowReservoir],
    ) -> None:
        addresses = frame[TonIotColumn.DESTINATION_IP].str.strip_chars()
        chosen = np.flatnonzero(addresses.is_in(sorted(self.selected)).to_numpy())
        if chosen.size == 0:
            return
        subset = frame[chosen]
        hosts = [IPv4AddressText(host) for host in addresses.gather(chosen).to_list()]
        timestamps, features, attack_types = _parsed_block(subset, hosts, source_file)
        self._stage_groups(
            chunk_groups, hosts, attack_types, sequence[chosen], timestamps, features
        )

    def _stage_groups(
        self,
        chunk_groups: dict[_GroupKey, FloatRowReservoir],
        hosts: list[IPv4AddressText],
        attack_types: list[AttackType | None],
        sequence: RowIndexArray,
        timestamps: FloatArray,
        features: FloatMatrix,
    ) -> None:
        groups: dict[_GroupKey, ObservationCount] = {}
        buckets: list[FloatRowReservoir] = []
        ids = np.empty(len(hosts), dtype=np.int64)
        for position, key in enumerate(
            _GroupKey(host, attack_type)
            for host, attack_type in zip(hosts, attack_types, strict=True)
        ):
            if key not in groups:
                groups[key] = len(buckets)
                buckets.append(
                    chunk_groups.setdefault(key, _bucket(self.sampling.chunk_reservoir_cap))
                )
            ids[position] = groups[key]
        extend_reservoirs(buckets, ids, timestamps, sequence, features, self.random)

    def read(self, source_file: Path, required: set[SourceColumn]) -> None:
        try:
            with source_file.open("rb") as stream:
                text = stream.readline().removeprefix(b"\xef\xbb\xbf").decode()
                header = [SourceColumn(name) for name in next(csv.reader([text]), ())]
                if missing := required.difference(header):
                    raise DatasetValidationError(
                        f"{source_file} is missing TON-IoT fields: {sorted(missing)}"
                    )
                columns = (
                    ColumnIndex(header.index(SourceColumn(TonIotColumn.SOURCE_IP))),
                    ColumnIndex(header.index(SourceColumn(TonIotColumn.DESTINATION_IP))),
                )
                for block in record_blocks(
                    stream, self.sampling.source_chunk_rows, RecordCommentPolicy.NONE
                ):
                    frame, sequence = self._frame(block, header, columns, source_file)
                    chunk_groups: dict[_GroupKey, FloatRowReservoir] = {}
                    self._stage_block(frame, sequence, source_file, chunk_groups)
                    self.flush(chunk_groups)
                    self.sequence += len(block)
        except OSError as error:
            raise DatasetValidationError(f"Could not read TON-IoT source {source_file}") from error
        logger.info(
            LogEvent.ADAPTER_SOURCE_READ,
            dataset=DatasetName.TON_IOT.name,
            source_file=source_file.name,
            row_count=self.sequence,
        )

    def clients(self, source_directory: Path) -> tuple[DatasetClientData, ...]:
        clients: list[DatasetClientData] = []
        for host in sorted(set(self.benign_rows).intersection(self.attack_rows)):
            benign_features, benign_order = ordered_reservoir_matrix((self.benign_rows[host],))
            captures: list[AttackCapture] = []
            for attack_type in sorted(self.attack_rows[host], key=lambda item: item.identifier):
                attack_features, source_order = ordered_reservoir_matrix(
                    (self.attack_rows[host][attack_type],)
                )
                captures.append(
                    AttackCapture(
                        attack_type,
                        source_directory,
                        attack_features,
                        source_order,
                        self.attack_population[host][attack_type],
                    )
                )
            clients.append(
                DatasetClientData(
                    DatasetClientId(DatasetName.TON_IOT, ClientName(host)),
                    StudyStratum.TON_IOT,
                    FeatureSchemaId.TON_IOT_ZEEK_ENGINEERED_V1,
                    ton_iot_feature_names(),
                    benign_features,
                    benign_order,
                    tuple(captures),
                    TargetCohortRole.REPORTED_TARGET,
                    self.benign_population[host],
                )
            )
        return tuple(clients)


def load_ton_iot_clients(
    data_root: Path,
    lab_network: LabNetwork,
    client_identities: Iterable[TonIoTStudyHost | IPv4AddressText] | None = None,
    sampling: TonIotSampling | None = None,
) -> tuple[DatasetClientData, ...]:
    source_files = ton_iot_source_files(data_root)
    logger.info(
        LogEvent.DATASET_LOADED, dataset=DatasetName.TON_IOT.name, source_count=len(source_files)
    )
    accumulator = _TonIoTAccumulator(
        _selected_hosts(
            ton_iot_study_targets() if client_identities is None else client_identities,
            lab_network,
        ),
        sampling or TonIotSampling.standard(),
    )
    required = _required_fields()
    for source_file in source_files:
        accumulator.read(source_file, required)
    clients = accumulator.clients(source_files[0].parent)
    if not clients:
        raise DatasetValidationError(
            "TON-IoT selection produced no clients with benign and attack rows"
        )
    return clients
