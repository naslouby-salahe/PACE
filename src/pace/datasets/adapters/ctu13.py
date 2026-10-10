from __future__ import annotations

import io
import re
import tarfile
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import lru_cache
from math import isfinite
from pathlib import Path

import indexed_bzip2
import numpy as np
import polars as pl
import structlog

from pace.datasets.common import (
    decompression_workers,
    extend_reservoirs,
    feature_columns,
    first_misaligned_line,
    flagged_positions,
    is_privileged_port,
    ordered_reservoir_matrix,
    parse_integer,
    parse_numeric_literal,
    polars_text,
    polars_texts,
    privileged_port_indicator,
    read_ahead,
    record_blocks,
    target_cohort_role,
)
from pace.types import (
    AttackCapture,
    AttackType,
    AttackTypeName,
    ClientName,
    CorpusPath,
    Ctu13BotnetLabel,
    Ctu13Column,
    Ctu13LabelMarker,
    Ctu13Scenario,
    CtuArgusState,
    CtuFlowDirection,
    CtuRootAlias,
    CtuSourceDirectionToken,
    CtuSourceSuffix,
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
    FieldDelimiter,
    FloatMatrix,
    FloatRowReservoir,
    LogEvent,
    MissingValueToken,
    NetworkProtocolCategory,
    ObservationCount,
    RecordBlockRows,
    RecordCommentPolicy,
    ReservoirCaps,
    ReservoirGroup,
    RowIndexArray,
    SamplingSeed,
    ScenarioNumber,
    SourceBinaryStream,
    SourceCell,
    SourceColumn,
    SourceLine,
    SourceRecord,
    SourceRowNumber,
    StudyStratum,
    TimeScale,
)

logger = structlog.get_logger()

_NON_ALPHANUMERIC = re.compile(r"[^a-z0-9]+")
_BOTNET_VERSION = re.compile(r"botnet-v?([a-z0-9]+)")
_SCENARIO_COMPONENT = re.compile(r"(?:scenario[-_ ]?)?(\d+)", re.IGNORECASE)
_SCENARIO_STEM = re.compile(r"(?:scenario|capture)[-_ ]?(\d+)", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class _FlowLabel:
    group: ReservoirGroup
    attack_type: AttackType | None


@lru_cache(maxsize=1)
def ctu13_feature_names() -> tuple[FeatureName, ...]:
    return (
        *(column.feature for column in Ctu13Column.numeric()),
        *(FeatureGroup.PROTOCOL.feature(protocol) for protocol in NetworkProtocolCategory),
        *(FeatureGroup.DIRECTION.feature(direction) for direction in CtuFlowDirection),
        *(FeatureGroup.STATE.feature(state) for state in CtuArgusState),
        *(name for column in Ctu13Column.ports() for name in column.port_features),
    )


def _bucket(capacity: ObservationCount) -> FloatRowReservoir:
    return FloatRowReservoir(capacity, len(ctu13_feature_names()))


def _epoch_seconds(timestamp: datetime) -> EpochSeconds:
    elapsed = timestamp - datetime.fromtimestamp(0, UTC).replace(tzinfo=None)
    return EpochSeconds(elapsed.days * 86400 + elapsed.seconds + elapsed.microseconds / 1e6)


@lru_cache(maxsize=512)
def _normalized_label(value: SourceCell) -> SourceCell:
    return SourceCell(_NON_ALPHANUMERIC.sub("-", value.strip().lower()).strip("-"))


@lru_cache(maxsize=512)
def _classify_label(value: SourceCell) -> _FlowLabel:
    normalized = _normalized_label(value)
    skipped = _FlowLabel(ReservoirGroup.SKIPPED, None)
    if not normalized:
        return skipped
    if Ctu13LabelMarker.NORMAL in normalized:
        return _FlowLabel(ReservoirGroup.BENIGN, None)
    if Ctu13LabelMarker.BACKGROUND in normalized:
        return skipped
    is_command_and_control = Ctu13LabelMarker.COMMAND_AND_CONTROL in normalized
    if Ctu13LabelMarker.BOTNET not in normalized and not is_command_and_control:
        return skipped
    version = _BOTNET_VERSION.search(normalized)
    if version:
        family = f"{Ctu13BotnetLabel.BOTNET}_V{version.group(1).upper()}"
    elif is_command_and_control:
        family = Ctu13BotnetLabel.COMMAND_AND_CONTROL
    else:
        family = Ctu13BotnetLabel.BOTNET
    return _FlowLabel(ReservoirGroup.ATTACK, AttackType(AttackTypeName(family)))


def _timestamp(value: SourceCell, row_number: SourceRowNumber) -> datetime:
    text = value.strip().replace("/", "-", 2)
    try:
        timestamp = datetime.fromisoformat(text)
        if timestamp.tzinfo is not None:
            raise ValueError("CTU-13 flow timestamps must be timezone-naive")
        return timestamp
    except ValueError as error:
        raise DatasetValidationError(
            f"Invalid CTU-13 flow timestamp at source row {row_number}"
        ) from error


def _numeric(value: SourceCell, column: Ctu13Column, row_number: SourceRowNumber) -> FeatureValue:
    result = parse_numeric_literal(value)
    if result is None:
        raise DatasetValidationError(
            f"Invalid numeric CTU-13 value in {column} at source row {row_number}"
        )
    if not isfinite(result):
        raise DatasetValidationError(
            f"Non-finite CTU-13 value in {column} at source row {row_number}"
        )
    return result


def _features(record: SourceRecord, row_number: SourceRowNumber) -> tuple[FeatureValue, ...]:
    protocol = record.cell(Ctu13Column.PROTOCOL).strip().lower()
    source_direction = record.cell(Ctu13Column.DIRECTION).strip()
    try:
        direction = CtuSourceDirectionToken(source_direction).direction
    except ValueError as error:
        raise DatasetValidationError(
            f"Unmapped CTU-13 flow direction {source_direction!r} at source row {row_number}"
        ) from error
    values = [_numeric(record.cell(column), column, row_number) for column in Ctu13Column.numeric()]
    values.extend(FeatureValue(protocol == item) for item in NetworkProtocolCategory)
    values.extend(FeatureValue(direction is item) for item in CtuFlowDirection)
    state = record.cell(Ctu13Column.STATE).strip()
    values.extend(FeatureValue(state == item) for item in CtuArgusState)
    for column in Ctu13Column.ports():
        port = _numeric(record.cell(column), column, row_number)
        values.extend((is_privileged_port(port), port))
    return tuple(values)


def _scenario_number(source_file: Path) -> ScenarioNumber | None:
    for component in reversed(source_file.parts[:-1]):
        match = _SCENARIO_COMPONENT.fullmatch(component)
        if match:
            return ScenarioNumber(parse_integer(SourceCell(match.group(1))))
    match = _SCENARIO_STEM.search(source_file.stem)
    return ScenarioNumber(parse_integer(SourceCell(match.group(1)))) if match else None


def _client_identity(source_file: Path) -> ClientName:
    number = _scenario_number(source_file)
    if number is None:
        return ClientName(source_file.stem)
    return ClientName(f"CTU13-scenario-{number:02d}")


def _scenario_attack_type(source_path: Path) -> AttackType:
    number = _scenario_number(source_path)
    if number is None or number >= 100:
        raise DatasetValidationError(
            f"Could not map {source_path} to an official CTU-13 scenario family"
        )
    try:
        scenario = Ctu13Scenario(f"{number:02d}")
    except ValueError as error:
        raise DatasetValidationError(
            f"CTU-13 scenario {number:02d} is not an official scenario"
        ) from error
    return AttackType(AttackTypeName(scenario.attack_family))


def _number(column: Ctu13Column) -> pl.Expr:
    text = pl.col(column).str.strip_chars()
    return (
        pl.when(text.is_in(polars_texts(MissingValueToken.numeric())))
        .then(pl.lit(0.0))
        .otherwise(text.cast(pl.Float64, strict=False))
    )


def _direction_expression() -> pl.Expr:
    mapping = {
        polars_text(token): polars_text(token.direction) for token in CtuSourceDirectionToken
    }
    return pl.col(Ctu13Column.DIRECTION).str.strip_chars().replace_strict(mapping, default=None)


def _feature_expressions() -> tuple[pl.Expr, ...]:
    direction = _direction_expression()
    protocol = pl.col(Ctu13Column.PROTOCOL).str.strip_chars().str.to_lowercase()
    state = pl.col(Ctu13Column.STATE).str.strip_chars()
    ports = [_number(column) for column in Ctu13Column.ports()]
    return (
        *(_number(column) for column in Ctu13Column.numeric()),
        *((protocol == polars_text(item)).cast(pl.Float64) for item in NetworkProtocolCategory),
        *((direction == polars_text(item)).cast(pl.Float64) for item in CtuFlowDirection),
        *((state == polars_text(item)).cast(pl.Float64) for item in CtuArgusState),
        *(expression for port in ports for expression in (privileged_port_indicator(port), port)),
    )


def _block_columns(frame: pl.DataFrame, first_row: SourceRowNumber) -> pl.DataFrame:
    groups = {
        label: _classify_label(SourceCell(label)).group
        for label in frame[Ctu13Column.LABEL].unique().to_list()
    }
    direction = _direction_expression()
    features = _feature_expressions()
    columns = feature_columns(len(ctu13_feature_names()))
    started = (
        pl.col(Ctu13Column.START_TIME)
        .str.strip_chars()
        .str.replace("/", "-", n=2, literal=True)
        .str.to_datetime(format="%Y-%m-%d %H:%M:%S%.f", time_unit="us", strict=False)
        .dt.epoch("us")
    )
    computed = frame.select(
        (pl.int_range(pl.len(), dtype=pl.Int64) + first_row).alias(DerivedColumn.ROW),
        pl.col(Ctu13Column.LABEL)
        .replace_strict(groups, default=ReservoirGroup.SKIPPED, return_dtype=pl.Int64)
        .alias(DerivedColumn.KIND),
        started.alias(DerivedColumn.STARTED),
        direction.is_not_null().alias(DerivedColumn.KNOWN_DIRECTION),
        *(feature.alias(column) for feature, column in zip(features, columns, strict=True)),
    )
    return computed.with_columns(
        (
            pl.col(DerivedColumn.STARTED).is_not_null()
            & pl.col(DerivedColumn.KNOWN_DIRECTION)
            & pl.all_horizontal(pl.col(*columns).is_finite().fill_null(False))
        ).alias(DerivedColumn.USABLE)
    )


def _row_record(frame: pl.DataFrame, index: ObservationCount) -> SourceRecord:
    return SourceRecord(
        {
            SourceColumn(column): SourceCell(cell)
            for column, cell in zip(frame.columns, frame.row(index), strict=True)
        }
    )


def _block_rows(
    frame: pl.DataFrame, first_row: SourceRowNumber
) -> tuple[RowIndexArray, RowIndexArray, FloatMatrix, FloatMatrix]:
    computed = _block_columns(frame, first_row)
    kinds = computed[DerivedColumn.KIND].to_numpy()
    seconds = computed[DerivedColumn.STARTED].fill_null(0).to_numpy()
    microseconds = TimeScale.MICROSECONDS_PER_SECOND
    timestamps = (seconds // microseconds) + (seconds % microseconds) / microseconds
    features = computed.select(feature_columns(len(ctu13_feature_names()))).to_numpy().copy()
    usable = computed[DerivedColumn.USABLE].to_numpy()
    for index in flagged_positions((kinds >= ReservoirGroup.BENIGN) & ~usable):
        row_number = SourceRowNumber(first_row + index)
        record = _row_record(frame, index)
        timestamps[index] = _epoch_seconds(
            _timestamp(record.cell(Ctu13Column.START_TIME), row_number)
        )
        features[index] = _features(record, row_number)
    kept = np.flatnonzero(kinds >= ReservoirGroup.BENIGN)
    rows = computed[DerivedColumn.ROW].to_numpy()
    return kinds[kept], rows[kept], timestamps[kept], features[kept]


def _parse_block(block: list[SourceLine], fields: list[SourceColumn]) -> pl.DataFrame:
    return pl.read_csv(
        io.BytesIO(b"\n".join(block)),
        has_header=False,
        new_columns=fields,
        quote_char=None,
        infer_schema=False,
    ).fill_null("")


def _read_fields(stream: SourceBinaryStream, identity_path: Path) -> list[SourceColumn]:
    header = stream.readline()
    if not header:
        raise DatasetValidationError(f"{identity_path} has no CTU-13 header")
    fields = [
        SourceColumn(field.strip())
        for field in header.removeprefix(b"\xef\xbb\xbf").decode().split(FieldDelimiter.COMMA.text)
    ]
    required = {
        Ctu13Column.START_TIME,
        Ctu13Column.LABEL,
        Ctu13Column.PROTOCOL,
        *Ctu13Column.numeric(),
        *Ctu13Column.ports(),
        Ctu13Column.STATE,
        Ctu13Column.DIRECTION,
    }
    if missing := required.difference(fields):
        raise DatasetValidationError(f"{identity_path} is missing CTU-13 fields: {sorted(missing)}")
    return fields


def _sample_block(
    block: list[SourceLine],
    fields: list[SourceColumn],
    first_row: SourceRowNumber,
    buckets: tuple[FloatRowReservoir, FloatRowReservoir],
    generator: np.random.Generator,
) -> None:
    comma = FieldDelimiter.COMMA
    malformed = (
        None
        if b"\n".join(block).count(comma.token) == len(block) * (len(fields) - 1)
        else first_misaligned_line(block, comma, {len(fields) - 1})
    )
    usable = block if malformed is None else block[:malformed]
    if usable:
        try:
            frame = _parse_block(usable, fields)
        except pl.exceptions.PolarsError as error:
            position = first_misaligned_line(usable, comma, {len(fields) - 1}) or 0
            raise DatasetValidationError(
                f"Malformed CTU-13 record at source row {first_row + position}"
            ) from error
        kinds, rows, timestamps, features = _block_rows(frame, first_row)
        extend_reservoirs(buckets, kinds, timestamps, rows, features, generator)
    if malformed is not None:
        raise DatasetValidationError(
            f"Malformed CTU-13 record at source row {first_row + malformed}"
        )


def _load_ctu13_stream(
    stream: SourceBinaryStream,
    identity_path: Path,
    source_path: Path,
    generator: np.random.Generator,
    caps: ReservoirCaps,
) -> DatasetClientData | None:
    client_identity = _client_identity(identity_path)
    scenario_attack_type = _scenario_attack_type(identity_path)
    fields = _read_fields(stream, identity_path)
    benign = _bucket(caps.benign)
    attack = _bucket(caps.attack)
    first_row = SourceRowNumber(0)
    for block in record_blocks(stream, RecordBlockRows.STANDARD, RecordCommentPolicy.NONE):
        _sample_block(block, fields, first_row, (benign, attack), generator)
        first_row = SourceRowNumber(first_row + len(block))

    if benign.sampling_population_rows == 0 or attack.sampling_population_rows == 0:
        logger.info(
            LogEvent.ADAPTER_SOURCE_READ,
            dataset=DatasetName.CTU_13.name,
            client=client_identity,
            usable=False,
            row_count=first_row,
        )
        return None
    benign_features, benign_order = ordered_reservoir_matrix((benign,))
    features, order = ordered_reservoir_matrix((attack,))
    client = DatasetClientId(DatasetName.CTU_13, client_identity)
    logger.info(
        LogEvent.ADAPTER_CLIENT_BUILT,
        dataset=DatasetName.CTU_13.name,
        client=client_identity,
        row_count=first_row,
        benign_rows=benign_features.shape[0],
        attack_rows=features.shape[0],
    )
    return DatasetClientData(
        client,
        StudyStratum.CTU_13,
        FeatureSchemaId.CTU_13_ARGUS_ENGINEERED_V1,
        ctu13_feature_names(),
        benign_features,
        benign_order,
        (
            AttackCapture(
                scenario_attack_type, source_path, features, order, attack.source_population_rows
            ),
        ),
        target_cohort_role(client),
        benign.source_population_rows,
    )


def _load_ctu13_file(
    source_file: Path, generator: np.random.Generator, caps: ReservoirCaps
) -> DatasetClientData | None:
    try:
        with source_file.open("rb") as stream:
            return _load_ctu13_stream(stream, source_file, source_file, generator, caps)
    except OSError as error:
        raise DatasetValidationError(f"Could not read CTU-13 scenario {source_file}") from error


def load_ctu13_archive(
    archive_file: Path, caps: ReservoirCaps | None = None
) -> tuple[DatasetClientData, ...]:
    selected_caps = caps or ReservoirCaps.standard(DatasetName.CTU_13)
    clients: list[DatasetClientData] = []
    generator = np.random.default_rng(SamplingSeed.CTU_13)
    try:
        with (
            indexed_bzip2.open(
                archive_file.as_posix(), parallelization=decompression_workers()
            ) as compressed,
            tarfile.open(fileobj=compressed, mode="r|") as archive,
        ):
            for member in archive:
                if not member.isfile() or not member.name.lower().endswith(
                    CtuSourceSuffix.FLOW_FILE
                ):
                    continue
                member_stream = archive.extractfile(member)
                if member_stream is None:
                    raise DatasetValidationError(
                        f"Could not read CTU-13 archive member {member.name}"
                    )
                with read_ahead(member_stream) as buffered:
                    client = _load_ctu13_stream(
                        buffered, Path(member.name), archive_file, generator, selected_caps
                    )
                if client is not None:
                    clients.append(client)
    except (OSError, tarfile.TarError) as error:
        raise DatasetValidationError(f"Could not read CTU-13 archive {archive_file}") from error
    if not clients:
        raise DatasetValidationError(f"No CTU-13 .binetflow members were found in {archive_file}")
    return tuple(clients)


def _candidate_roots(data_root: Path) -> list[Path]:
    candidates = [data_root / CorpusPath.CTU_13_EXTRACTED, data_root / CorpusPath.CTU_13]
    if data_root.name.lower() in tuple(CtuRootAlias):
        candidates.append(data_root)
    return candidates


def ctu13_source_files(data_root: Path) -> tuple[Path, ...]:
    for root in _candidate_roots(data_root):
        if not root.is_dir():
            continue
        files = tuple(
            sorted(
                path
                for path in root.rglob("*")
                if path.is_file()
                and path.suffix.lower() in (CtuSourceSuffix.FLOW_FILE, CtuSourceSuffix.CSV_FILE)
            )
        )
        if files:
            return files
    archive_candidates = (
        data_root / CorpusPath.CTU_13_ARCHIVE,
        data_root / CorpusPath.CTU_13_ARCHIVE_FLAT,
    )
    for archive_file in archive_candidates:
        if archive_file.is_file():
            return (archive_file,)
    raise DatasetValidationError(f"No CTU-13 flow files or archive found under {data_root}")


def load_ctu13_clients(
    data_root: Path, caps: ReservoirCaps | None = None
) -> tuple[DatasetClientData, ...]:
    selected_caps = caps or ReservoirCaps.standard(DatasetName.CTU_13)
    sources = ctu13_source_files(data_root)
    logger.info(LogEvent.DATASET_LOADED, dataset=DatasetName.CTU_13.name, source_count=len(sources))
    if len(sources) == 1 and sources[0].name.lower().endswith(CtuSourceSuffix.COMPRESSED_ARCHIVE):
        return load_ctu13_archive(sources[0], selected_caps)
    clients: list[DatasetClientData] = []
    generator = np.random.default_rng(SamplingSeed.CTU_13)
    for source in sources:
        client = _load_ctu13_file(source, generator, selected_caps)
        if client is not None:
            clients.append(client)
    if not clients:
        raise DatasetValidationError("CTU-13 sources contain no Normal-plus-attack scenarios")
    return tuple(clients)
