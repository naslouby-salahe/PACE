import csv
import io
import math
import re
from dataclasses import dataclass
from datetime import UTC, datetime
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
    is_privileged_port,
    parse_integer,
    parse_numeric_literal,
    polars_text,
    polars_texts,
    privileged_port_indicator,
    record_blocks,
    reservoir_columns,
    signed_log_feature_transform,
    stable_timestamp_order,
    target_cohort_role,
)
from pace.types import (
    AttackCapture,
    AttackType,
    AttackTypeName,
    CalendarMonth,
    ClientName,
    CorpusPath,
    DatasetClientData,
    DatasetClientId,
    DatasetName,
    DatasetValidationError,
    DerivedColumn,
    FeatureGroup,
    FeatureName,
    FeatureSchemaId,
    FeatureValue,
    FloatMatrix,
    GothamAggregateFeature,
    GothamBaseFeature,
    GothamColumn,
    GothamLabel,
    IntegerBase,
    IntRowReservoir,
    IpProtocolNumber,
    LearningSeed,
    LogEvent,
    MissingValueToken,
    ObservationCount,
    PacketProtocolToken,
    RecordBlockRows,
    RecordCommentPolicy,
    ReservoirCaps,
    ReservoirGroup,
    RowIndexArray,
    SamplingSeed,
    SourceCell,
    SourceColumn,
    SourceLine,
    SourceRecord,
    SourceRowNumber,
    StudyStratum,
    TcpFlagMask,
    TimeScale,
    TimestampPart,
    UnixTimestampNanoseconds,
)

logger = structlog.get_logger()

_TIMESTAMP_PATTERN = re.compile(
    r"(?P<month>[A-Za-z]{3})\s+(?P<day>[0-9]{1,2}), (?P<year>[0-9]{4}) "
    r"(?P<hour>[0-9]{2}):(?P<minute>[0-9]{2}):(?P<second>[0-9]{2})"
    r"\.(?P<fraction>[0-9]{1,9}) GMT"
)
_LABEL_SEPARATORS = re.compile(r"[^a-z0-9]+")


@lru_cache(maxsize=1)
def gotham_feature_names() -> tuple[FeatureName, ...]:
    return (
        *(FeatureName(feature) for feature in GothamBaseFeature),
        *(FeatureGroup.STACK.feature(token) for token in PacketProtocolToken),
        *(FeatureName(feature) for feature in GothamAggregateFeature),
    )


def _feature_width() -> ObservationCount:
    return len(GothamBaseFeature) + len(PacketProtocolToken)


def _bucket(capacity: ObservationCount) -> IntRowReservoir:
    return IntRowReservoir(capacity, _feature_width())


def _required_columns() -> frozenset[SourceColumn]:
    return frozenset(SourceColumn(column) for column in GothamColumn)


def _unix_epoch() -> datetime:
    return datetime.fromtimestamp(0, UTC).replace(tzinfo=None)


def _numeric_text(
    value: SourceCell, column: GothamColumn, row_number: SourceRowNumber
) -> FeatureValue:
    number = parse_numeric_literal(value)
    if number is None:
        raise DatasetValidationError(f"Invalid numeric {column} at source row {row_number}")
    if not isfinite(number):
        raise DatasetValidationError(f"Non-finite {column} at source row {row_number}")
    return number


def _number(
    record: SourceRecord, column: GothamColumn, row_number: SourceRowNumber
) -> FeatureValue:
    return _numeric_text(record.cell(column), column, row_number)


def _port(
    record: SourceRecord,
    tcp_column: GothamColumn,
    udp_column: GothamColumn,
    row_number: SourceRowNumber,
) -> FeatureValue:
    raw = SourceCell(record.cell(tcp_column).strip() or record.cell(udp_column).strip())
    if raw in MissingValueToken.numeric():
        return FeatureValue(0.0)
    return _numeric_text(raw, tcp_column, row_number)


def _hex_number(value: SourceCell) -> FeatureValue:
    text = SourceCell(value.strip())
    if not text.lower().startswith("0x"):
        return FeatureValue(0.0)
    return parse_numeric_literal(text) or FeatureValue(0.0)


def _flag_set(flags: FeatureValue, mask: TcpFlagMask) -> FeatureValue:
    return FeatureValue((math.trunc(flags) & mask) != 0)


def _packet_features(record: SourceRecord, row_number: SourceRowNumber) -> tuple[FeatureValue, ...]:
    protocol = _number(record, GothamColumn.IP_PROTOCOL, row_number)
    flags = _hex_number(record.cell(GothamColumn.TCP_FLAGS))
    source_port = _port(
        record, GothamColumn.TCP_SOURCE_PORT, GothamColumn.UDP_SOURCE_PORT, row_number
    )
    destination_port = _port(
        record, GothamColumn.TCP_DESTINATION_PORT, GothamColumn.UDP_DESTINATION_PORT, row_number
    )
    stack = record.cell(GothamColumn.FRAME_PROTOCOLS).lower()
    return (
        _number(record, GothamColumn.FRAME_LENGTH, row_number),
        _number(record, GothamColumn.IP_TTL, row_number),
        protocol,
        _hex_number(record.cell(GothamColumn.IP_TOS)),
        _hex_number(record.cell(GothamColumn.IP_FLAGS)),
        *(_flag_set(flags, mask) for mask in TcpFlagMask.feature_order()),
        _number(record, GothamColumn.TCP_WINDOW_SIZE, row_number),
        _number(record, GothamColumn.TCP_PDU_SIZE, row_number),
        *(FeatureValue(protocol == number) for number in IpProtocolNumber.feature_order()),
        is_privileged_port(source_port),
        is_privileged_port(destination_port),
        source_port,
        destination_port,
        *(FeatureValue(token in stack) for token in PacketProtocolToken),
    )


def _parse_timestamp(value: SourceCell, row_number: SourceRowNumber) -> UnixTimestampNanoseconds:
    match = _TIMESTAMP_PATTERN.fullmatch(value.strip())
    if match is None:
        raise DatasetValidationError(f"Invalid Gotham timestamp at source row {row_number}")

    def part(name: TimestampPart) -> SourceCell:
        return SourceCell(match.group(name))

    try:
        base = datetime(
            parse_integer(part(TimestampPart.YEAR)),
            CalendarMonth(part(TimestampPart.MONTH)).number,
            parse_integer(part(TimestampPart.DAY)),
            parse_integer(part(TimestampPart.HOUR)),
            parse_integer(part(TimestampPart.MINUTE)),
            parse_integer(part(TimestampPart.SECOND)),
        )
    except ValueError as error:
        raise DatasetValidationError(
            f"Invalid Gotham timestamp at source row {row_number}"
        ) from error
    elapsed = base - _unix_epoch()
    seconds = elapsed.days * 86400 + elapsed.seconds
    nanoseconds = parse_integer(SourceCell(part(TimestampPart.FRACTION).ljust(9, "0")))
    return UnixTimestampNanoseconds(seconds * TimeScale.NANOSECONDS_PER_SECOND + nanoseconds)


@lru_cache(maxsize=256)
def _attack_name(label: SourceCell) -> AttackType:
    normalized = _LABEL_SEPARATORS.sub("_", label.strip().lower()).strip("_")
    if not normalized:
        raise DatasetValidationError("Gotham contains an empty packet label")
    return AttackType(AttackTypeName(normalized.upper()))


def _decimal(column: GothamColumn) -> pl.Expr:
    text = pl.col(column).str.strip_chars()
    return (
        pl.when(text.is_in(polars_texts(MissingValueToken.numeric())))
        .then(pl.lit(0.0))
        .otherwise(text.cast(pl.Float64, strict=False))
    )


def _hexadecimal(column: GothamColumn) -> pl.Expr:
    text = pl.col(column).str.strip_chars().str.to_lowercase()
    return (
        pl.when(text.str.starts_with("0x"))
        .then(text.str.slice(2).str.to_integer(base=IntegerBase.HEXADECIMAL, strict=False))
        .otherwise(pl.lit(0, dtype=pl.Int64))
        .fill_null(0)
    )


def _port_expression(tcp_column: GothamColumn, udp_column: GothamColumn) -> pl.Expr:
    tcp = pl.col(tcp_column).str.strip_chars()
    raw = (
        pl.when(tcp != polars_text(MissingValueToken.EMPTY))
        .then(tcp)
        .otherwise(pl.col(udp_column).str.strip_chars())
    )
    return (
        pl.when(raw.is_in(polars_texts(MissingValueToken.numeric())))
        .then(pl.lit(0.0))
        .otherwise(raw.cast(pl.Float64, strict=False))
    )


def _timestamp_nanoseconds(frame: pl.DataFrame) -> pl.Series:
    parts = frame.select(
        pl.col(GothamColumn.FRAME_TIME)
        .str.strip_chars()
        .str.extract_groups(f"^{_TIMESTAMP_PATTERN.pattern}$")
        .alias(DerivedColumn.SPLIT)
    ).unnest(DerivedColumn.SPLIT)
    month = pl.col(TimestampPart.MONTH).replace_strict(
        {polars_text(month): month.number for month in CalendarMonth},
        default=None,
        return_dtype=pl.Int64,
    )
    day, year, hour, minute, second = (
        pl.col(part).cast(pl.Int64)
        for part in (
            TimestampPart.DAY,
            TimestampPart.YEAR,
            TimestampPart.HOUR,
            TimestampPart.MINUTE,
            TimestampPart.SECOND,
        )
    )
    date = pl.date(year, month, day).cast(pl.Int32).cast(pl.Int64)
    fraction = pl.col(TimestampPart.FRACTION).str.pad_end(9, "0").cast(pl.Int64)
    valid = (hour < 24) & (minute < 60) & (second < 60)
    nanoseconds = (
        date * 86400 + hour * 3600 + minute * 60 + second
    ) * TimeScale.NANOSECONDS_PER_SECOND + fraction
    return parts.select(
        pl.when(valid).then(nanoseconds).alias(DerivedColumn.NANOSECONDS)
    ).to_series()


def _base_expressions() -> list[pl.Expr]:
    flags = _hexadecimal(GothamColumn.TCP_FLAGS)
    protocol = _decimal(GothamColumn.IP_PROTOCOL)
    source = _port_expression(GothamColumn.TCP_SOURCE_PORT, GothamColumn.UDP_SOURCE_PORT)
    destination = _port_expression(
        GothamColumn.TCP_DESTINATION_PORT, GothamColumn.UDP_DESTINATION_PORT
    )
    stack = pl.col(GothamColumn.FRAME_PROTOCOLS).str.to_lowercase()
    return [
        _decimal(GothamColumn.FRAME_LENGTH),
        _decimal(GothamColumn.IP_TTL),
        protocol,
        _hexadecimal(GothamColumn.IP_TOS).cast(pl.Float64),
        _hexadecimal(GothamColumn.IP_FLAGS).cast(pl.Float64),
        *(((flags & mask) != 0).cast(pl.Float64) for mask in TcpFlagMask.feature_order()),
        _decimal(GothamColumn.TCP_WINDOW_SIZE),
        _decimal(GothamColumn.TCP_PDU_SIZE),
        *((protocol == number).cast(pl.Float64) for number in IpProtocolNumber.feature_order()),
        privileged_port_indicator(source),
        privileged_port_indicator(destination),
        source,
        destination,
        *(
            stack.str.contains(polars_text(token), literal=True).cast(pl.Float64)
            for token in PacketProtocolToken
        ),
    ]


def _row_record(frame: pl.DataFrame, index: ObservationCount) -> SourceRecord:
    return SourceRecord(
        {
            SourceColumn(column): SourceCell(cell)
            for column, cell in zip(frame.columns, frame.row(index), strict=True)
        }
    )


def _block_arrays(
    frame: pl.DataFrame, first_row: SourceRowNumber
) -> tuple[RowIndexArray, FloatMatrix, pl.DataFrame]:
    columns = feature_columns(_feature_width())
    base = frame.select(
        expression.alias(column)
        for expression, column in zip(_base_expressions(), columns, strict=True)
    )
    try:
        nanoseconds = _timestamp_nanoseconds(frame)
    except pl.exceptions.PolarsError:
        nanoseconds = pl.Series([None] * frame.height, dtype=pl.Int64)
    usable = (
        nanoseconds.is_not_null().to_numpy()
        & base.select(pl.all_horizontal(pl.all().is_finite().fill_null(False)))
        .to_series()
        .to_numpy()
    )
    timestamps = nanoseconds.fill_null(0).to_numpy().astype(np.int64)
    features = base.to_numpy().copy()
    for index in flagged_positions(~usable):
        row_number = SourceRowNumber(first_row + index)
        record = _row_record(frame, index)
        timestamps[index] = _parse_timestamp(record.cell(GothamColumn.FRAME_TIME), row_number)
        features[index] = _packet_features(record, row_number)
    aggregates = pl.DataFrame(
        {
            DerivedColumn.SECOND: timestamps // TimeScale.NANOSECONDS_PER_SECOND,
            DerivedColumn.LENGTH: features[:, GothamBaseFeature.FRAME_LENGTH.position],
            DerivedColumn.PORT: features[:, GothamBaseFeature.DESTINATION_PORT.position],
            DerivedColumn.ADDRESS: frame[GothamColumn.IP_DESTINATION].str.strip_chars(),
            DerivedColumn.SYN: features[:, GothamBaseFeature.TCP_SYN.position],
        }
    )
    return timestamps, features, aggregates


class _SecondTotals:
    def __init__(self) -> None:
        self.sums: list[pl.DataFrame] = []
        self.ports: list[pl.DataFrame] = []
        self.addresses: list[pl.DataFrame] = []

    def add(self, aggregates: pl.DataFrame) -> None:
        self.sums.append(
            aggregates.group_by(DerivedColumn.SECOND).agg(
                pl.len().alias(DerivedColumn.PACKETS),
                pl.col(DerivedColumn.LENGTH).sum().alias(DerivedColumn.BYTES),
                pl.col(DerivedColumn.SYN).sum().alias(DerivedColumn.SYN),
            )
        )
        self.ports.append(aggregates.select(DerivedColumn.SECOND, DerivedColumn.PORT).unique())
        self.addresses.append(
            aggregates.select(DerivedColumn.SECOND, DerivedColumn.ADDRESS).unique()
        )

    def table(self) -> pl.DataFrame:
        totals = (
            pl.concat(self.sums)
            .group_by(DerivedColumn.SECOND)
            .agg(
                pl.col(DerivedColumn.PACKETS).sum(),
                pl.col(DerivedColumn.BYTES).sum(),
                pl.col(DerivedColumn.SYN).sum(),
            )
        )
        port_counts = (
            pl.concat(self.ports)
            .unique()
            .group_by(DerivedColumn.SECOND)
            .agg(pl.len().alias(DerivedColumn.PORTS))
        )
        address_counts = (
            pl.concat(self.addresses)
            .unique()
            .group_by(DerivedColumn.SECOND)
            .agg(pl.len().alias(DerivedColumn.ADDRESSES))
        )
        return (
            totals.join(port_counts, on=DerivedColumn.SECOND)
            .join(address_counts, on=DerivedColumn.SECOND)
            .sort(DerivedColumn.SECOND)
        )


def _ordered_packet_matrix(
    rows: IntRowReservoir, totals: pl.DataFrame
) -> tuple[FloatMatrix, RowIndexArray]:
    timestamps, source_rows, packet_features = reservoir_columns((rows,))
    order, _ = stable_timestamp_order(timestamps, source_rows)
    seconds = totals[DerivedColumn.SECOND].to_numpy()
    position = np.searchsorted(seconds, timestamps[order] // TimeScale.NANOSECONDS_PER_SECOND)
    packets = totals[DerivedColumn.PACKETS].to_numpy()[position]
    aggregate_features = np.column_stack(
        (
            packets,
            totals[DerivedColumn.BYTES].to_numpy()[position],
            totals[DerivedColumn.PORTS].to_numpy()[position],
            totals[DerivedColumn.ADDRESSES].to_numpy()[position],
            totals[DerivedColumn.SYN].to_numpy()[position] / packets,
        )
    )
    features = np.concatenate((packet_features[order], aggregate_features), axis=1).astype(
        np.float64, copy=False
    )
    ordered_features = signed_log_feature_transform(features)
    ordered_features.setflags(write=False)
    return ordered_features, np.arange(timestamps.size, dtype=np.int64)


def _parse_frame(block: list[SourceLine], names: list[SourceColumn]) -> pl.DataFrame:
    return pl.read_csv(
        io.BytesIO(b"\n".join(block)), has_header=False, new_columns=names, infer_schema=False
    ).fill_null("")


def _first_malformed_row(
    block: list[SourceLine], width: ObservationCount
) -> ObservationCount | None:
    return next(
        (
            index
            for index, row in enumerate(csv.reader(line.decode() for line in block))
            if len(row) != width
        ),
        None,
    )


def _checked_frame(
    block: list[SourceLine], names: list[SourceColumn], first_row: SourceRowNumber
) -> pl.DataFrame:
    if not has_regular_field_layout(block, len(names)):
        malformed = _first_malformed_row(block, len(names))
        raise DatasetValidationError(
            f"Malformed Gotham record at source row {first_row + (malformed or 0)}"
        )
    try:
        return _parse_frame(block, names)
    except pl.exceptions.PolarsError as error:
        raise DatasetValidationError(
            f"Malformed Gotham record at source row {first_row}"
        ) from error


def _group_labels(
    frame: pl.DataFrame,
    buckets: list[IntRowReservoir],
    attack_indexes: dict[AttackType, ObservationCount],
    attack_cap: ObservationCount,
) -> RowIndexArray:
    groups = np.full(frame.height, ReservoirGroup.SKIPPED, dtype=np.int64)
    labels = frame[GothamColumn.LABEL].str.strip_chars()
    for label in labels.unique().to_list():
        known = GothamLabel.known(SourceCell(label))
        if not label or (known is not None and not known.is_benign):
            continue
        if known is not None:
            index = ReservoirGroup.BENIGN
        else:
            attack_type = _attack_name(SourceCell(label))
            if attack_type not in attack_indexes:
                attack_indexes[attack_type] = len(buckets)
                buckets.append(_bucket(attack_cap))
            index = attack_indexes[attack_type]
        groups[(labels == label).to_numpy()] = index
    return groups


@dataclass(slots=True)
class _CaptureSample:
    buckets: list[IntRowReservoir]
    attack_indexes: dict[AttackType, ObservationCount]
    totals: _SecondTotals

    @property
    def benign(self) -> IntRowReservoir:
        return self.buckets[0]

    @property
    def is_usable(self) -> bool:
        return self.benign.sampling_population_rows > 0 and len(self.attack_indexes) > 0


def _sample_capture(source_file: Path, seed: LearningSeed, caps: ReservoirCaps) -> _CaptureSample:
    random = np.random.default_rng(seed)
    sample = _CaptureSample([_bucket(caps.benign)], {}, _SecondTotals())
    try:
        with source_file.open("rb") as stream:
            header = stream.readline().removeprefix(b"\xef\xbb\xbf").decode()
            names = [SourceColumn(name) for name in next(csv.reader([header]), ())]
            if missing := _required_columns() - set(names):
                raise DatasetValidationError(
                    f"{source_file} is missing Gotham columns: {', '.join(sorted(missing))}"
                )
            first_row = SourceRowNumber(0)
            for block in record_blocks(stream, RecordBlockRows.STANDARD, RecordCommentPolicy.NONE):
                frame = _checked_frame(block, names, first_row)
                timestamps, features, aggregates = _block_arrays(frame, first_row)
                sample.totals.add(aggregates)
                groups = _group_labels(frame, sample.buckets, sample.attack_indexes, caps.attack)
                kept = np.flatnonzero(groups >= ReservoirGroup.BENIGN)
                extend_reservoirs(
                    sample.buckets,
                    groups[kept],
                    timestamps[kept],
                    kept.astype(np.int64) + first_row,
                    features[kept],
                    random,
                )
                first_row = SourceRowNumber(first_row + len(block))
    except OSError as error:
        raise DatasetValidationError(f"Could not read Gotham capture {source_file}") from error
    return sample


def _client_from_sample(source_file: Path, sample: _CaptureSample) -> DatasetClientData:
    table = sample.totals.table()
    benign_features, benign_order = _ordered_packet_matrix(sample.benign, table)
    captures = tuple(
        AttackCapture(
            label,
            source_file,
            *_ordered_packet_matrix(sample.buckets[index], table),
            sample.buckets[index].source_population_rows,
        )
        for label, index in sorted(
            sample.attack_indexes.items(), key=lambda item: item[0].identifier
        )
    )
    client = DatasetClientId(DatasetName.GOTHAM_2025, ClientName(source_file.stem))
    logger.info(
        LogEvent.ADAPTER_CLIENT_BUILT,
        dataset=DatasetName.GOTHAM_2025.name,
        client=client.name,
        benign_rows=benign_features.shape[0],
        attack_types=len(captures),
    )
    return DatasetClientData(
        client,
        StudyStratum.GOTHAM_2025,
        FeatureSchemaId.GOTHAM_TSHARK_ENGINEERED_V1,
        gotham_feature_names(),
        benign_features,
        benign_order,
        captures,
        target_cohort_role(client),
        sample.benign.source_population_rows,
    )


def _read_gotham_capture(
    source_file: Path, seed: LearningSeed, caps: ReservoirCaps
) -> DatasetClientData | None:
    sample = _sample_capture(source_file, seed, caps)
    if not sample.is_usable:
        logger.info(
            LogEvent.ADAPTER_SOURCE_READ,
            dataset=DatasetName.GOTHAM_2025.name,
            source_file=source_file.name,
            usable=False,
        )
        return None
    return _client_from_sample(source_file, sample)


def load_gotham_clients(
    data_root: Path, caps: ReservoirCaps | None = None
) -> tuple[DatasetClientData, ...]:
    selected_caps = caps or ReservoirCaps.standard(DatasetName.GOTHAM_2025)
    source_files = gotham_source_files(data_root)
    logger.info(
        LogEvent.DATASET_LOADED,
        dataset=DatasetName.GOTHAM_2025.name,
        source_count=len(source_files),
    )
    clients = tuple(
        client
        for index, source_file in enumerate(source_files)
        if (
            client := _read_gotham_capture(
                source_file, SamplingSeed.GOTHAM_BASE + index, selected_caps
            )
        )
        is not None
    )
    if not clients:
        raise DatasetValidationError(
            "Gotham corpus contains no clients with benign and attack data"
        )
    return clients


def gotham_source_files(data_root: Path) -> tuple[Path, ...]:
    processed = data_root / CorpusPath.GOTHAM_PROCESSED
    if not processed.is_dir():
        raise DatasetValidationError(f"Gotham processed data is unavailable: {processed}")
    files = tuple(sorted(path for path in processed.glob("*.csv") if path.is_file()))
    if not files:
        raise DatasetValidationError(f"No Gotham processed CSV files found in {processed}")
    return files
