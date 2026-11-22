import io
import math
import re
import tarfile
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np
import polars as pl
import structlog

from pace.datasets.common import (
    extend_reservoirs,
    feature_columns,
    first_misaligned_line,
    first_seen_order,
    ordered_reservoir_matrix,
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
    ChunkThinning,
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
    FieldDelimiter,
    FloatRowReservoir,
    Iot23AttackName,
    Iot23Column,
    Iot23DetailAlias,
    Iot23Label,
    Iot23Sampling,
    Iot23SourceName,
    LogEvent,
    MissingValueToken,
    NetworkProtocolCategory,
    ObservationCount,
    PopulationCounting,
    RecordCommentPolicy,
    SamplingSeed,
    SharedFeature,
    SourceBinaryStream,
    SourceCell,
    SourceColumn,
    SourceLine,
    SourceRowNumber,
    StudyStratum,
    TargetCohortRole,
    TcpHistoryFlag,
    ZeekConnectionState,
    ZeekHeaderToken,
)

logger = structlog.get_logger()

_CAPTURE_PATTERN = re.compile(r"(?:^|[-_])capture[-_](?P<identity>\d+-\d+)(?:$|[-_])", re.I)
_SPACE_RUN = re.compile(r" {2,}")
_CAMEL_BOUNDARY = re.compile(r"([A-Z])([A-Z][a-z])")
_WORD_BOUNDARY = re.compile(r"([a-z0-9])([A-Z])")
_NON_ALPHANUMERIC = re.compile(r"[^A-Za-z0-9]+")


class _UnpairedLabelClasses(DatasetValidationError):
    pass


@lru_cache(maxsize=1)
def iot23_feature_names() -> tuple[FeatureName, ...]:
    return (
        *(FeatureName(column) for column in Iot23Column.numeric()),
        *(FeatureGroup.STATE.feature(state) for state in ZeekConnectionState),
        *(FeatureGroup.PROTOCOL.feature(protocol) for protocol in NetworkProtocolCategory),
        *(name for column in Iot23Column.ports() for name in column.port_features),
        FeatureName(SharedFeature.SERVICE_KNOWN),
        FeatureName(SharedFeature.HISTORY_LENGTH),
        *(FeatureGroup.HISTORY.feature(flag) for flag in TcpHistoryFlag),
    )


def _bucket(capacity: ObservationCount) -> FloatRowReservoir:
    return FloatRowReservoir(capacity, len(iot23_feature_names()))


@lru_cache(maxsize=512)
def normalize_iot23_attack_type(label: SourceCell) -> AttackType:
    value = label.strip()
    if not value or value == MissingValueToken.DASH:
        return AttackType(AttackTypeName(Iot23AttackName.UNSPECIFIED))
    camel_case_separated = _CAMEL_BOUNDARY.sub(r"\1_\2", value)
    camel_case_separated = _WORD_BOUNDARY.sub(r"\1_\2", camel_case_separated)
    normalized = _NON_ALPHANUMERIC.sub("_", camel_case_separated).strip("_").upper()
    try:
        return AttackType(AttackTypeName(Iot23DetailAlias(normalized).attack_name))
    except ValueError:
        return AttackType(AttackTypeName(normalized))


def _capture_identity(path: Path) -> ClientName:
    for component in reversed(path.parts):
        match = _CAPTURE_PATTERN.search(component)
        if match:
            return ClientName(match.group("identity"))
    return ClientName(path.parent.name)


def _zeek_fields(stream: SourceBinaryStream, source_file: Path) -> tuple[SourceColumn, ...]:
    first = True
    while line := stream.readline():
        text = line.removeprefix(b"\xef\xbb\xbf") if first else line
        first = False
        row = text.decode().rstrip("\r\n").split(FieldDelimiter.TAB.text)
        if row[0] != ZeekHeaderToken.FIELDS:
            continue
        fields = tuple(SourceColumn(field) for field in row[1:])
        if fields and fields[-1].split() == list(Iot23Column.expanded_terminal()):
            return (
                *fields[:-1],
                *(SourceColumn(column) for column in Iot23Column.expanded_terminal()),
            )
        return fields
    raise DatasetValidationError(f"{source_file} has no Zeek fields header")


def _parse_block(
    block: list[SourceLine], fields: tuple[SourceColumn, ...], source_file: Path
) -> pl.DataFrame:
    blob = b"\n".join(block)
    tab = FieldDelimiter.TAB
    layouts = {len(fields) - 1, len(fields) - 3}
    tabs = blob.count(tab.token)
    combined = tabs == len(block) * (len(fields) - 3)
    if tabs != len(block) * (len(fields) - 1) and not combined:
        raise DatasetValidationError(f"{source_file} has a malformed Zeek record")
    if first_misaligned_line(block, tab, layouts) is not None:
        raise DatasetValidationError(f"{source_file} has a malformed Zeek record")
    names = [*fields[:-3], SourceColumn(DerivedColumn.TERMINAL)] if combined else list(fields)
    try:
        frame = pl.read_csv(
            io.BytesIO(blob),
            separator=tab.text,
            has_header=False,
            new_columns=names,
            quote_char=None,
            infer_schema=False,
        ).fill_null("")
    except pl.exceptions.PolarsError as error:
        raise DatasetValidationError(f"{source_file} has a malformed Zeek record") from error
    if not combined:
        return frame
    terminal = (
        pl.col(DerivedColumn.TERMINAL)
        .str.strip_chars()
        .str.replace_all(_SPACE_RUN.pattern, "\t")
        .str.splitn("\t", len(Iot23Column.expanded_terminal()))
    )
    split = frame.select(terminal.alias(DerivedColumn.SPLIT)).unnest(DerivedColumn.SPLIT)
    if split.null_count().sum_horizontal().item() != 0:
        raise DatasetValidationError(f"{source_file} has a malformed Zeek record")
    parts = Iot23Column.expanded_terminal()
    return frame.drop(DerivedColumn.TERMINAL).with_columns(
        *(split.to_series(index).alias(part) for index, part in enumerate(parts))
    )


def _number(column: Iot23Column) -> pl.Expr:
    text = pl.col(column).str.strip_chars()
    return (
        pl.when(text.is_in(polars_texts(MissingValueToken.zeek())))
        .then(pl.lit(0.0))
        .otherwise(text.cast(pl.Float64, strict=False))
    )


def _feature_expressions() -> list[pl.Expr]:
    history_text = pl.col(Iot23Column.HISTORY).str.strip_chars()
    history = (
        pl.when(history_text == polars_text(MissingValueToken.EMPTY))
        .then(pl.lit(polars_text(MissingValueToken.DASH)))
        .otherwise(history_text)
    )
    state = pl.col(Iot23Column.CONNECTION_STATE).str.strip_chars()
    protocol = pl.col(Iot23Column.PROTOCOL).str.strip_chars().str.to_lowercase()
    ports = [_number(column) for column in Iot23Column.ports()]
    return [
        *(_number(column).alias(FeatureName(column)) for column in Iot23Column.numeric()),
        *((state == polars_text(item)).cast(pl.Float64) for item in ZeekConnectionState),
        *((protocol == polars_text(item)).cast(pl.Float64) for item in NetworkProtocolCategory),
        *(expression for port in ports for expression in (privileged_port_indicator(port), port)),
        (
            ~pl.col(Iot23Column.SERVICE)
            .str.strip_chars()
            .is_in(polars_texts(MissingValueToken.zeek()))
        ).cast(pl.Float64),
        history.str.len_chars().cast(pl.Float64),
        *(
            history.str.count_matches(polars_text(flag), literal=True).cast(pl.Float64)
            for flag in TcpHistoryFlag
        ),
    ]


def _usable_rows(block: pl.DataFrame, first_row: SourceRowNumber) -> pl.DataFrame:
    columns = feature_columns(len(iot23_feature_names()))
    detail_text = pl.col(Iot23Column.DETAILED_LABEL).str.strip_chars()
    parsed = block.select(
        (pl.int_range(pl.len(), dtype=pl.Int64) + first_row).alias(DerivedColumn.SOURCE_ROW),
        pl.col(Iot23Column.LABEL).str.strip_chars().str.to_lowercase().alias(DerivedColumn.LABEL),
        pl.when(detail_text == polars_text(MissingValueToken.EMPTY))
        .then(pl.lit(polars_text(MissingValueToken.DASH)))
        .otherwise(detail_text)
        .alias(DerivedColumn.DETAIL),
        pl.col(Iot23Column.TIMESTAMP)
        .str.strip_chars()
        .cast(pl.Float64, strict=False)
        .alias(DerivedColumn.TIMESTAMP),
        *(
            expression.alias(column)
            for expression, column in zip(_feature_expressions(), columns, strict=True)
        ),
    )
    return parsed.filter(
        pl.col(DerivedColumn.LABEL).is_in(polars_texts(Iot23Label))
        & pl.all_horizontal(pl.col(DerivedColumn.TIMESTAMP, *columns).is_finite())
    )


class _Sampler:
    def __init__(self, sampling: Iot23Sampling) -> None:
        self.sampling = sampling
        self.random = np.random.default_rng(SamplingSeed.IOT_23_RESERVOIR)
        self.chunk_random = np.random.default_rng(SamplingSeed.IOT_23_CHUNK)
        self.buckets: list[FloatRowReservoir] = []
        self.indexes: dict[tuple[Iot23Label, SourceCell], ObservationCount] = {}
        self.benign: dict[SourceCell, FloatRowReservoir] = {}
        self.attacks: dict[AttackType, dict[SourceCell, FloatRowReservoir]] = defaultdict(dict)

    def bucket_index(self, label: Iot23Label, detail: SourceCell) -> ObservationCount:
        key = (label, detail)
        if key not in self.indexes:
            if label is Iot23Label.BENIGN:
                bucket = self.benign.setdefault(detail, _bucket(self.sampling.benign_cap))
            else:
                bucket = self.attacks[normalize_iot23_attack_type(detail)].setdefault(
                    detail, _bucket(self.sampling.malicious_cap_per_detail)
                )
            self.indexes[key] = len(self.buckets)
            self.buckets.append(bucket)
        return self.indexes[key]

    def add(self, rows: pl.DataFrame, thinning: ChunkThinning) -> None:
        if rows.height == 0:
            return
        keys = zip(
            (Iot23Label(label) for label in rows[DerivedColumn.LABEL].to_list()),
            (SourceCell(detail) for detail in rows[DerivedColumn.DETAIL].to_list()),
            strict=True,
        )
        ids = np.asarray([self.bucket_index(*key) for key in keys], dtype=np.int64)
        features = rows.select(feature_columns(len(iot23_feature_names()))).to_numpy()
        timestamps = rows[DerivedColumn.TIMESTAMP].to_numpy()
        source_rows = rows[DerivedColumn.SOURCE_ROW].to_numpy()
        if thinning is ChunkThinning.NONE:
            extend_reservoirs(self.buckets, ids, timestamps, source_rows, features, self.random)
            return
        for index in first_seen_order(ids):
            members = np.flatnonzero(ids == index)
            bucket = self.buckets[index]
            bucket.source_population_rows += members.size
            if members.size > self.sampling.thin_group_min_rows:
                members = members[
                    np.sort(
                        self.chunk_random.choice(
                            members.size,
                            math.floor(members.size * self.sampling.thin_fraction),
                            replace=False,
                        )
                    )
                ]
            extend_reservoirs(
                [bucket],
                np.zeros(members.size, dtype=np.int64),
                timestamps[members],
                source_rows[members],
                features[members],
                self.random,
                population_counting=PopulationCounting.UNCOUNTED,
            )


def _sample_stream(
    stream: SourceBinaryStream,
    source_file: Path,
    sampling: Iot23Sampling,
    thinning: ChunkThinning,
) -> tuple[_Sampler, SourceRowNumber]:
    fields = _zeek_fields(stream, source_file)
    required = {
        Iot23Column.TIMESTAMP,
        Iot23Column.LABEL,
        Iot23Column.DETAILED_LABEL,
        *Iot23Column.numeric(),
        *Iot23Column.ports(),
        Iot23Column.CONNECTION_STATE,
        Iot23Column.PROTOCOL,
        Iot23Column.SERVICE,
        Iot23Column.HISTORY,
    }
    if missing := required.difference(fields):
        raise DatasetValidationError(f"{source_file} is missing IoT-23 fields: {sorted(missing)}")
    sampler = _Sampler(sampling)
    first_row = SourceRowNumber(0)
    for block in record_blocks(stream, sampling.chunk_rows, RecordCommentPolicy.ZEEK):
        rows = _usable_rows(_parse_block(block, fields, source_file), first_row)
        sampler.add(rows, thinning)
        first_row = SourceRowNumber(first_row + len(block))
    return sampler, first_row


def _attack_captures(source_file: Path, sampler: _Sampler) -> tuple[AttackCapture, ...]:
    captures: list[AttackCapture] = []
    for attack_type, detail_groups in sorted(
        sampler.attacks.items(), key=lambda item: item[0].identifier
    ):
        rows = tuple(detail_groups.values())
        features, source_order = ordered_reservoir_matrix(rows)
        captures.append(
            AttackCapture(
                attack_type,
                source_file,
                features,
                source_order,
                sum(bucket.source_population_rows for bucket in rows),
            )
        )
    return tuple(captures)


def _load_iot23_stream(
    stream: SourceBinaryStream,
    source_file: Path,
    client_identity: ClientName,
    sampling: Iot23Sampling,
    thinning: ChunkThinning,
) -> DatasetClientData:
    client = DatasetClientId(DatasetName.IOT_23, client_identity)
    cohort_role = target_cohort_role(client)
    sampler, row_count = _sample_stream(stream, source_file, sampling, thinning)
    if not sampler.benign or (
        not sampler.attacks and cohort_role is not TargetCohortRole.PEER_ONLY
    ):
        raise _UnpairedLabelClasses(
            f"{source_file} must contain both benign and labelled malicious records"
        )
    benign_features, benign_order = ordered_reservoir_matrix(tuple(sampler.benign.values()))
    captures = _attack_captures(source_file, sampler)
    logger.info(
        LogEvent.ADAPTER_CLIENT_BUILT,
        dataset=DatasetName.IOT_23.name,
        client=client_identity,
        row_count=row_count,
        benign_rows=benign_features.shape[0],
        attack_types=len(captures),
    )
    return DatasetClientData(
        client,
        StudyStratum.IOT_23,
        FeatureSchemaId.IOT_23_ZEEK_ENGINEERED_V1,
        iot23_feature_names(),
        benign_features,
        benign_order,
        captures,
        cohort_role,
        sum(bucket.source_population_rows for bucket in sampler.benign.values()),
    )


def load_iot23_capture(
    source_file: Path, sampling: Iot23Sampling | None = None
) -> DatasetClientData:
    selected = sampling or Iot23Sampling.standard()
    try:
        with source_file.open("rb") as stream:
            return _load_iot23_stream(
                stream,
                source_file,
                _capture_identity(source_file),
                selected,
                ChunkThinning.HISTORICAL
                if source_file.stat().st_size > selected.large_file_bytes
                else ChunkThinning.NONE,
            )
    except OSError as error:
        raise DatasetValidationError(f"Could not read IoT-23 source {source_file}") from error


def load_iot23_archive(
    archive_file: Path, sampling: Iot23Sampling | None = None
) -> tuple[DatasetClientData, ...]:
    selected = sampling or Iot23Sampling.standard()
    clients: list[DatasetClientData] = []
    unpaired_captures = 0
    try:
        with tarfile.open(archive_file, mode="r|gz") as archive:
            for member in archive:
                if not member.isfile() or Path(member.name).name != Iot23SourceName.CONNECTION_LOG:
                    continue
                binary_stream = archive.extractfile(member)
                if binary_stream is None:
                    raise DatasetValidationError(
                        f"Could not read IoT-23 archive member {member.name}"
                    )
                member_path = Path(member.name)
                try:
                    try:
                        with read_ahead(binary_stream) as buffered:
                            clients.append(
                                _load_iot23_stream(
                                    buffered,
                                    archive_file,
                                    _capture_identity(member_path),
                                    selected,
                                    ChunkThinning.HISTORICAL
                                    if member.size > selected.large_file_bytes
                                    else ChunkThinning.NONE,
                                )
                            )
                    except _UnpairedLabelClasses:
                        unpaired_captures += 1
                        logger.info(
                            LogEvent.ADAPTER_SOURCE_READ,
                            dataset=DatasetName.IOT_23.name,
                            source_file=member_path.as_posix(),
                            client=_capture_identity(member_path),
                            usable=False,
                        )
                finally:
                    binary_stream.close()
    except (OSError, tarfile.TarError) as error:
        raise DatasetValidationError(f"Could not read IoT-23 archive {archive_file}") from error
    if not clients:
        raise DatasetValidationError(
            f"No mixed benign-and-malicious IoT-23 captures were found in {archive_file}; "
            f"{unpaired_captures} captures had only one label class"
        )
    logger.info(
        LogEvent.DATASET_LOADED,
        dataset=DatasetName.IOT_23.name,
        client_count=len(clients),
        unpaired_captures=unpaired_captures,
    )
    return tuple(sorted(clients, key=lambda item: item.client.name))


def load_iot23_clients(
    data_root: Path, sampling: Iot23Sampling | None = None
) -> tuple[DatasetClientData, ...]:
    sources = iot23_source_files(data_root)
    logger.info(LogEvent.DATASET_LOADED, dataset=DatasetName.IOT_23.name, source_count=len(sources))
    if len(sources) == 1 and sources[0].name.endswith(Iot23SourceName.ARCHIVE_SUFFIX):
        return load_iot23_archive(sources[0], sampling)
    return tuple(load_iot23_capture(source_file, sampling) for source_file in sources)


def iot23_source_files(data_root: Path) -> tuple[Path, ...]:
    candidates = (data_root / CorpusPath.IOT_23, data_root)
    for root in candidates:
        if root.is_dir():
            files = tuple(sorted(root.rglob(Iot23SourceName.CONNECTION_LOG)))
            if files:
                return files
            archives = tuple(sorted(root.glob(Iot23SourceName.ARCHIVE)))
            if archives:
                return archives
    raise DatasetValidationError(
        "No extracted IoT-23 conn.log.labeled files were found; unpack the official archive "
        "to a separate derived-data directory first"
    )
