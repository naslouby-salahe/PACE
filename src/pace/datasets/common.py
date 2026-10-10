import io
import os
import queue
import re
import threading
from collections.abc import Buffer, Generator, Iterable, Iterator, Sequence
from contextlib import contextmanager
from enum import StrEnum

import numpy as np
import polars as pl

from pace.types import (
    ArtifactPayload,
    AttackTypeName,
    BooleanArray,
    ChronologicalOrder,
    ChunkSize,
    ClientName,
    Ctu13Scenario,
    DatasetClientData,
    DatasetClientId,
    DatasetName,
    DatasetValidationError,
    DecompressionWorkers,
    FeatureName,
    FeatureSchemaId,
    FeatureValue,
    FieldDelimiter,
    Float32Matrix,
    FloatMatrix,
    FloatRowReservoir,
    FrozenStrictModel,
    GothamDevice,
    IntegerBase,
    IntRowReservoir,
    Iot23Capture,
    LabNetwork,
    LengthPrefixedHasher,
    LineBreak,
    MissingValueToken,
    NBaIoTDevice,
    ObservationCount,
    PollInterval,
    PopulationCounting,
    QueueCapacity,
    QueueDepth,
    RecordCommentPolicy,
    RowIndexArray,
    ScientificDigest,
    SourceBinaryStream,
    SourceCell,
    SourceColumn,
    SourceInteger,
    SourceLine,
    StreamBufferSize,
    StudyStratum,
    TargetCohortRole,
    TextEncoding,
    TimestampValues,
    TonIoTStudyHost,
    canonical_json,
)


def _ton_iot_cohort_role(
    client: DatasetClientId, lab_network: LabNetwork | None
) -> TargetCohortRole:
    if lab_network is None:
        raise DatasetValidationError("TON-IoT cohort roles require the lab network")
    study_hosts = frozenset(ClientName(host.host_ip(lab_network)) for host in TonIoTStudyHost)
    if client.name in study_hosts:
        return TargetCohortRole.REPORTED_TARGET
    return TargetCohortRole.PEER_ONLY


def _iot23_cohort_role(client: DatasetClientId) -> TargetCohortRole:
    try:
        return Iot23Capture(client.name).cohort_role
    except ValueError:
        return TargetCohortRole.OUTSIDE_PEER_POOL


def _declared_cohort_role[Member: StrEnum](
    client: DatasetClientId, members: type[Member]
) -> TargetCohortRole:
    try:
        members(client.name)
    except ValueError:
        return TargetCohortRole.PEER_ONLY
    return TargetCohortRole.REPORTED_TARGET


def _ctu13_cohort_role(client: DatasetClientId) -> TargetCohortRole:
    declared = tuple(scenario for scenario in Ctu13Scenario if scenario.client_name == client.name)
    if declared:
        return declared[0].cohort_role
    return TargetCohortRole.PEER_ONLY


def target_cohort_role(
    client: DatasetClientId, lab_network: LabNetwork | None = None
) -> TargetCohortRole:
    match client.dataset:
        case DatasetName.TON_IOT:
            return _ton_iot_cohort_role(client, lab_network)
        case DatasetName.IOT_23:
            return _iot23_cohort_role(client)
        case DatasetName.GOTHAM_2025:
            return _declared_cohort_role(client, GothamDevice)
        case DatasetName.CTU_13:
            return _ctu13_cohort_role(client)
        case DatasetName.N_BAIOT:
            return _declared_cohort_role(client, NBaIoTDevice)
        case DatasetName.UNSW_IOT_ATTACK_FLOWS:
            return TargetCohortRole.PEER_ONLY


def polars_text(member: StrEnum) -> SourceCell:
    return SourceCell(f"{member}")


def polars_texts(members: Iterable[StrEnum]) -> list[SourceCell]:
    return [polars_text(member) for member in members]


def privileged_port_indicator(port: pl.Expr) -> pl.Expr:
    return ((port > 0.0) & (port < 1024.0)).cast(pl.Float64)


def is_privileged_port(port: FeatureValue) -> FeatureValue:
    return FeatureValue(0.0 < port < 1024.0)


def has_regular_field_layout(block: Sequence[SourceLine], width: ObservationCount) -> bool:
    joined = LineBreak.LINE_FEED.token.join(block)
    collapsed = re.sub(rb'"[^"]*"', b'""', joined)
    return collapsed.count(FieldDelimiter.COMMA.token) == len(block) * (width - 1)


def feature_columns(feature_count: ObservationCount) -> tuple[SourceColumn, ...]:
    return tuple(SourceColumn(f"f{index}") for index in range(feature_count))


def flagged_positions(mask: BooleanArray) -> list[ObservationCount]:
    return np.flatnonzero(mask).tolist()


def first_seen_order(group_ids: RowIndexArray) -> list[ObservationCount]:
    return list(dict.fromkeys(group_ids.tolist()))


def decompression_workers() -> ObservationCount:
    return min(DecompressionWorkers.MAXIMUM, os.cpu_count() or 1)


def signed_log_feature_transform(
    features: FloatMatrix | Float32Matrix,
) -> FloatMatrix:
    if not np.isfinite(features).all():
        raise DatasetValidationError("Feature values must be finite before transformation")
    feature_values = features.astype(np.float32)
    negative_values = feature_values < 0.0
    np.abs(feature_values, out=feature_values)
    np.log1p(feature_values, out=feature_values)
    np.negative(feature_values, out=feature_values, where=negative_values)
    result = feature_values.astype(np.float64)
    if not np.isfinite(result).all():
        raise DatasetValidationError("Transformed features must remain finite")
    return result


def stable_timestamp_order(
    timestamps: TimestampValues, source_rows: RowIndexArray
) -> tuple[RowIndexArray, ChronologicalOrder]:
    if timestamps.ndim != 1 or source_rows.ndim != 1:
        raise DatasetValidationError("Chronology arrays must be one-dimensional")
    if timestamps.size != source_rows.size:
        raise DatasetValidationError("Chronology and source-order arrays must align")
    row_count = timestamps.size
    if row_count < 2:
        return np.arange(row_count, dtype=np.int64), ChronologicalOrder.ALREADY_ORDERED
    source_ordered = np.all(
        (timestamps[1:] > timestamps[:-1])
        | ((timestamps[1:] == timestamps[:-1]) & (source_rows[1:] >= source_rows[:-1]))
    )
    if source_ordered:
        return np.arange(row_count, dtype=np.int64), ChronologicalOrder.ALREADY_ORDERED
    return np.lexsort((source_rows, timestamps)), ChronologicalOrder.REORDERED


class _ReadAhead(io.RawIOBase):
    def __init__(
        self, source: SourceBinaryStream, chunk_bytes: ChunkSize, depth: QueueCapacity
    ) -> None:
        super().__init__()
        self._chunks: queue.Queue[SourceLine | BaseException] = queue.Queue(maxsize=depth)
        self._pending = SourceLine(b"")
        self._finished = False
        self._stopped = threading.Event()
        self._thread = threading.Thread(target=self._fill, args=(source, chunk_bytes), daemon=True)
        self._thread.start()

    def _has_accepted(self, item: SourceLine | BaseException) -> bool:
        try:
            self._chunks.put(item, timeout=PollInterval.SOURCE_QUEUE.interval.total_seconds())
        except queue.Full:
            return False
        return True

    def _publish(self, item: SourceLine | BaseException) -> None:
        accepted = False
        while not accepted and not self._stopped.is_set():
            accepted = self._has_accepted(item)

    def _fill(self, source: SourceBinaryStream, chunk_bytes: ChunkSize) -> None:
        completed = False
        try:
            while not self._stopped.is_set():
                item: SourceLine | BaseException
                try:
                    item = SourceLine(source.read(chunk_bytes))
                except (OSError, EOFError, ValueError, RuntimeError) as error:
                    item = error
                self._publish(item)
                if isinstance(item, BaseException) or not item:
                    completed = True
                    return
        finally:
            if not completed and not self._stopped.is_set():
                self._publish(RuntimeError("Read-ahead source failed unexpectedly"))

    def readable(self) -> bool:
        return True

    def readinto(self, buffer: Buffer) -> ObservationCount:
        view = memoryview(buffer).cast("B")
        while not self._pending and not self._finished:
            item = self._chunks.get()
            if isinstance(item, BaseException):
                self._finished = True
                raise item
            self._pending, self._finished = item, not item
        count = min(len(view), len(self._pending))
        view[:count] = self._pending[:count]
        self._pending = SourceLine(self._pending[count:])
        return count

    def close(self) -> None:
        self._stopped.set()
        while not self._chunks.empty():
            self._chunks.get_nowait()
        self._thread.join()
        super().close()


@contextmanager
def read_ahead(
    source: SourceBinaryStream,
    chunk_bytes: ChunkSize = StreamBufferSize.READ_AHEAD_CHUNK,
    depth: QueueCapacity = QueueDepth.READ_AHEAD,
) -> Generator[SourceBinaryStream]:
    reader = io.BufferedReader(
        _ReadAhead(source, chunk_bytes, depth), buffer_size=StreamBufferSize.READER
    )
    try:
        yield reader
    finally:
        reader.close()


def _without_blank_and_comment_lines(
    lines: list[SourceLine], data: SourceLine, comment: SourceLine | None
) -> list[SourceLine]:
    line_feed = LineBreak.LINE_FEED.token
    carriage_return = LineBreak.CARRIAGE_RETURN.token
    marker = comment or line_feed
    if not (
        data.startswith(marker)
        or line_feed + marker in data
        or line_feed + line_feed in data
        or carriage_return in data
    ):
        return lines
    return [
        line
        for line in lines
        if line.rstrip(carriage_return) and not (comment and line.startswith(comment))
    ]


def record_blocks(
    stream: SourceBinaryStream,
    block_rows: ObservationCount,
    comments: RecordCommentPolicy = RecordCommentPolicy.ZEEK,
    chunk_bytes: ChunkSize = StreamBufferSize.RECORD_CHUNK,
) -> Iterator[list[SourceLine]]:
    comment = comments.marker
    pending: list[SourceLine] = []
    remainder = SourceLine(b"")
    while True:
        chunk = SourceLine(stream.read(chunk_bytes))
        data = SourceLine(remainder + chunk)
        lines = [SourceLine(line) for line in data.split(LineBreak.LINE_FEED.token)]
        remainder = lines.pop() if chunk else SourceLine(b"")
        if not chunk and lines and not lines[-1]:
            lines.pop()
        pending.extend(_without_blank_and_comment_lines(lines, data, comment))
        while len(pending) >= block_rows:
            yield pending[:block_rows]
            del pending[:block_rows]
        if not chunk:
            break
    if pending:
        yield pending


def first_misaligned_line(
    lines: Sequence[SourceLine], separator: FieldDelimiter, counts: set[ObservationCount]
) -> ObservationCount | None:
    token = separator.token
    return next(
        (index for index, line in enumerate(lines) if line.count(token) not in counts), None
    )


def extend_reservoirs(
    buckets: Sequence[IntRowReservoir] | Sequence[FloatRowReservoir],
    group_ids: RowIndexArray,
    timestamps: TimestampValues,
    source_rows: RowIndexArray,
    features: Float32Matrix | FloatMatrix,
    generator: np.random.Generator,
    population_counting: PopulationCounting = PopulationCounting.COUNTED,
) -> None:
    if group_ids.size == 0:
        return
    capacities = np.asarray([bucket.capacity for bucket in buckets], dtype=np.int64)
    members = [np.flatnonzero(group_ids == index) for index in range(len(buckets))]
    positions = np.empty(group_ids.size, dtype=np.int64)
    for bucket, rows in zip(buckets, members, strict=True):
        positions[rows] = bucket.sampling_population_rows + np.arange(rows.size)
    row_capacity = capacities[group_ids]
    slots = np.full(group_ids.size, -1, dtype=np.int64)
    overflow = np.flatnonzero(positions >= row_capacity)
    if overflow.size:
        draws = generator.integers(0, positions[overflow] + 1)
        slots[overflow] = np.where(draws < row_capacity[overflow], draws, -1)
    for bucket, rows in zip(buckets, members, strict=True):
        if rows.size == 0:
            continue
        filling = rows[positions[rows] < bucket.capacity]
        bucket.timestamps.frombytes(
            np.ascontiguousarray(
                timestamps[filling], dtype=np.dtype(bucket.timestamps.typecode)
            ).tobytes()
        )
        bucket.source_rows.frombytes(
            np.ascontiguousarray(source_rows[filling], dtype=np.int64).tobytes()
        )
        bucket.features.frombytes(
            np.ascontiguousarray(features[filling], dtype=np.float32).tobytes()
        )
        replacing = rows[slots[rows] >= 0]
        if replacing.size:
            target = slots[replacing]
            np.frombuffer(bucket.timestamps, dtype=bucket.timestamps.typecode)[target] = timestamps[
                replacing
            ]
            np.frombuffer(bucket.source_rows, dtype=np.int64)[target] = source_rows[replacing]
            np.frombuffer(bucket.features, dtype=np.float32).reshape(-1, bucket.feature_width)[
                target
            ] = features[replacing]
        bucket.sampling_population_rows += rows.size
        if population_counting is PopulationCounting.COUNTED:
            bucket.source_population_rows += rows.size


def reservoir_columns(
    buckets: Sequence[IntRowReservoir | FloatRowReservoir],
) -> tuple[TimestampValues, RowIndexArray, Float32Matrix]:
    timestamps = np.concatenate(
        [np.frombuffer(bucket.timestamps, dtype=bucket.timestamps.typecode) for bucket in buckets]
    )
    source_rows = np.concatenate(
        [np.frombuffer(bucket.source_rows, dtype=np.int64) for bucket in buckets]
    )
    features = np.concatenate(
        [
            np.frombuffer(bucket.features, dtype=np.float32).reshape(-1, bucket.feature_width)
            for bucket in buckets
        ],
        axis=0,
    )
    return timestamps, source_rows, features


def ordered_reservoir_matrix(
    buckets: Sequence[IntRowReservoir | FloatRowReservoir],
) -> tuple[FloatMatrix, RowIndexArray]:
    timestamps, source_rows, features = reservoir_columns(buckets)
    order, chronology = stable_timestamp_order(timestamps, source_rows)
    matrix = signed_log_feature_transform(
        features if chronology is ChronologicalOrder.ALREADY_ORDERED else features[order]
    )
    matrix.setflags(write=False)
    return matrix, np.arange(timestamps.size, dtype=np.int64)


class _ClientDigestMaterial(FrozenStrictModel, frozen=True):
    dataset: DatasetName
    client: ClientName
    study_stratum: StudyStratum
    feature_schema: FeatureSchemaId
    feature_names: tuple[FeatureName, ...]
    target_cohort_role: TargetCohortRole
    benign_source_population_rows: ObservationCount | None


class _CaptureDigestMaterial(FrozenStrictModel, frozen=True):
    attack_type: AttackTypeName
    source_population_rows: ObservationCount | None


def _hash_array(
    hasher: LengthPrefixedHasher, values: Float32Matrix | FloatMatrix | RowIndexArray
) -> None:
    shape = FieldDelimiter.COMMA.text.join(f"{extent}" for extent in values.shape)
    hasher.framed(ArtifactPayload(values.dtype.name.encode(TextEncoding.ASCII)))
    hasher.framed(ArtifactPayload(f"[{shape}]".encode(TextEncoding.ASCII)))
    hasher.framed(ArtifactPayload(values.tobytes(order="C")))


def digest_dataset_clients(clients: tuple[DatasetClientData, ...]) -> ScientificDigest:
    hasher = LengthPrefixedHasher()
    for client in sorted(clients, key=lambda item: item.client.name):
        hasher.framed(
            canonical_json(
                _ClientDigestMaterial(
                    dataset=client.client.dataset,
                    client=client.client.name,
                    study_stratum=client.study_stratum,
                    feature_schema=client.feature_schema,
                    feature_names=client.feature_names,
                    target_cohort_role=client.target_cohort_role,
                    benign_source_population_rows=client.benign_source_population_rows,
                )
            )
        )
        _hash_array(hasher, client.benign_features)
        _hash_array(hasher, client.benign_source_order)
        for capture in sorted(client.attack_captures, key=lambda item: item.attack_type.identifier):
            hasher.framed(
                canonical_json(
                    _CaptureDigestMaterial(
                        attack_type=capture.attack_type.identifier,
                        source_population_rows=capture.source_population_rows,
                    )
                )
            )
            _hash_array(hasher, capture.features)
            _hash_array(hasher, capture.source_order)
    return hasher.hexdigest()


def parse_decimal(text: SourceCell) -> FeatureValue:
    return FeatureValue(float(text))


def parse_integer(text: SourceCell, base: IntegerBase = IntegerBase.DECIMAL) -> SourceInteger:
    return SourceInteger(int(text, base))


def _parse_integer_literal(text: SourceCell) -> FeatureValue | None:
    try:
        return FeatureValue(float(parse_integer(text, IntegerBase.AUTO)))
    except ValueError:
        return None


def parse_numeric_literal(text: SourceCell) -> FeatureValue | None:
    stripped = SourceCell(text.strip())
    if stripped in MissingValueToken.numeric():
        return FeatureValue(0.0)
    try:
        return parse_decimal(stripped)
    except ValueError:
        return _parse_integer_literal(stripped)
