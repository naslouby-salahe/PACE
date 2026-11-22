import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import structlog

from pace.datasets.common import parse_decimal, parse_integer, signed_log_feature_transform
from pace.types import (
    AttackCapture,
    AttackType,
    ColumnIndex,
    CorpusPath,
    CsvNewline,
    DatasetClientData,
    DatasetClientId,
    DatasetName,
    DatasetValidationError,
    FeatureName,
    FeatureSchemaId,
    FeatureValue,
    FloatMatrix,
    LogEvent,
    RowIndexArray,
    SourceCell,
    SourceInteger,
    StudyStratum,
    TargetCohortRole,
    TextEncoding,
    UnswAttackFamily,
    UnswColumn,
    UnswDevice,
    UnswFileSuffix,
)

logger = structlog.get_logger()


@dataclass(frozen=True, slots=True)
class _AttackInterval:
    start: SourceInteger
    end: SourceInteger
    attack_type: AttackType


def _flow_file(dataset_root: Path, device: UnswDevice) -> Path:
    return dataset_root / CorpusPath.UNSW_FLOWS / f"{device}{UnswFileSuffix.FLOW_STATS}"


def _annotation_file(dataset_root: Path, device: UnswDevice) -> Path:
    return dataset_root / CorpusPath.UNSW_ANNOTATIONS / f"{device}{UnswFileSuffix.ANNOTATIONS}"


def _read_intervals(annotation_file: Path) -> tuple[_AttackInterval, ...]:
    intervals: list[_AttackInterval] = []
    try:
        with annotation_file.open(
            newline=CsvNewline.UNIVERSAL, encoding=TextEncoding.UTF8_SIG
        ) as source:
            for line_number, row in enumerate(csv.reader(source), start=1):
                if len(row) < 4:
                    raise DatasetValidationError(
                        f"{annotation_file}:{line_number} has fewer than four annotation fields"
                    )
                start = parse_integer(SourceCell(row[0]))
                end = parse_integer(SourceCell(row[1]))
                attack_name = SourceCell(row[-1].strip())
                if start > end or not attack_name:
                    raise DatasetValidationError(
                        f"{annotation_file}:{line_number} has an invalid attack interval"
                    )
                family = UnswAttackFamily.from_annotation(attack_name)
                intervals.append(_AttackInterval(start, end, family.attack_type))
    except (OSError, ValueError) as error:
        raise DatasetValidationError(
            f"Could not read attack annotations from {annotation_file}"
        ) from error
    if not intervals:
        raise DatasetValidationError(f"No attack intervals found in {annotation_file}")
    return tuple(
        sorted(
            intervals,
            key=lambda value: (value.start, value.end, value.attack_type.identifier),
        )
    )


def _canonical_columns(
    header: list[SourceCell], source_file: Path, counters: tuple[UnswColumn, ...]
) -> tuple[tuple[FeatureName, ColumnIndex], ...]:
    names = tuple(SourceCell(name.strip()) for name in header)
    if names.count(UnswColumn.TIMESTAMP) != 1:
        raise DatasetValidationError(f"{source_file} must contain exactly one Timestamp column")
    positions: dict[SourceCell, ColumnIndex] = {}
    for index, name in enumerate(names):
        if name in positions:
            if name in counters:
                raise DatasetValidationError(f"{source_file} repeats source column {name}")
            continue
        positions[name] = ColumnIndex(index)
    mapped: list[tuple[FeatureName, ColumnIndex]] = []
    for counter in counters:
        if counter not in positions:
            raise DatasetValidationError(
                f"{source_file} lacks shared counter for canonical feature {counter.feature}"
            )
        mapped.append((counter.feature, positions[SourceCell(counter)]))
    return tuple(mapped)


@dataclass(frozen=True, slots=True)
class _FlowCounters:
    times: RowIndexArray
    features: FloatMatrix
    names: tuple[FeatureName, ...]


def _read_flow_counters(flow_file: Path, counters: tuple[UnswColumn, ...]) -> _FlowCounters:
    try:
        with flow_file.open(newline=CsvNewline.UNIVERSAL, encoding=TextEncoding.UTF8_SIG) as source:
            reader = csv.reader(source)
            header = [SourceCell(name) for name in next(reader)]
            columns = _canonical_columns(header, flow_file, counters)
            time_index = next(
                index for index, name in enumerate(header) if name.strip() == UnswColumn.TIMESTAMP
            )
            canonical_rows: list[list[FeatureValue]] = []
            source_times: list[SourceInteger] = []
            for line_number, row in enumerate(reader, start=2):
                if len(row) != len(header):
                    raise DatasetValidationError(
                        f"{flow_file}:{line_number} has the wrong column count"
                    )
                source_times.append(parse_integer(SourceCell(row[time_index])))
                canonical_rows.append(
                    [parse_decimal(SourceCell(row[index])) for _, index in columns]
                )
    except (OSError, StopIteration, ValueError) as error:
        raise DatasetValidationError(f"Could not read flow counters from {flow_file}") from error
    if not canonical_rows:
        raise DatasetValidationError(f"{flow_file} contains no flow-stat records")
    times = np.asarray(source_times, dtype=np.int64)
    if np.any(times[1:] < times[:-1]):
        raise DatasetValidationError(f"{flow_file} is not chronologically ordered")
    raw_features = np.asarray(canonical_rows, dtype=np.float64)
    if not np.isfinite(raw_features).all():
        raise DatasetValidationError(f"{flow_file} contains non-finite feature values")
    return _FlowCounters(
        times, signed_log_feature_transform(raw_features), tuple(feature for feature, _ in columns)
    )


def _row_labels(
    times: RowIndexArray, intervals: tuple[_AttackInterval, ...], annotation_file: Path
) -> list[set[AttackType]]:
    labels: list[set[AttackType]] = [set() for _ in range(times.size)]
    for interval in intervals:
        first = np.searchsorted(times, interval.start * 1000, side="left").item()
        stop = np.searchsorted(times, (interval.end + 1) * 1000, side="left").item()
        for row_index in range(first, stop):
            labels[row_index].add(interval.attack_type)
    ambiguous = tuple(index for index, row_labels in enumerate(labels) if len(row_labels) > 1)
    if ambiguous:
        raise DatasetValidationError(
            f"{annotation_file} assigns overlapping attack types to {len(ambiguous)} flow-stat rows"
        )
    return labels


def _attack_captures(
    flow_file: Path, features: FloatMatrix, labels: list[set[AttackType]]
) -> tuple[AttackCapture, ...]:
    attack_types = tuple(
        sorted(
            {label for row_labels in labels for label in row_labels},
            key=lambda item: item.identifier,
        )
    )
    captures: list[AttackCapture] = []
    for attack_type in attack_types:
        rows = np.fromiter((attack_type in item for item in labels), dtype=np.bool_)
        captures.append(
            AttackCapture(
                attack_type, flow_file, features[rows], np.flatnonzero(rows).astype(np.int64)
            )
        )
    return tuple(captures)


def load_unsw_device(data_root: Path, device: UnswDevice) -> DatasetClientData:
    dataset_root = data_root / CorpusPath.UNSW
    flow_file = _flow_file(dataset_root, device)
    annotation_file = _annotation_file(dataset_root, device)
    intervals = _read_intervals(annotation_file)
    flows = _read_flow_counters(flow_file, UnswColumn.counters(device.group))
    labels = _row_labels(flows.times, intervals, annotation_file)
    benign_indices = np.fromiter(
        (index for index, row_labels in enumerate(labels) if not row_labels),
        dtype=np.int64,
    )
    attacks = _attack_captures(flow_file, flows.features, labels)
    if benign_indices.size == 0 or not attacks:
        raise DatasetValidationError(f"{flow_file} needs benign and annotated attack records")
    client = DatasetClientId(DatasetName.UNSW_IOT_ATTACK_FLOWS, device.identifier)
    logger.info(
        LogEvent.ADAPTER_CLIENT_BUILT,
        dataset=DatasetName.UNSW_IOT_ATTACK_FLOWS.name,
        client=device.identifier,
        study_stratum=device.group.name,
        benign_rows=benign_indices.size,
        attack_types=len(attacks),
    )
    return DatasetClientData(
        client,
        device.group,
        FeatureSchemaId.UNSW_MUD_G1_MINUTE_COUNTERS
        if device.group is StudyStratum.UNSW_IOT_ATTACK_FLOWS_G1
        else FeatureSchemaId.UNSW_MUD_G2_MINUTE_COUNTERS,
        flows.names,
        flows.features[benign_indices],
        benign_indices,
        attacks,
        TargetCohortRole.REPORTED_TARGET,
    )


def _group_devices(group: StudyStratum | None) -> tuple[UnswDevice, ...]:
    devices = tuple(device for device in UnswDevice if group is None or device.group is group)
    if not devices:
        raise DatasetValidationError(f"No UNSW IoT devices are defined for study stratum {group}")
    return devices


def load_unsw_clients(
    data_root: Path, group: StudyStratum | None = None
) -> tuple[DatasetClientData, ...]:
    devices = _group_devices(group)
    logger.info(
        LogEvent.DATASET_LOADED,
        dataset=DatasetName.UNSW_IOT_ATTACK_FLOWS.name,
        device_count=len(devices),
    )
    return tuple(load_unsw_device(data_root, device) for device in devices)


def unsw_source_files(data_root: Path, group: StudyStratum | None = None) -> tuple[Path, ...]:
    dataset_root = data_root / CorpusPath.UNSW
    paths = tuple(
        path
        for device in _group_devices(group)
        for path in (_flow_file(dataset_root, device), _annotation_file(dataset_root, device))
    )
    if any(not path.is_file() for path in paths):
        raise DatasetValidationError(f"UNSW IoT source files are incomplete under {dataset_root}")
    return paths
