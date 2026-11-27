from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import structlog

from pace.experiment.sources import (
    FileFingerprint,
    FileStamp,
    SourceBaseline,
    combined_digest,
    fingerprint_file,
    stamp_of,
)
from pace.types import (
    ByteBlockSize,
    DisplayText,
    ExecutionPhase,
    LogEvent,
    ReadStrategy,
    ScientificDigest,
    SegmentIndex,
    SourceChange,
    SourceFileRecord,
    SourceFinding,
    SourceIntegrityError,
    SourceVerification,
    TextSeparator,
    short_digest,
)

logger = structlog.get_logger()


@dataclass(frozen=True, slots=True)
class FileDifference:
    name: PurePosixPath
    change: SourceChange
    finding: SourceFinding | None
    expected: ScientificDigest | None
    observed: FileFingerprint | None
    reads: tuple[FileFingerprint, FileFingerprint] | None = None


def _stacked(lines: Iterable[DisplayText]) -> DisplayText:
    return DisplayText(TextSeparator.LINE.join(lines))


def _listed(items: Iterable[DisplayText]) -> DisplayText:
    return DisplayText(TextSeparator.LIST.join(items))


def _stamp_text(stamp: FileStamp) -> DisplayText:
    return DisplayText(f"size {stamp.size}, mtime_ns {stamp.modified_ns}, inode {stamp.inode}")


def _differing_segments(
    first: FileFingerprint, second: FileFingerprint
) -> tuple[SegmentIndex, ...]:
    longer = max(len(first.segments), len(second.segments))
    return tuple(
        SegmentIndex(index)
        for index in range(longer)
        if first.segments[index : index + 1] != second.segments[index : index + 1]
    )


def _segment_text(segments: tuple[SegmentIndex, ...]) -> DisplayText:
    size = ByteBlockSize.SEGMENT
    return _listed(
        DisplayText(f"{index} (bytes {index * size}-{(index + 1) * size - 1})")
        for index in segments
    ) or DisplayText("no difference")


def _verdict(
    read: FileFingerprint, expected: ScientificDigest | None, observed: FileFingerprint | None
) -> DisplayText:
    if read.digest == expected:
        return DisplayText("matches expected")
    if observed is not None and read.digest == observed.digest:
        return DisplayText("repeats observed")
    return DisplayText("differs from both")


def _difference_lines(difference: FileDifference) -> list[DisplayText]:
    finding = "" if difference.finding is None else f": {difference.finding}"
    lines = [DisplayText(f"  {difference.name.as_posix()}: {difference.change}{finding}")]
    if difference.expected is not None:
        lines.append(DisplayText(f"    expected sha256 {difference.expected}"))
    observed = difference.observed
    if observed is not None:
        lines.append(
            DisplayText(f"    observed sha256 {observed.digest} ({_stamp_text(observed.after)})")
        )
    if difference.reads is not None:
        cached, evicted = difference.reads
        for label, read in (("page-cache read", cached), ("evicted read", evicted)):
            verdict = _verdict(read, difference.expected, observed)
            lines.append(DisplayText(f"    {label} sha256 {read.digest} ({verdict})"))
        differing = _segment_text(_differing_segments(cached, evicted))
        lines.append(DisplayText(f"    segments differing between reads: {differing}"))
    return lines


def _classify(
    expected: ScientificDigest,
    observed: ScientificDigest,
    reads: tuple[FileFingerprint, FileFingerprint],
) -> SourceFinding:
    if any(read.digest == expected for read in reads):
        return SourceFinding.TRANSIENT
    if all(read.digest == observed for read in reads):
        return SourceFinding.REPRODUCED
    return SourceFinding.UNREPEATABLE


def _confirmed(root: Path, expected: ScientificDigest, observed: FileFingerprint) -> FileDifference:
    path = root / observed.name
    reads = (
        fingerprint_file(path, root, ReadStrategy.SEGMENTED),
        fingerprint_file(path, root, ReadStrategy.SEGMENTED_EVICTED),
    )
    finding = _classify(expected, observed.digest, reads)
    return FileDifference(observed.name, SourceChange.CHANGED, finding, expected, observed, reads)


def _differences(
    expected: Mapping[PurePosixPath, ScientificDigest],
    observed: Mapping[PurePosixPath, FileFingerprint],
    check: Callable[[PurePosixPath, ScientificDigest, FileFingerprint], FileDifference | None],
) -> tuple[FileDifference, ...]:
    differences: list[FileDifference] = []
    for name in sorted(expected.keys() | observed.keys()):
        if name not in observed:
            differences.append(
                FileDifference(name, SourceChange.REMOVED, None, expected[name], None)
            )
        elif name not in expected:
            differences.append(FileDifference(name, SourceChange.ADDED, None, None, observed[name]))
        elif (difference := check(name, expected[name], observed[name])) is not None:
            differences.append(difference)
    return tuple(differences)


def describe_input_difference(
    recorded: Sequence[SourceFileRecord], operation: SourceBaseline
) -> DisplayText:
    differences = _differences(
        {PurePosixPath(item.name): item.digest for item in recorded},
        {item.name: item for item in operation.files},
        lambda _, expected, observed: (
            None if observed.digest == expected else _confirmed(operation.root, expected, observed)
        ),
    )
    lines = [
        DisplayText(
            f"operation fingerprint recorded by {operation.recorded_by}; expected digests "
            "come from the artifact manifest"
        )
    ]
    if not recorded:
        lines.append(DisplayText("  the artifact carries no source manifest to compare with"))
    for difference in differences:
        lines.extend(_difference_lines(difference))
    return _stacked(lines)


def _conclude(
    phase: ExecutionPhase,
    baseline: SourceBaseline,
    observed_digest: ScientificDigest,
    differences: tuple[FileDifference, ...],
) -> None:
    names = _listed(DisplayText(item.name.as_posix()) for item in differences) or "file set differs"
    message = _stacked(
        [
            DisplayText(
                f"Dataset source files did not match their baseline during {phase} execution: "
                f"{names} (expected {short_digest(baseline.digest)}, observed "
                f"{short_digest(observed_digest)})"
            ),
            DisplayText(
                f"  expected {baseline.digest}, observed {observed_digest}; baseline recorded by "
                f"{baseline.recorded_by}"
            ),
            *(line for item in differences for line in _difference_lines(item)),
        ]
    )
    strict = baseline.verification is SourceVerification.STRICT
    log: Callable[..., None] = logger.error if strict else logger.warning
    log(
        LogEvent.SOURCE_MISMATCH,
        phase=phase.name,
        verification=baseline.verification.name,
        expected_digest=baseline.digest,
        observed_digest=observed_digest,
        detail=message,
    )
    if strict:
        raise SourceIntegrityError(message)


def _relative(path: Path, root: Path) -> PurePosixPath:
    return PurePosixPath(path.relative_to(root).as_posix())


def require_unchanged_sources(
    source_files: Sequence[Path],
    data_root: Path,
    baseline: SourceBaseline,
    phase: ExecutionPhase = ExecutionPhase.CAMPAIGN,
) -> None:
    stamps = {item.name: item.after for item in baseline.files}

    def check(
        name: PurePosixPath, expected: ScientificDigest, observed: FileFingerprint
    ) -> FileDifference | None:
        if not observed.is_stable:
            finding = SourceFinding.CHANGED_WHILE_HASHING
        elif observed.digest == expected:
            return None
        elif observed.before != stamps[name]:
            finding = SourceFinding.STAMP_CHANGED
        else:
            return _confirmed(data_root, expected, observed)
        return FileDifference(observed.name, SourceChange.CHANGED, finding, expected, observed)

    observed = {
        _relative(path, data_root): fingerprint_file(path, data_root)
        for path in sorted(source_files)
        if path.is_file()
    }
    differences = _differences({item.name: item.digest for item in baseline.files}, observed, check)
    observed_digest = combined_digest(tuple(observed[name] for name in sorted(observed)))
    if differences or observed_digest != baseline.digest:
        _conclude(phase, baseline, observed_digest, differences)


def require_stable_manifest(
    source_files: Sequence[Path],
    data_root: Path,
    baseline: SourceBaseline,
    phase: ExecutionPhase = ExecutionPhase.CAMPAIGN,
) -> None:
    observed = {
        _relative(path, data_root): stamp_of(path) for path in source_files if path.is_file()
    }
    if len(observed) != len(source_files) or observed != {
        item.name: item.after for item in baseline.files
    }:
        require_unchanged_sources(source_files, data_root, baseline, phase)
