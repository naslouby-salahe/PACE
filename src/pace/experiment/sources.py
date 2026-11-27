import hashlib
import os
import shlex
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import cache
from pathlib import Path, PurePosixPath

import structlog

from pace.types import (
    ByteBlockSize,
    CriterionSatisfied,
    DatasetValidationError,
    DeviceNumber,
    DisplayText,
    FileDescriptor,
    InodeNumber,
    LengthPrefixedHasher,
    LogEvent,
    ReadStrategy,
    ScientificDigest,
    SourceBinaryStream,
    SourceByteCount,
    SourceFileName,
    SourceFileRecord,
    SourceIntegrityError,
    SourceVerification,
    UnixTimestampNanoseconds,
)

logger = structlog.get_logger()


@dataclass(frozen=True, slots=True)
class FileStamp:
    size: SourceByteCount
    modified_ns: UnixTimestampNanoseconds
    inode: InodeNumber
    device: DeviceNumber

    @classmethod
    def of(cls, status: os.stat_result) -> "FileStamp":
        return cls(
            status.st_size,
            UnixTimestampNanoseconds(status.st_mtime_ns),
            InodeNumber(status.st_ino),
            DeviceNumber(status.st_dev),
        )


@dataclass(frozen=True, slots=True)
class FileFingerprint:
    name: PurePosixPath
    digest: ScientificDigest
    before: FileStamp
    after: FileStamp
    bytes_read: SourceByteCount
    consistent: CriterionSatisfied
    segments: tuple[ScientificDigest, ...] = ()

    @property
    def is_stable(self) -> bool:
        return self.consistent and self.before == self.after and self.bytes_read == self.after.size

    @property
    def record(self) -> SourceFileRecord:
        return SourceFileRecord(
            name=SourceFileName(self.name.as_posix()), size=self.after.size, digest=self.digest
        )


@dataclass(frozen=True, slots=True)
class SourceBaseline:
    digest: ScientificDigest
    files: tuple[FileFingerprint, ...]
    verification: SourceVerification
    recorded_by: DisplayText
    root: Path

    @property
    def records(self) -> tuple[SourceFileRecord, ...]:
        return tuple(item.record for item in self.files)


def stamp_of(path: Path) -> FileStamp:
    try:
        return FileStamp.of(path.stat())
    except OSError as error:
        raise SourceIntegrityError(f"Dataset source file cannot be inspected: {path}") from error


def _evict_from_page_cache(
    descriptor: FileDescriptor, offset: SourceByteCount, length: SourceByteCount
) -> None:
    os.posix_fadvise(descriptor, offset, length, os.POSIX_FADV_DONTNEED)


def _read_stream(
    source: SourceBinaryStream, strategy: ReadStrategy
) -> tuple[ScientificDigest, SourceByteCount, tuple[ScientificDigest, ...]]:
    whole = hashlib.sha256()
    segment = hashlib.sha256()
    segments: list[ScientificDigest] = []
    descriptor = FileDescriptor(source.fileno())
    evicting = strategy is ReadStrategy.SEGMENTED_EVICTED
    if evicting:
        _evict_from_page_cache(descriptor, 0, 0)
    bytes_read = 0
    in_segment = 0
    while chunk := source.read(ByteBlockSize.DIGEST):
        whole.update(chunk)
        bytes_read += len(chunk)
        if strategy is ReadStrategy.STREAMING:
            continue
        segment.update(chunk)
        in_segment += len(chunk)
        if in_segment >= ByteBlockSize.SEGMENT:
            segments.append(segment.hexdigest())
            if evicting:
                _evict_from_page_cache(descriptor, bytes_read - in_segment, in_segment)
            segment = hashlib.sha256()
            in_segment = 0
    if in_segment:
        segments.append(segment.hexdigest())
    return whole.hexdigest(), bytes_read, tuple(segments)


def fingerprint_file(
    path: Path, root: Path, strategy: ReadStrategy = ReadStrategy.STREAMING
) -> FileFingerprint:
    before = stamp_of(path)
    try:
        with path.open("rb") as source:
            opened = FileStamp.of(os.fstat(source.fileno()))
            digest, bytes_read, segments = _read_stream(source, strategy)
            closed = FileStamp.of(os.fstat(source.fileno()))
    except OSError as error:
        raise SourceIntegrityError(f"Dataset source file cannot be read: {path}") from error
    logger.debug(LogEvent.SOURCE_READ, source_file=path.relative_to(root).as_posix(), digest=digest)
    return FileFingerprint(
        PurePosixPath(path.relative_to(root).as_posix()),
        digest,
        before,
        stamp_of(path),
        bytes_read,
        opened == before and closed == before,
        segments,
    )


def combined_digest(files: Sequence[FileFingerprint]) -> ScientificDigest:
    hasher = LengthPrefixedHasher()
    for item in files:
        hasher.framed_path(item.name)
        hasher.framed_label(item.digest)
    return hasher.hexdigest()


@cache
def _remembered_fingerprint(path: Path, root: Path, stamp: FileStamp) -> FileFingerprint:
    item = fingerprint_file(path, root)
    if not item.is_stable or item.before != stamp:
        raise SourceIntegrityError(
            f"Dataset source file changed while it was being hashed: {item.name.as_posix()} "
            f"(size {item.before.size} to {item.after.size}, modification time "
            f"{item.before.modified_ns} to {item.after.modified_ns} ns, bytes read "
            f"{item.bytes_read})"
        )
    return item


def _fingerprints(source_files: Sequence[Path], data_root: Path) -> tuple[FileFingerprint, ...]:
    resolved_root = data_root.resolve()
    resolved_files = {path.resolve() for path in source_files}
    if (
        not source_files
        or len(resolved_files) != len(source_files)
        or any(
            not path.is_file() or not path.is_relative_to(resolved_root) for path in resolved_files
        )
    ):
        raise DatasetValidationError(
            "Dataset source files must be unique and exist under the configured raw data root"
        )
    return tuple(
        _remembered_fingerprint(path, data_root, stamp_of(path)) for path in sorted(source_files)
    )


def _current_process() -> DisplayText:
    return DisplayText(
        f"process {os.getpid()} at {datetime.now(UTC).isoformat(timespec='seconds')}; "
        f"command: {shlex.join(sys.argv)}"
    )


def digest_source_files(source_files: Sequence[Path], data_root: Path) -> ScientificDigest:
    return combined_digest(_fingerprints(source_files, data_root))


def fingerprint_source_files(
    source_files: Sequence[Path], data_root: Path, verification: SourceVerification
) -> SourceBaseline:
    files = _fingerprints(source_files, data_root)
    return SourceBaseline(
        combined_digest(files), files, verification, _current_process(), data_root
    )


def baseline_source_files(
    source_files: Sequence[Path], data_root: Path, verification: SourceVerification
) -> SourceBaseline:
    baseline = fingerprint_source_files(source_files, data_root, verification)
    logger.info(
        LogEvent.SOURCE_BASELINE_RECORDED,
        digest=baseline.digest,
        verification=verification.name,
        file_count=len(baseline.files),
        total_bytes=sum(item.after.size for item in baseline.files),
        recorded_by=baseline.recorded_by,
    )
    return baseline
