import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from pace.types import (
    ByteBlockSize,
    DatasetValidationError,
    ExecutionPhase,
    LengthPrefixedHasher,
    ScientificDigest,
)


@dataclass(frozen=True, slots=True)
class SourceBaseline:
    digest: ScientificDigest
    files: tuple[tuple[PurePosixPath, ScientificDigest], ...]


def _file_digests(
    paths: Sequence[Path], root: Path
) -> tuple[tuple[PurePosixPath, ScientificDigest], ...]:
    entries: list[tuple[PurePosixPath, ScientificDigest]] = []
    for path in sorted(paths):
        hasher = hashlib.sha256()
        with path.open("rb") as source:
            while chunk := source.read(ByteBlockSize.DIGEST):
                hasher.update(chunk)
        entries.append((PurePosixPath(path.relative_to(root).as_posix()), hasher.hexdigest()))
    return tuple(entries)


def _source_baseline(paths: Sequence[Path], root: Path) -> SourceBaseline:
    files = _file_digests(paths, root)
    hasher = LengthPrefixedHasher()
    for name, digest in files:
        hasher.framed_path(name)
        hasher.framed_label(digest)
    return SourceBaseline(hasher.hexdigest(), files)


def digest_files(paths: Sequence[Path], root: Path) -> ScientificDigest:
    return _source_baseline(paths, root).digest


def _require_valid_sources(source_files: Sequence[Path], data_root: Path) -> None:
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


def digest_source_files(source_files: Sequence[Path], data_root: Path) -> ScientificDigest:
    _require_valid_sources(source_files, data_root)
    return digest_files(source_files, data_root)


def baseline_source_files(source_files: Sequence[Path], data_root: Path) -> SourceBaseline:
    _require_valid_sources(source_files, data_root)
    return _source_baseline(source_files, data_root)


def require_unchanged_sources(
    source_files: tuple[Path, ...],
    data_root: Path,
    baseline: SourceBaseline,
    phase: ExecutionPhase = ExecutionPhase.CAMPAIGN,
) -> None:
    current = _source_baseline(source_files, data_root)
    if current.digest == baseline.digest:
        return
    recorded = dict(baseline.files)
    changed = sorted(
        name.as_posix() for name, digest in current.files if recorded.get(name) != digest
    )
    raise DatasetValidationError(
        f"Dataset source files changed during {phase} execution: "
        f"{', '.join(changed) or 'file set differs'} (digest {baseline.digest[:12]} became "
        f"{current.digest[:12]})"
    )
