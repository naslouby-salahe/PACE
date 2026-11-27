import dataclasses
import os
import threading
from collections.abc import Generator, MutableMapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from enum import IntEnum
from pathlib import Path, PurePosixPath

import pytest
import structlog.testing

import pace.experiment.source_integrity as integrity
import pace.experiment.sources as sources
from pace.experiment.source_integrity import (
    describe_input_difference,
    require_stable_manifest,
    require_unchanged_sources,
)
from pace.experiment.sources import (
    FileFingerprint,
    SourceBaseline,
    baseline_source_files,
    digest_source_files,
    fingerprint_file,
    fingerprint_source_files,
    stamp_of,
)
from pace.types import (
    ExecutionPhase,
    LogEvent,
    ReadStrategy,
    SourceFinding,
    SourceIntegrityError,
    SourceVerification,
    short_digest,
)

WRONG = "f" * 64
OTHER = "e" * 64
THIRD = "d" * 64


@contextmanager
def _captured(monkeypatch: pytest.MonkeyPatch) -> Generator[list[MutableMapping[str, object]]]:
    with structlog.testing.capture_logs() as logged:
        monkeypatch.setattr(sources, "logger", structlog.get_logger())
        monkeypatch.setattr(integrity, "logger", structlog.get_logger())
        yield logged


def _baseline(
    files: Sequence[Path],
    root: Path,
    verification: SourceVerification = SourceVerification.STRICT,
) -> SourceBaseline:
    return baseline_source_files(files, root, verification)


def _corpus(root: Path) -> tuple[Path, ...]:
    first = root / "corpus" / "a.csv"
    second = root / "corpus" / "b.csv"
    first.parent.mkdir(parents=True)
    first.write_bytes(b"alpha" * 4000)
    second.write_bytes(b"beta" * 4000)
    return first, second


def _rewrite_same_size(path: Path) -> None:
    content = bytearray(path.read_bytes())
    content[7] ^= 0x01
    path.write_bytes(bytes(content))


def _interfere_while_hashing(monkeypatch: pytest.MonkeyPatch, path: Path) -> None:
    real_sha256 = sources.hashlib.sha256

    class Interfering:
        def __init__(self) -> None:
            self._inner = real_sha256()
            self._fired = False

        def update(self, chunk: bytes) -> None:
            if not self._fired:
                self._fired = True
                with path.open("ab") as handle:
                    handle.write(b"appended while hashing")
            self._inner.update(chunk)

        def hexdigest(self) -> str:
            return self._inner.hexdigest()

    monkeypatch.setattr(sources.hashlib, "sha256", Interfering)


def _scripted_reads(monkeypatch: pytest.MonkeyPatch, digests: Sequence[str]) -> None:
    calls: list[int] = []
    real = integrity.fingerprint_file

    def scripted(
        path: Path, root: Path, strategy: ReadStrategy = ReadStrategy.STREAMING
    ) -> FileFingerprint:
        calls.append(1)
        return dataclasses.replace(real(path, root, strategy), digest=digests[len(calls) - 1])

    monkeypatch.setattr(integrity, "fingerprint_file", scripted)


class _SmallBlocks(IntEnum):
    DIGEST = 4096
    SEGMENT = 8192


def test_fingerprint_fields(tmp_path: Path) -> None:
    first, _ = _corpus(tmp_path)

    item = fingerprint_file(first, tmp_path)

    assert item.name == PurePosixPath("corpus/a.csv")
    assert item.is_stable
    assert item.bytes_read == first.stat().st_size == item.after.size
    assert item.before == item.after == stamp_of(first)
    assert item.before.inode == first.stat().st_ino
    assert item.record.digest == item.digest
    assert item.record.size == first.stat().st_size


def test_changed_content_is_reported(tmp_path: Path) -> None:
    first, second = _corpus(tmp_path)
    baseline = _baseline((first, second), tmp_path)
    _rewrite_same_size(second)

    with pytest.raises(SourceIntegrityError) as caught:
        require_unchanged_sources((first, second), tmp_path, baseline, ExecutionPhase.MATRIX)

    message = str(caught.value)
    assert short_digest(baseline.digest) in message
    assert baseline.files[1].digest in message
    assert fingerprint_file(second, tmp_path).digest in message
    assert "corpus/b.csv" in message
    assert "corpus/a.csv" not in message


def test_unchanged_sources_pass_without_rehashing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus = _corpus(tmp_path)
    baseline = _baseline(corpus, tmp_path)
    require_unchanged_sources(corpus, tmp_path, baseline)

    def refuse_to_hash(*_arguments: object) -> FileFingerprint:
        raise AssertionError("an intact manifest must not require rehashing")

    monkeypatch.setattr(integrity, "fingerprint_file", refuse_to_hash)
    require_stable_manifest(corpus, tmp_path, baseline)


def test_touched_file_is_not_a_change(tmp_path: Path) -> None:
    corpus = _corpus(tmp_path)
    baseline = _baseline(corpus, tmp_path)
    os.utime(corpus[0], ns=(1_700_000_000_000_000_000, 1_700_000_000_000_000_000))

    require_stable_manifest(corpus, tmp_path, baseline)


def test_stamp_change_is_classified(tmp_path: Path) -> None:
    corpus = _corpus(tmp_path)
    baseline = _baseline(corpus, tmp_path)
    corpus[0].write_bytes(b"entirely different")

    with pytest.raises(SourceIntegrityError) as caught:
        require_stable_manifest(corpus, tmp_path, baseline)

    assert SourceFinding.STAMP_CHANGED in str(caught.value)
    assert "page-cache read" not in str(caught.value)


def test_added_and_removed_files_are_named(tmp_path: Path) -> None:
    first, second = _corpus(tmp_path)
    extra = first.with_name("c.csv")
    extra.write_bytes(b"gamma")
    baseline = _baseline((first, second), tmp_path)
    second.unlink()

    with pytest.raises(SourceIntegrityError) as caught:
        require_unchanged_sources((first, extra), tmp_path, baseline)

    message = str(caught.value)
    assert "corpus/b.csv" in message
    assert "corpus/c.csv" in message
    assert "corpus/a.csv" not in message


def test_file_modified_while_hashed_is_unstable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first, _ = _corpus(tmp_path)
    _interfere_while_hashing(monkeypatch, first)
    with pytest.raises(SourceIntegrityError):
        digest_source_files((first,), tmp_path)
    item = fingerprint_file(first, tmp_path)
    assert not item.is_stable
    assert item.before.size != item.after.size


def test_modification_during_check_has_no_confirmations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first, _ = _corpus(tmp_path)
    baseline = _baseline((first,), tmp_path)
    _interfere_while_hashing(monkeypatch, first)

    with pytest.raises(SourceIntegrityError) as caught:
        require_unchanged_sources((first,), tmp_path, baseline)

    assert SourceFinding.CHANGED_WHILE_HASHING in str(caught.value)
    assert "page-cache read" not in str(caught.value)


@pytest.mark.parametrize(
    ("reads", "finding"),
    [
        (("wrong", "baseline", "baseline"), SourceFinding.TRANSIENT),
        (("wrong", "wrong", "baseline"), SourceFinding.TRANSIENT),
        (("wrong", "wrong", "wrong"), SourceFinding.REPRODUCED),
        (("wrong", "wrong", "other"), SourceFinding.UNREPEATABLE),
        (("wrong", "other", "other"), SourceFinding.UNREPEATABLE),
        (("wrong", "other", "third"), SourceFinding.UNREPEATABLE),
    ],
)
def test_stable_mismatch_is_classified_by_reads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    reads: tuple[str, str, str],
    finding: SourceFinding,
) -> None:
    first, _ = _corpus(tmp_path)
    baseline = _baseline((first,), tmp_path)
    named = {"baseline": baseline.files[0].digest, "wrong": WRONG, "other": OTHER, "third": THIRD}
    digests = [named[read] for read in reads]
    _scripted_reads(monkeypatch, digests)

    with pytest.raises(SourceIntegrityError) as caught:
        require_unchanged_sources((first,), tmp_path, baseline)

    message = str(caught.value)
    assert finding in message
    assert "corpus/a.csv" in message
    assert all(digest in message for digest in set(digests))


def test_page_cache_error_is_attributed_by_evicted_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first, _ = _corpus(tmp_path)
    baseline = _baseline((first,), tmp_path)
    real = integrity.fingerprint_file
    calls: list[ReadStrategy] = []
    monkeypatch.setattr(integrity, "ByteBlockSize", _SmallBlocks)

    def flaky(
        path: Path, root: Path, strategy: ReadStrategy = ReadStrategy.STREAMING
    ) -> FileFingerprint:
        calls.append(strategy)
        item = real(path, root, strategy)
        if strategy is ReadStrategy.SEGMENTED_EVICTED:
            return dataclasses.replace(item, segments=("a" * 64, "b" * 64))
        return dataclasses.replace(item, digest="c" * 64, segments=("a" * 64, "c" * 64))

    monkeypatch.setattr(integrity, "fingerprint_file", flaky)
    with pytest.raises(SourceIntegrityError) as caught:
        require_unchanged_sources((first,), tmp_path, baseline)

    assert calls == [
        ReadStrategy.STREAMING,
        ReadStrategy.SEGMENTED,
        ReadStrategy.SEGMENTED_EVICTED,
    ]
    assert SourceFinding.TRANSIENT in str(caught.value)
    assert "1 (bytes 8192-16383)" in str(caught.value)


@pytest.mark.parametrize(
    ("reads", "finding"),
    [
        (("recorded", "recorded"), SourceFinding.TRANSIENT),
        (("operation", "operation"), SourceFinding.REPRODUCED),
        (("operation", "other"), SourceFinding.UNREPEATABLE),
    ],
)
def test_input_difference_classifies_the_reads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    reads: tuple[str, str],
    finding: SourceFinding,
) -> None:
    corpus = _corpus(tmp_path)
    recorded = _baseline(corpus, tmp_path).records
    _rewrite_same_size(corpus[1])
    operation = fingerprint_source_files(corpus, tmp_path, SourceVerification.STRICT)
    named = {"recorded": recorded[1].digest, "operation": operation.files[1].digest, "other": OTHER}
    _scripted_reads(monkeypatch, [named[read] for read in reads])

    description = describe_input_difference(recorded, operation)

    assert "corpus/b.csv" in description
    assert finding in description


def test_input_difference_names_each_changed_file(tmp_path: Path) -> None:
    first, second = _corpus(tmp_path)
    recorded = _baseline((first, second), tmp_path).records
    _rewrite_same_size(first)
    third = first.with_name("c.csv")
    third.write_bytes(b"new")
    operation = fingerprint_source_files((first, third), tmp_path, SourceVerification.STRICT)

    description = describe_input_difference(recorded, operation)

    assert "corpus/a.csv" in description
    assert "corpus/b.csv" in description
    assert "corpus/c.csv" in description
    assert recorded[0].digest in description
    assert operation.files[0].digest in description
    assert "no source manifest" in describe_input_difference((), operation)


def test_development_mismatch_is_logged_not_raised(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus = _corpus(tmp_path)
    baseline = _baseline(corpus, tmp_path, SourceVerification.DEVELOPMENT)
    _rewrite_same_size(corpus[1])

    with _captured(monkeypatch) as logged:
        require_unchanged_sources(corpus, tmp_path, baseline, ExecutionPhase.MATRIX)

    (mismatch,) = (entry for entry in logged if entry["event"] == LogEvent.SOURCE_MISMATCH)
    assert mismatch["log_level"] == "warning"
    assert mismatch["expected_digest"] == baseline.digest
    assert mismatch["observed_digest"] != baseline.digest


def test_strict_mismatch_is_logged_before_raising(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus = _corpus(tmp_path)
    baseline = _baseline(corpus, tmp_path)
    _rewrite_same_size(corpus[0])

    with _captured(monkeypatch) as logged, pytest.raises(SourceIntegrityError):
        require_unchanged_sources(corpus, tmp_path, baseline)

    (mismatch,) = (entry for entry in logged if entry["event"] == LogEvent.SOURCE_MISMATCH)
    assert mismatch["log_level"] == "error"
    assert mismatch["expected_digest"] == baseline.digest


def test_baseline_records_manifest_and_event(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus = _corpus(tmp_path)

    with _captured(monkeypatch) as logged:
        baseline = _baseline(corpus, tmp_path)

    assert baseline.verification is SourceVerification.STRICT
    assert [item.name.as_posix() for item in baseline.files] == ["corpus/a.csv", "corpus/b.csv"]
    assert [record.size for record in baseline.records] == [20000, 16000]
    recorded = next(
        entry for entry in logged if entry["event"] == LogEvent.SOURCE_BASELINE_RECORDED
    )
    assert recorded["digest"] == baseline.digest
    assert recorded["file_count"] == 2


def test_fingerprints_are_cached_per_unchanged_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus = _corpus(tmp_path)
    reads: list[Path] = []
    real = sources.fingerprint_file

    def counting(path: Path, root: Path) -> FileFingerprint:
        reads.append(path)
        return real(path, root)

    monkeypatch.setattr(sources, "fingerprint_file", counting)
    sources._remembered_fingerprint.cache_clear()  # pyright: ignore[reportPrivateUsage]

    first = digest_source_files(corpus, tmp_path)
    assert _baseline(corpus, tmp_path).digest == first
    assert len(reads) == 2

    corpus[0].write_bytes(b"changed content with another size")
    assert digest_source_files(corpus, tmp_path) != first
    assert len(reads) == 3


@pytest.mark.parametrize("modified", [False, True])
def test_concurrent_verifiers_agree(tmp_path: Path, modified: bool) -> None:
    corpus = _corpus(tmp_path)
    baseline = _baseline(corpus, tmp_path)
    if modified:
        _rewrite_same_size(corpus[1])
    gate = threading.Barrier(6)

    def verify(_worker: int) -> bool:
        gate.wait()
        try:
            require_unchanged_sources(corpus, tmp_path, baseline)
        except SourceIntegrityError as error:
            return "corpus/b.csv" in str(error)
        return not modified

    with ThreadPoolExecutor(max_workers=6) as pool:
        assert all(pool.map(verify, range(6)))


def test_unreadable_sources_raise_integrity_error(tmp_path: Path) -> None:
    with pytest.raises(SourceIntegrityError):
        stamp_of(tmp_path / "missing.csv")
    directory = tmp_path / "folder"
    directory.mkdir()
    with pytest.raises(SourceIntegrityError):
        fingerprint_file(directory, tmp_path)


def test_segmented_reads_agree_with_streaming(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus = _corpus(tmp_path)
    monkeypatch.setattr(sources, "ByteBlockSize", _SmallBlocks)
    streaming = fingerprint_file(corpus[0], tmp_path)
    cached = fingerprint_file(corpus[0], tmp_path, ReadStrategy.SEGMENTED)
    evicted = fingerprint_file(corpus[0], tmp_path, ReadStrategy.SEGMENTED_EVICTED)

    assert streaming.segments == ()
    assert cached.digest == evicted.digest == streaming.digest
    assert len(cached.segments) == 3
    assert cached.segments == evicted.segments
