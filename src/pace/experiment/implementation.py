import contextlib
import importlib.metadata
import os
import platform
import sys
from collections.abc import Iterator
from functools import cache
from pathlib import Path, PurePosixPath

import torch

from pace.types import (
    ArtifactConflictError,
    ArtifactPayload,
    DigestComponentName,
    DigestNamespace,
    EnvironmentFile,
    EnvironmentVariable,
    FrozenStrictModel,
    ImplementationScope,
    LengthPrefixedHasher,
    MachineArchitecture,
    MissingRuntimePackage,
    NumericalPolicyError,
    NumericalThreadCount,
    PlatformName,
    PythonVersionText,
    RuntimePackage,
    RuntimePackageVersion,
    ScientificDigest,
    ThreadSetting,
    canonical_json,
    payload_digest,
)


class RuntimeEnvironment(FrozenStrictModel, frozen=True):
    python: PythonVersionText
    platform: PlatformName
    machine: MachineArchitecture
    distributions: dict[RuntimePackage, RuntimePackageVersion | MissingRuntimePackage]
    thread_environment: dict[EnvironmentVariable, ThreadSetting]
    torch_intraop_threads: NumericalThreadCount


def _runtime_environment() -> RuntimeEnvironment:
    versions: dict[RuntimePackage, RuntimePackageVersion | MissingRuntimePackage] = {}
    for package in RuntimePackage:
        try:
            versions[package] = RuntimePackageVersion(importlib.metadata.version(package))
        except importlib.metadata.PackageNotFoundError:
            versions[package] = MissingRuntimePackage.UNAVAILABLE
    return RuntimeEnvironment(
        python=PythonVersionText(sys.version),
        platform=PlatformName(platform.system()),
        machine=MachineArchitecture(platform.machine()),
        distributions=versions,
        thread_environment={
            variable: ThreadSetting(os.environ.get(variable, ""))
            for variable in (
                EnvironmentVariable.OMP_THREADS,
                EnvironmentVariable.MKL_THREADS,
                EnvironmentVariable.OPENBLAS_THREADS,
            )
        },
        torch_intraop_threads=torch.get_num_threads(),
    )


def _implementation_components(
    scope: ImplementationScope,
) -> Iterator[tuple[DigestComponentName, ArtifactPayload]]:
    package_root = Path(__file__).resolve().parents[1]
    excluded = scope.excluded_sources
    for source_path in sorted(package_root.rglob("*.py")):
        relative_path = PurePosixPath(source_path.relative_to(package_root).as_posix())
        if not any(relative_path.is_relative_to(prefix) for prefix in excluded):
            yield (
                DigestComponentName(f"{DigestNamespace.SOURCE}/{relative_path}"),
                ArtifactPayload(source_path.read_bytes()),
            )
    repository_root = next(
        (
            parent
            for parent in package_root.parents
            if (parent / EnvironmentFile.PYPROJECT).is_file()
        ),
        None,
    )
    if repository_root is not None:
        for environment_file in (EnvironmentFile.PYPROJECT, EnvironmentFile.LOCK):
            configuration_path = repository_root / environment_file
            if configuration_path.is_file():
                yield (
                    DigestComponentName(f"{DigestNamespace.ENVIRONMENT}/{environment_file}"),
                    ArtifactPayload(configuration_path.read_bytes()),
                )
    yield (
        DigestComponentName(f"{DigestNamespace.ENVIRONMENT}/{EnvironmentFile.RUNTIME}"),
        canonical_json(_runtime_environment()),
    )


def _component_digests(
    scope: ImplementationScope,
) -> tuple[tuple[DigestComponentName, ScientificDigest], ...]:
    return tuple(
        (name, payload_digest(content)) for name, content in _implementation_components(scope)
    )


@cache
def _implementation_snapshot(
    scope: ImplementationScope,
) -> tuple[tuple[DigestComponentName, ScientificDigest], ...]:
    return _component_digests(scope)


def _combined_digest(
    components: tuple[tuple[DigestComponentName, ScientificDigest], ...],
) -> ScientificDigest:
    hasher = LengthPrefixedHasher()
    for name, digest in components:
        hasher.framed_label(name)
        hasher.framed_label(digest)
    return hasher.hexdigest()


def digest_implementation(
    scope: ImplementationScope = ImplementationScope.CAMPAIGN,
) -> ScientificDigest:
    return _combined_digest(_implementation_snapshot(scope))


def require_implementation_unchanged(
    scope: ImplementationScope = ImplementationScope.CAMPAIGN,
) -> None:
    snapshot = dict(_implementation_snapshot(scope))
    current = dict(_component_digests(scope))
    changed = sorted(
        name for name in snapshot.keys() | current.keys() if snapshot.get(name) != current.get(name)
    )
    if changed:
        raise ArtifactConflictError(
            "Implementation sources changed after this process started; the running code no "
            f"longer matches the files on disk: {', '.join(changed)}"
        )


def require_numerical_policy(threads: NumericalThreadCount) -> None:
    expected = f"{threads}"
    mismatched = [
        variable
        for variable in (
            EnvironmentVariable.OMP_THREADS,
            EnvironmentVariable.MKL_THREADS,
            EnvironmentVariable.OPENBLAS_THREADS,
        )
        if os.environ.get(variable) != expected
    ]
    if torch.get_num_interop_threads() != threads:
        with contextlib.suppress(RuntimeError):
            torch.set_num_interop_threads(threads)
    if (
        mismatched
        or torch.get_num_threads() != threads
        or torch.get_num_interop_threads() != threads
    ):
        raise NumericalPolicyError(
            f"The scientific configuration requires {threads} numerical thread(s) per process; "
            f"set {', '.join(mismatched) or 'the thread environment'} to {expected} before "
            f"starting (PyTorch reports {torch.get_num_threads()} intra-operation and "
            f"{torch.get_num_interop_threads()} inter-operation threads)"
        )
