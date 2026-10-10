import ast
import os
import tomllib
from dataclasses import dataclass
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPOSITORY_ROOT / "src" / "pace"
TYPE_MODULE = SOURCE_ROOT / "types.py"


class ScanError(AssertionError):
    pass


@dataclass(frozen=True)
class ScanReport:
    discovered: tuple[Path, ...]
    parsed: tuple[Path, ...]


def production_files(root: Path = SOURCE_ROOT) -> tuple[Path, ...]:
    if not root.is_dir():
        raise FileNotFoundError(root)
    return tuple(sorted(path for path in root.rglob("*.py") if path.is_file()))


def filesystem_python_files(root: Path) -> tuple[Path, ...]:
    discovered: list[Path] = []
    for directory, subdirectories, filenames in os.walk(root):
        subdirectories[:] = [name for name in subdirectories if name != "__pycache__"]
        discovered.extend(Path(directory) / name for name in filenames if name.endswith(".py"))
    return tuple(sorted(discovered))


def parse_source(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def source_roots_declared_by_packaging(repository_root: Path = REPOSITORY_ROOT) -> tuple[Path, ...]:
    configuration = tomllib.loads((repository_root / "pyproject.toml").read_text(encoding="utf-8"))
    wheel = configuration["tool"]["hatch"]["build"]["targets"]["wheel"]
    return tuple(repository_root / package for package in wheel["packages"])


def undeclared_source_roots(source_parent: Path, declared: tuple[Path, ...]) -> tuple[Path, ...]:
    return tuple(
        sorted(
            directory
            for directory in source_parent.iterdir()
            if directory.is_dir()
            and directory.name != "__pycache__"
            and directory not in declared
            and any(directory.rglob("*.py"))
        )
    )


def scan_production(
    root: Path = SOURCE_ROOT, declared: tuple[Path, ...] | None = None
) -> ScanReport:
    declared_roots = declared if declared is not None else source_roots_declared_by_packaging()
    if root not in declared_roots:
        raise ScanError(f"{root} is not a packaging-declared production root")
    stray = undeclared_source_roots(root.parent, declared_roots)
    if stray:
        raise ScanError(f"Python source roots outside the scanner: {stray}")
    by_glob = production_files(root)
    by_walk = filesystem_python_files(root)
    if by_glob != by_walk:
        raise ScanError(f"Discovery disagrees: {sorted(set(by_glob) ^ set(by_walk))}")
    if not by_glob:
        raise ScanError(f"No production Python files discovered under {root}")
    unpackaged = sorted(
        directory
        for directory in {path.parent for path in by_glob}
        if not (directory / "__init__.py").is_file()
    )
    if unpackaged:
        raise ScanError(f"Directories with Python files but no package marker: {unpackaged}")
    parsed: list[Path] = []
    for path in by_glob:
        try:
            parse_source(path)
        except SyntaxError as error:
            raise ScanError(f"Cannot parse {path}: {error}") from error
        parsed.append(path)
    if tuple(parsed) != by_glob:
        raise ScanError("Not every discovered production file was scanned")
    return ScanReport(by_glob, tuple(parsed))


def annotation_names(annotation: ast.expr) -> set[str]:
    return {
        node.id if isinstance(node, ast.Name) else node.attr
        for node in ast.walk(annotation)
        if isinstance(node, ast.Name | ast.Attribute)
    }


def source_location(path: Path, line: int) -> str:
    return f"{path.relative_to(REPOSITORY_ROOT)}:{line}"
