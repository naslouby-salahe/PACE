import json
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from tests.architecture.rules import SourceUnit, rule_suppressions
from tests.architecture.source_index import (
    REPOSITORY_ROOT,
    SOURCE_ROOT,
    ScanError,
    filesystem_python_files,
    production_files,
    scan_production,
)


def _package(root: Path, files: dict[str, str]) -> Path:
    package = root / "src" / "pace"
    for relative, text in files.items():
        target = package / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    return package


def test_scanner_discovery_agrees_with_independent_filesystem_walk() -> None:
    assert production_files() == filesystem_python_files(SOURCE_ROOT)
    assert set(scan_production().parsed) == set(production_files())


def test_scanner_rejects_unparsable_production_file(tmp_path: Path) -> None:
    package = _package(tmp_path, {"__init__.py": "", "broken.py": "def (:\n"})
    with pytest.raises(ScanError, match="Cannot parse"):
        scan_production(package, (package,))


def test_scanner_rejects_a_root_that_packaging_does_not_declare(tmp_path: Path) -> None:
    package = _package(tmp_path, {"__init__.py": ""})
    with pytest.raises(ScanError, match="not a packaging-declared"):
        scan_production(package, ())


def test_scanner_rejects_a_new_unscanned_source_root(tmp_path: Path) -> None:
    package = _package(tmp_path, {"__init__.py": ""})
    other = tmp_path / "src" / "other"
    other.mkdir()
    (other / "__init__.py").write_text("", encoding="utf-8")
    with pytest.raises(ScanError, match="outside the scanner"):
        scan_production(package, (package,))


def test_scanner_rejects_directories_without_package_markers(tmp_path: Path) -> None:
    package = _package(tmp_path, {"__init__.py": "", "sub/module.py": "x = 1\n"})
    with pytest.raises(ScanError, match="no package marker"):
        scan_production(package, (package,))


def test_scanner_rejects_an_empty_production_root(tmp_path: Path) -> None:
    package = tmp_path / "src" / "pace"
    package.mkdir(parents=True)
    with pytest.raises(ScanError, match="No production Python files"):
        scan_production(package, (package,))


def test_scanner_accepts_a_wellformed_package(tmp_path: Path) -> None:
    package = _package(tmp_path, {"__init__.py": "", "a.py": "x = 1\n"})
    assert len(scan_production(package, (package,)).parsed) == 2


def test_suppression_escape_attempts_are_detected() -> None:
    for line in ("x = 1  # type: ignore", "import os  # noqa: F401", "x = 1  # pyright: ignore"):
        assert rule_suppressions(SourceUnit.parse("src/pace/x.py", line + "\n"))


def _tool_text() -> str:
    return (REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8")


def test_pyright_is_strict_over_all_source_and_tests_without_weakening() -> None:
    configuration = json.loads((REPOSITORY_ROOT / "pyrightconfig.json").read_text())
    assert configuration["typeCheckingMode"] == "strict"
    assert {"src", "tests"} <= set(configuration["include"])
    assert not set(configuration) & {"exclude", "ignore", "executionEnvironments"}
    assert not any(key.startswith("report") for key in configuration)
    assert "[tool.pyright]" not in _tool_text()


def test_ruff_covers_the_whole_repository_without_per_file_ignores() -> None:
    configuration = tomllib.loads(_tool_text())
    ruff = configuration.get("tool", {}).get("ruff", {})
    assert isinstance(ruff, dict)
    text = _tool_text()
    assert "per-file-ignores" not in text
    assert "extend-exclude" not in text
    assert "ignore = [" not in text
    assert '"ANN"' in text
    assert '"ARG"' in text


def test_ci_and_make_run_every_quality_tool_including_architecture_tests() -> None:
    workflow = (REPOSITORY_ROOT / ".github" / "workflows" / "quality.yml").read_text()
    makefile = (REPOSITORY_ROOT / "Makefile").read_text()
    for text in (workflow, makefile):
        for tool in ("ruff check", "pyright", "semgrep", "lint-imports", "vulture", "pytest"):
            assert tool in text, tool
    assert not (REPOSITORY_ROOT / "rules" / "vulture_whitelist.py").exists()


def test_full_suite_keeps_the_coverage_floor_and_architecture_only_runs_do_not() -> None:
    from tests.conftest import architecture_only

    architecture = (REPOSITORY_ROOT / "tests/architecture/test_domain_rules.py").resolve()
    behavioral = (REPOSITORY_ROOT / "tests/test_cli.py").resolve()
    assert "--cov-fail-under=90" in _tool_text()
    assert architecture_only([architecture])
    assert architecture_only([architecture.parent / "test_reachability.py"])
    assert not architecture_only([architecture, behavioral])
    assert not architecture_only([])


def test_pytest_collects_the_architecture_suite_in_the_default_command() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "--no-cov"],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert "tests/architecture/test_domain_rules.py" in result.stdout
    assert "tests/architecture/test_reachability.py" in result.stdout
    assert "tests/architecture/test_hotspots.py" in result.stdout
    assert "tests/architecture/test_logging_rules.py" in result.stdout


def test_semgrep_rules_scan_every_production_path_for_print() -> None:
    text = (REPOSITORY_ROOT / "rules" / "semgrep" / "architecture.yml").read_text()
    assert "print(...)" in text
    assert "/src/**" in text
