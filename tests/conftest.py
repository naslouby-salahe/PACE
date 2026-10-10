from pathlib import Path

import pytest

ARCHITECTURE_ROOT = Path(__file__).resolve().parent / "architecture"


def architecture_only(paths: list[Path]) -> bool:
    if not paths:
        return False
    return all(path == ARCHITECTURE_ROOT or ARCHITECTURE_ROOT in path.parents for path in paths)


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    collected = [Path(str(item.path)).resolve() for item in items]
    if not architecture_only(collected):
        return
    config.option.cov_fail_under = 0
    coverage = config.pluginmanager.get_plugin("_cov")
    if coverage is not None:
        coverage.options.cov_fail_under = 0
