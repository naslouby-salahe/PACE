import textwrap
from functools import cache

import pytest

from tests.architecture.call_graph import (
    CallGraph,
    Reachability,
    SymbolKind,
    build_call_graph,
    build_call_graph_from_sources,
    callable_records,
    command_metrics,
    dead_enum_members,
    orphans,
    reachability,
)
from tests.architecture.source_index import SOURCE_ROOT, production_files

PRODUCTION_ENTRY_POINTS = ("pace.cli.main",)
COMMANDS = {"doctor", "status", "preflight", "run", "report", "accept"}


@cache
def _production() -> tuple[CallGraph, Reachability]:
    graph = build_call_graph(SOURCE_ROOT, production_files(), PRODUCTION_ENTRY_POINTS)
    return graph, reachability(graph)


def _synthetic(*modules: tuple[str, str]) -> tuple[CallGraph, Reachability]:
    sources = {name: (f"{name}.py", textwrap.dedent(text)) for name, text in modules}
    graph = build_call_graph_from_sources(sources, ("app.cli.main",))
    return graph, reachability(graph)


CLI_MODULE = (
    "app.cli",
    """
    import typer
    from app.service import serve

    app = typer.Typer()

    def hello():
        serve()

    def main():
        app()

    for _command in (hello,):
        app.command()(_command)
    """,
)
SERVICE_MODULE = (
    "app.service",
    """
    def serve():
        return helper()

    def helper():
        return 1
    """,
)


def test_every_cli_command_resolves_to_a_production_handler() -> None:
    graph, _ = _production()
    assert set(graph.roots) >= COMMANDS
    for command in COMMANDS:
        symbol = graph.symbols[graph.roots[command]]
        assert symbol.kind is SymbolKind.FUNCTION, command


def test_every_production_callable_is_reachable_from_a_cli_root() -> None:
    graph, reached = _production()
    assert not orphans(graph, reached)


def test_no_enum_member_is_dead() -> None:
    graph, reached = _production()
    assert not dead_enum_members(graph, reached)


def test_every_handler_reaches_real_production_functionality() -> None:
    graph, reached = _production()
    for metrics in command_metrics(graph, reached):
        if metrics.command in COMMANDS:
            assert metrics.reachable_callables > 50, metrics
            assert metrics.modules_crossed >= 10, metrics
            assert metrics.leaf_callables > 0, metrics
            assert metrics.direct_callees >= 3, metrics


def test_no_command_bypasses_the_workflow_layers() -> None:
    graph, reached = _production()
    workflow_modules = {"pace.experiment", "pace.reporting", "pace.config"}
    for command in COMMANDS:
        modules = {
            graph.symbols[name].module
            for name in reached.distance_by_root[command]
            if graph.symbols[name].kind in {SymbolKind.FUNCTION, SymbolKind.METHOD}
        }
        assert any(module.startswith(tuple(workflow_modules)) for module in modules), command


def test_call_graph_has_no_disconnected_component_of_callables() -> None:
    graph, reached = _production()
    live = reached.reached()
    disconnected = [
        record.qualname
        for record in callable_records(graph, reached)
        if record.qualname not in live
    ]
    assert not disconnected, disconnected


def test_reachability_diagnostics_name_callers_roots_distance_and_descendants() -> None:
    graph, reached = _production()
    records = {record.qualname: record for record in callable_records(graph, reached)}
    record = records["pace.experiment.runner.run_registered_dataset_matrix_workflow"]
    assert record.direct_callers
    assert "run" in record.cli_roots
    assert record.nearest_distance is not None
    assert record.nearest_distance >= 1
    assert record.descendants > 50
    path = reached.shortest_path("run", record.qualname)
    assert path[0] == "pace.cli.run"
    assert path[-1] == record.qualname


def test_command_metrics_report_depth_leaves_and_modules() -> None:
    graph, reached = _production()
    by_command = {item.command: item for item in command_metrics(graph, reached)}
    assert by_command["run"].reachable_callables > by_command["status"].reachable_callables
    assert by_command["run"].maximum_depth >= by_command["status"].maximum_depth > 0


def test_synthetic_live_graph_has_no_orphans() -> None:
    graph, reached = _synthetic(CLI_MODULE, SERVICE_MODULE)
    assert not orphans(graph, reached)


@pytest.mark.parametrize(
    ("dead_source", "dead_name"),
    [
        ("def unused():\n    return 0\n", "app.dead.unused"),
        (
            "def a():\n    return b()\ndef b():\n    return 0\n",
            "app.dead.a",
        ),
        ("class Dead:\n    pass\n", "app.dead.Dead"),
        (
            "class Dead:\n    def method(self):\n        return 0\n",
            "app.dead.Dead.method",
        ),
        ("UNUSED_NAME = 3\n", "app.dead.UNUSED_NAME"),
        (
            "def a():\n    return b()\ndef b():\n    return c()\ndef c():\n    return 0\n",
            "app.dead.c",
        ),
    ],
)
def test_dead_code_shapes_are_detected(dead_source: str, dead_name: str) -> None:
    graph, reached = _synthetic(CLI_MODULE, SERVICE_MODULE, ("app.dead", dead_source))
    found = orphans(graph, reached)
    assert any(dead_name in line for line in found), found
    assert all("unreachable from CLI roots" in line for line in found)
    assert all("direct callers" in line for line in found)


def test_method_called_only_by_dead_method_is_dead() -> None:
    graph, reached = _synthetic(
        CLI_MODULE,
        SERVICE_MODULE,
        (
            "app.dead",
            """
            class Dead:
                def outer(self):
                    return self.inner()
                def inner(self):
                    return 1
            """,
        ),
    )
    found = " ".join(orphans(graph, reached))
    assert "app.dead.Dead.outer" in found
    assert "app.dead.Dead.inner" in found


def test_dead_module_and_disconnected_helper_chain_are_detected() -> None:
    graph, reached = _synthetic(
        CLI_MODULE,
        SERVICE_MODULE,
        ("app.orphan_module", "def entry():\n    return leaf()\ndef leaf():\n    return 0\n"),
    )
    found = " ".join(orphans(graph, reached))
    assert "app.orphan_module.entry" in found
    assert "app.orphan_module.leaf" in found


def test_unused_enum_members_and_unused_enum_are_detected() -> None:
    graph, reached = _synthetic(
        CLI_MODULE,
        (
            "app.service",
            """
            from enum import Enum
            from app.kinds import Kind

            def serve():
                return Kind.USED
            """,
        ),
        (
            "app.kinds",
            """
            from enum import Enum

            class Kind(Enum):
                USED = 1
                UNUSED = 2
            """,
        ),
    )
    assert any("Kind.UNUSED" in line for line in dead_enum_members(graph, reached))
    assert not any("Kind.USED" in line for line in dead_enum_members(graph, reached))


def test_dynamically_enumerated_enum_keeps_every_member_alive() -> None:
    graph, reached = _synthetic(
        CLI_MODULE,
        (
            "app.service",
            """
            from app.kinds import Kind

            def serve():
                return tuple(Kind)
            """,
        ),
        (
            "app.kinds",
            """
            from enum import Enum

            class Kind(Enum):
                A = 1
                B = 2
            """,
        ),
    )
    assert not dead_enum_members(graph, reached)


def test_registration_decorators_and_framework_overrides_are_live_by_semantic_rule() -> None:
    graph, reached = _synthetic(
        CLI_MODULE,
        (
            "app.service",
            """
            import io
            from pydantic import BaseModel, field_validator

            class Settings(BaseModel):
                value: int

                @field_validator("value")
                @classmethod
                def positive(cls, value):
                    return value

            class Reader(io.RawIOBase):
                def readable(self):
                    return True

            def serve():
                return Settings(value=1), Reader()
            """,
        ),
    )
    assert not orphans(graph, reached)


def test_framework_override_on_unreached_class_stays_dead() -> None:
    graph, reached = _synthetic(
        CLI_MODULE,
        SERVICE_MODULE,
        (
            "app.dead",
            """
            import io

            class Reader(io.RawIOBase):
                def readable(self):
                    return True
            """,
        ),
    )
    assert any("app.dead.Reader" in line for line in orphans(graph, reached))


def test_type_hint_only_enum_reference_does_not_keep_members_alive() -> None:
    graph, reached = _synthetic(
        CLI_MODULE,
        (
            "app.service",
            """
            from app.kinds import Kind

            def serve(kind: Kind) -> Kind:
                return kind
            """,
        ),
        (
            "app.kinds",
            """
            from enum import Enum

            class Kind(Enum):
                A = 1
            """,
        ),
    )
    assert any("Kind.A" in line for line in dead_enum_members(graph, reached))
