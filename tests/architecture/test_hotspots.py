import ast
import textwrap
from collections import defaultdict
from functools import cache

from tests.architecture.call_graph import SymbolKind, build_call_graph
from tests.architecture.rules import (
    SourceUnit,
    argument_count,
    cyclomatic_complexity,
    function_length,
    function_nodes,
    import_cycles,
    read_units,
)
from tests.architecture.source_index import (
    REPOSITORY_ROOT,
    SOURCE_ROOT,
    TYPE_MODULE,
    production_files,
)

MAX_FUNCTION_LINES = 65
MAX_COMPLEXITY = 20
MAX_ARGUMENTS = 10
MAX_FAN_OUT = 20
MAX_CLASS_METHODS = 14
MAX_MODULE_LINES = 700
MAX_MODULE_IMPORTS = 14


@cache
def _units() -> tuple[SourceUnit, ...]:
    return read_units(production_files(), REPOSITORY_ROOT, TYPE_MODULE)


def _violations(units: tuple[SourceUnit, ...]) -> list[str]:
    found: list[str] = []
    for unit in units:
        for name, node in function_nodes(unit):
            where = f"{unit.label}:{node.lineno} {name}"
            if function_length(node) > MAX_FUNCTION_LINES:
                found.append(f"{where}: {function_length(node)} lines")
            if cyclomatic_complexity(node) > MAX_COMPLEXITY:
                found.append(f"{where}: complexity {cyclomatic_complexity(node)}")
            if argument_count(node) > MAX_ARGUMENTS:
                found.append(f"{where}: {argument_count(node)} arguments")
    return found


def _module_imports(units: tuple[SourceUnit, ...]) -> dict[str, set[str]]:
    edges: dict[str, set[str]] = defaultdict(set)
    for unit in units:
        module = unit.label.removeprefix("src/").removesuffix(".py").replace("/", ".")
        module = module.removesuffix(".__init__")
        for node in ast.walk(unit.tree):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("pace"):
                edges[module].add(node.module or "")
            elif isinstance(node, ast.Import):
                edges[module].update(a.name for a in node.names if a.name.startswith("pace"))
    return edges


def test_no_function_exceeds_size_complexity_or_argument_limits() -> None:
    assert not _violations(_units())


def test_no_function_has_extreme_fan_out() -> None:
    graph = build_call_graph(SOURCE_ROOT, production_files(), ("pace.cli.main",))
    heavy = [
        f"{name}: {len(targets)}"
        for name, targets in graph.edges.items()
        if graph.symbols[name].kind in {SymbolKind.FUNCTION, SymbolKind.METHOD}
        and len(
            [
                t
                for t in targets
                if graph.symbols[t].kind in {SymbolKind.FUNCTION, SymbolKind.METHOD}
            ]
        )
        > MAX_FAN_OUT
    ]
    assert not heavy, heavy


def test_no_class_or_non_type_module_grows_into_a_god_object() -> None:
    offenders: list[str] = []
    for unit in _units():
        if not unit.is_type_module and len(unit.text.splitlines()) > MAX_MODULE_LINES:
            offenders.append(f"{unit.label}: {len(unit.text.splitlines())} lines")
        for node in ast.walk(unit.tree):
            if isinstance(node, ast.ClassDef) and not unit.is_type_module:
                methods = [n for n in node.body if isinstance(n, ast.FunctionDef)]
                if len(methods) > MAX_CLASS_METHODS:
                    offenders.append(f"{unit.label}: {node.name} has {len(methods)} methods")
    assert not offenders, offenders


def test_modules_do_not_couple_to_too_many_project_modules() -> None:
    edges = _module_imports(_units())
    heavy = {m: len(t) for m, t in edges.items() if len(t) > MAX_MODULE_IMPORTS}
    assert not heavy, heavy


def test_production_import_graph_has_no_dependency_cycles() -> None:
    assert not import_cycles(_module_imports(_units()))


def _unit(source: str) -> SourceUnit:
    return SourceUnit.parse("src/pace/example.py", textwrap.dedent(source))


def test_hotspot_rules_reject_synthetic_extremes() -> None:
    branches = "\n".join(f"    if x == {i}:\n        y += 1" for i in range(25))
    complex_source = f"def f(x):\n    y = 0\n{branches}\n    return y\n"
    assert any("complexity" in v for v in _violations((_unit(complex_source),)))
    long_source = "def g():\n" + "    value = 1\n" * 80 + "    return value\n"
    assert any("lines" in v for v in _violations((_unit(long_source),)))
    wide = "def h(" + ", ".join(f"a{i}" for i in range(12)) + "):\n    return 1\n"
    assert any("arguments" in v for v in _violations((_unit(wide),)))


def test_hotspot_rules_accept_a_small_function() -> None:
    assert not _violations((_unit("def f(a, b):\n    return a + b\n"),))


def test_import_cycles_are_detected_and_acyclic_graphs_pass() -> None:
    assert import_cycles({"a": {"b"}, "b": {"c"}, "c": {"a"}}) == [("a", "b", "c")]
    assert not import_cycles({"a": {"b"}, "b": {"c"}, "c": set()})
