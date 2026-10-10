import ast
import re
from dataclasses import dataclass

from tests.architecture.call_graph import CallGraph, SymbolKind
from tests.architecture.rules import (
    LOG_METHODS,
    SourceUnit,
    called_names,
    contains_logging,
    function_nodes,
    logged_events,
    performs_filesystem_io,
)

LIFECYCLE_EVENTS = frozenset({"COMMAND_STARTED", "COMMAND_COMPLETED", "COMMAND_FAILED"})
ORCHESTRATION_FAN_OUT = 8


@dataclass(frozen=True)
class FunctionFact:
    qualname: str
    module: str
    name: str
    node: ast.FunctionDef | ast.AsyncFunctionDef
    logs: bool
    io: bool
    fan_out: int
    calls: frozenset[str]

    @property
    def is_public(self) -> bool:
        return not any(part.startswith("_") for part in self.name.split("."))


def module_of(label: str) -> str:
    stem = label.removeprefix("src/").removesuffix(".py").replace("/", ".")
    return stem.removesuffix(".__init__")


def function_facts(units: tuple[SourceUnit, ...], graph: CallGraph) -> dict[str, FunctionFact]:
    facts: dict[str, FunctionFact] = {}
    for unit in units:
        module = module_of(unit.label)
        for name, node in function_nodes(unit):
            qualname = f"{module}.{name}"
            callees = [
                target
                for target in graph.edges.get(qualname, ())
                if target in graph.symbols
                and graph.symbols[target].kind in {SymbolKind.FUNCTION, SymbolKind.METHOD}
            ]
            facts[qualname] = FunctionFact(
                qualname,
                module,
                name,
                node,
                contains_logging(node),
                performs_filesystem_io(node),
                len(callees),
                frozenset(called_names(node)),
            )
    return facts


def _observable(facts: dict[str, FunctionFact]) -> set[str]:
    by_name: dict[str, list[str]] = {}
    for qualname, fact in facts.items():
        by_name.setdefault(fact.name.rpartition(".")[2], []).append(qualname)
    logging_functions = {qualname for qualname, fact in facts.items() if fact.logs}
    observable = set(logging_functions)
    for qualname, fact in facts.items():
        if any(
            callee in logging_functions for call in fact.calls for callee in by_name.get(call, ())
        ):
            observable.add(qualname)
    return observable


def unobserved_orchestrators(facts: dict[str, FunctionFact], layers: tuple[str, ...]) -> list[str]:
    observable = _observable(facts)
    return [
        f"{fact.qualname}: public {'I/O ' if fact.io else ''}orchestration "
        f"(fan-out {fact.fan_out}) neither logs nor delegates to a logging function"
        for fact in sorted(facts.values(), key=lambda item: item.qualname)
        if fact.module.startswith(layers)
        and fact.is_public
        and (fact.io or fact.fan_out >= ORCHESTRATION_FAN_OUT)
        and fact.qualname not in observable
    ]


def commands_without_lifecycle_logging(
    facts: dict[str, FunctionFact], commands: dict[str, str]
) -> list[str]:
    lifecycle_names = {
        fact.name
        for fact in facts.values()
        if logged_events(fact.node) >= LIFECYCLE_EVENTS and contains_logging(fact.node)
    }
    problems: list[str] = []
    if not lifecycle_names:
        problems.append("no function logs the complete command lifecycle")
    for command, qualname in sorted(commands.items()):
        fact = facts.get(qualname)
        if fact is None:
            problems.append(f"{command}: handler {qualname} has no function body")
        elif (
            not (fact.calls & lifecycle_names) and not logged_events(fact.node) >= LIFECYCLE_EVENTS
        ):
            problems.append(f"{command}: handler {qualname} does not run the lifecycle logger")
    return problems


PER_SAMPLE_ITERABLE = re.compile(r"(row|sample|record|packet|flow|feature|batch|chunk|epoch)s?\b")


def _is_log_call(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in LOG_METHODS
    )


def _hot_logging(node: ast.AST) -> list[ast.AST]:
    found: list[ast.AST] = []
    for child in ast.walk(node):
        hot_body: list[ast.AST] = []
        if (
            isinstance(
                child, ast.ListComp | ast.SetComp | ast.DictComp | ast.GeneratorExp | ast.While
            )
            or isinstance(child, ast.For | ast.AsyncFor)
            and PER_SAMPLE_ITERABLE.search(ast.unparse(child.iter))
        ):
            hot_body = [child]
        for body in hot_body:
            found.extend(item for item in ast.walk(body) if _is_log_call(item))
    return found


def logging_inside_hot_loops(facts: dict[str, FunctionFact]) -> list[str]:
    return [
        f"{fact.qualname}: log call inside a comprehension, polling loop or per-sample loop"
        for fact in sorted(facts.values(), key=lambda item: item.qualname)
        if _hot_logging(fact.node)
    ]
