import ast
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

ENUM_BASES = frozenset({"Enum", "IntEnum", "StrEnum", "Flag", "IntFlag"})
DUNDER_ENTRY_POINTS = frozenset(
    {
        "__init__",
        "__post_init__",
        "__new__",
        "__call__",
        "__enter__",
        "__exit__",
        "__iter__",
        "__next__",
        "__len__",
        "__getitem__",
        "__contains__",
        "__eq__",
        "__hash__",
        "__repr__",
        "__str__",
        "__get_pydantic_core_schema__",
        "__set_name__",
    }
)
REGISTRATION_DECORATORS = frozenset({"field_validator", "model_validator", "computed_field"})


class SymbolKind(Enum):
    MODULE = "module"
    FUNCTION = "function"
    METHOD = "method"
    CLASS = "class"
    CONSTANT = "constant"


@dataclass(frozen=True)
class Symbol:
    qualname: str
    kind: SymbolKind
    path: str
    line: int
    module: str
    owner: str | None


@dataclass
class CallGraph:
    symbols: dict[str, Symbol] = field(default_factory=dict[str, Symbol])
    edges: dict[str, set[str]] = field(default_factory=dict[str, set[str]])
    roots: dict[str, str] = field(default_factory=dict[str, str])
    method_index: dict[str, set[str]] = field(default_factory=dict[str, set[str]])
    attributes: dict[str, set[str]] = field(default_factory=dict[str, set[str]])
    enum_members: dict[str, tuple[str, ...]] = field(default_factory=dict[str, tuple[str, ...]])
    enum_member_refs: dict[str, set[tuple[str, str]]] = field(
        default_factory=dict[str, set[tuple[str, str]]]
    )
    dynamic_enum_refs: dict[str, set[str]] = field(default_factory=dict[str, set[str]])

    def callers(self, target: str) -> tuple[str, ...]:
        return tuple(sorted(source for source, targets in self.edges.items() if target in targets))


@dataclass(frozen=True)
class Reachability:
    distance_by_root: dict[str, dict[str, int]]
    predecessor_by_root: dict[str, dict[str, str]]

    def roots_reaching(self, qualname: str) -> tuple[str, ...]:
        return tuple(
            sorted(root for root, seen in self.distance_by_root.items() if qualname in seen)
        )

    def nearest_root_distance(self, qualname: str) -> int | None:
        distances = [seen[qualname] for seen in self.distance_by_root.values() if qualname in seen]
        return min(distances) if distances else None

    def shortest_path(self, root: str, qualname: str) -> tuple[str, ...]:
        predecessors = self.predecessor_by_root[root]
        path = [qualname]
        while path[-1] in predecessors:
            path.append(predecessors[path[-1]])
        return tuple(reversed(path))

    def reached(self) -> frozenset[str]:
        return frozenset(name for seen in self.distance_by_root.values() for name in seen)


@dataclass(frozen=True)
class CallableRecord:
    qualname: str
    path: str
    line: int
    kind: SymbolKind
    direct_callers: tuple[str, ...]
    cli_roots: tuple[str, ...]
    nearest_distance: int | None
    descendants: int


@dataclass(frozen=True)
class CommandMetrics:
    command: str
    direct_callees: int
    reachable_callables: int
    maximum_depth: int
    leaf_callables: int
    modules_crossed: int


def module_name(package_root: Path, path: Path) -> str:
    parts = list(path.relative_to(package_root.parent).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _decorator_names(node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) -> set[str]:
    names: set[str] = set()
    for decorator in node.decorator_list:
        target = decorator.func if isinstance(decorator, ast.Call) else decorator
        if isinstance(target, ast.Name):
            names.add(target.id)
        elif isinstance(target, ast.Attribute):
            names.add(target.attr)
    return names


def _base_name(base: ast.expr) -> str:
    if isinstance(base, ast.Name):
        return base.id
    if isinstance(base, ast.Attribute):
        return base.attr
    if isinstance(base, ast.Subscript):
        return _base_name(base.value)
    return ast.unparse(base)


class _Indexer:
    def __init__(
        self, sources: Mapping[str, tuple[str, str]], entry_points: tuple[str, ...]
    ) -> None:
        self.graph = CallGraph()
        self.entry_points = entry_points
        self.imports: dict[str, dict[str, str]] = {}
        self.bodies: dict[str, list[ast.AST]] = {}
        self.class_bases: dict[str, list[str]] = {}
        self.trees: dict[str, tuple[str, ast.Module]] = {}
        for module, (path, text) in sources.items():
            self.trees[module] = (path, ast.parse(text, filename=path))

    def build(self) -> CallGraph:
        for module, (path, tree) in self.trees.items():
            self._declare_module(module, path, tree)
        for module in self.trees:
            self._link_module(module)
        for entry in self.entry_points:
            if entry in self.graph.symbols:
                self.graph.roots[entry.rpartition(".")[2]] = entry
        return self.graph

    def _add(self, symbol: Symbol, refs: list[ast.AST]) -> None:
        self.graph.symbols[symbol.qualname] = symbol
        self.graph.edges.setdefault(symbol.qualname, set())
        self.bodies[symbol.qualname] = refs

    def _declare_module(self, module: str, path: str, tree: ast.Module) -> None:
        self.imports[module] = {}
        module_refs: list[ast.AST] = []
        for statement in tree.body:
            match statement:
                case ast.Import():
                    for alias in statement.names:
                        bound = alias.asname or alias.name.split(".")[0]
                        self.imports[module][bound] = alias.name if alias.asname else bound
                case ast.ImportFrom():
                    for alias in statement.names:
                        base = statement.module or ""
                        self.imports[module][alias.asname or alias.name] = f"{base}.{alias.name}"
                case ast.FunctionDef() | ast.AsyncFunctionDef():
                    qualname = f"{module}.{statement.name}"
                    self._add(
                        Symbol(qualname, SymbolKind.FUNCTION, path, statement.lineno, module, None),
                        [statement],
                    )
                case ast.ClassDef():
                    self._declare_class(module, path, statement, module, None)
                case ast.Assign() | ast.AnnAssign() if self._constant_name(statement):
                    self._declare_constant(module, path, statement)
                case ast.For() if self._registered_commands(statement) is not None:
                    for command in self._registered_commands(statement) or ():
                        self.graph.roots[command] = f"{module}.{command}"
                    module_refs.extend(self._registration_support(statement))
                case _:
                    module_refs.append(statement)
        self._add(Symbol(module, SymbolKind.MODULE, path, 1, module, None), module_refs)

    @staticmethod
    def _registration_support(statement: ast.For) -> list[ast.AST]:
        return [
            node
            for node in ast.walk(statement)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Call)
            and isinstance(node.func.func, ast.Attribute)
            and isinstance(node.func.func.value, ast.Name)
        ]

    def _declare_constant(self, module: str, path: str, statement: ast.stmt) -> None:
        name = self._constant_name(statement)
        assert name is not None
        refs: list[ast.AST] = []
        if isinstance(statement, ast.Assign | ast.AnnAssign) and statement.value is not None:
            refs.append(statement.value)
        if isinstance(statement, ast.AnnAssign):
            refs.append(statement.annotation)
        self._add(
            Symbol(f"{module}.{name}", SymbolKind.CONSTANT, path, statement.lineno, module, None),
            refs,
        )

    @staticmethod
    def _constant_name(statement: ast.stmt) -> str | None:
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
            target = statement.targets[0]
            return target.id if isinstance(target, ast.Name) else None
        if isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name):
            return statement.target.id
        return None

    @staticmethod
    def _registered_commands(statement: ast.For) -> tuple[str, ...] | None:
        if not isinstance(statement.iter, ast.Tuple) or not isinstance(statement.target, ast.Name):
            return None
        registers = any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Call)
            and isinstance(node.func.func, ast.Attribute)
            and node.func.func.attr == "command"
            for node in ast.walk(statement)
        )
        names = tuple(item.id for item in statement.iter.elts if isinstance(item, ast.Name))
        return names if registers and len(names) == len(statement.iter.elts) else None

    def _declare_class(
        self, module: str, path: str, node: ast.ClassDef, prefix: str, owner: str | None
    ) -> None:
        qualname = f"{prefix}.{node.name}"
        header: list[ast.AST] = [*node.decorator_list, *node.bases, *node.keywords]
        class_body: list[ast.AST] = []
        members: list[str] = []
        for statement in node.body:
            if isinstance(statement, ast.FunctionDef | ast.AsyncFunctionDef):
                method = f"{qualname}.{statement.name}"
                self._add(
                    Symbol(method, SymbolKind.METHOD, path, statement.lineno, module, qualname),
                    [statement],
                )
                self.graph.method_index.setdefault(statement.name, set()).add(method)
            elif isinstance(statement, ast.ClassDef):
                self._declare_class(module, path, statement, qualname, qualname)
            else:
                class_body.append(statement)
                if isinstance(statement, ast.Assign):
                    members.extend(
                        target.id
                        for target in statement.targets
                        if isinstance(target, ast.Name) and not target.id.startswith("_")
                    )
        bases = [_base_name(base) for base in node.bases]
        self.class_bases[qualname] = bases
        if any(base in ENUM_BASES for base in bases):
            self.graph.enum_members[qualname] = tuple(members)
        self._add(
            Symbol(qualname, SymbolKind.CLASS, path, node.lineno, module, owner),
            [*header, *class_body],
        )

    def _resolve_import(self, target: str, seen: frozenset[str] = frozenset()) -> str | None:
        if target in self.graph.symbols:
            return target
        module, _, name = target.rpartition(".")
        if module in self.trees and target not in seen:
            forwarded = self.imports.get(module, {}).get(name)
            if forwarded is not None:
                return self._resolve_import(forwarded, seen | {target})
        return None

    def _targets_for_name(self, module: str, name: str) -> set[str]:
        local = f"{module}.{name}"
        if local in self.graph.symbols:
            return {local}
        imported = self.imports[module].get(name)
        if imported is None:
            return set()
        resolved = self._resolve_import(imported)
        if resolved is not None:
            return {resolved}
        return {imported} if imported in self.trees else set()

    def _link_module(self, module: str) -> None:
        for qualname, symbol in tuple(self.graph.symbols.items()):
            if symbol.module != module:
                continue
            targets = self.graph.edges[qualname]
            member_refs: set[tuple[str, str]] = set()
            dynamic_refs: set[str] = set()
            for body in self.bodies[qualname]:
                attribute_heads = {
                    id(node.value)
                    for node in ast.walk(body)
                    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
                }
                hint_only = self._hint_only_names(qualname, symbol, body)
                for node in ast.walk(body):
                    if isinstance(node, ast.Name):
                        resolved = self._targets_for_name(module, node.id)
                        targets |= resolved
                        for target in resolved:
                            if (
                                target in self.graph.enum_members
                                and id(node) not in attribute_heads
                                and id(node) not in hint_only
                            ):
                                dynamic_refs.add(target)
                    elif isinstance(node, ast.Attribute):
                        targets |= self._attribute_targets(module, node)
                        self.graph.attributes.setdefault(qualname, set()).add(node.attr)
                        self._record_member_access(module, node, member_refs, dynamic_refs)
            self.graph.enum_member_refs[qualname] = member_refs
            self.graph.dynamic_enum_refs[qualname] = dynamic_refs
            if symbol.kind is not SymbolKind.MODULE:
                targets.add(symbol.module)
            if symbol.kind is SymbolKind.CLASS:
                targets |= self._class_entry_points(qualname)
            if symbol.kind is SymbolKind.MODULE:
                parent = qualname.rpartition(".")[0]
                if parent in self.graph.symbols:
                    targets.add(parent)
            targets.discard(qualname)

    def _is_pydantic(self, qualname: str, seen: frozenset[str] = frozenset()) -> bool:
        for base in self.class_bases.get(qualname, ()):
            if base == "BaseModel":
                return True
            for candidate, symbol in self.graph.symbols.items():
                if (
                    symbol.kind is SymbolKind.CLASS
                    and candidate.endswith(f".{base}")
                    and candidate not in seen
                    and self._is_pydantic(candidate, seen | {qualname})
                ):
                    return True
        return False

    def _hint_only_names(self, qualname: str, symbol: Symbol, body: ast.AST) -> set[int]:
        hints: set[int] = set()
        pydantic_fields = symbol.kind is SymbolKind.CLASS and self._is_pydantic(qualname)
        for node in ast.walk(body):
            annotations: list[ast.expr] = []
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                annotations.extend(
                    argument.annotation
                    for argument in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs)
                    if argument.annotation is not None
                )
                if node.returns is not None:
                    annotations.append(node.returns)
            elif isinstance(node, ast.AnnAssign) and not pydantic_fields:
                annotations.append(node.annotation)
            for annotation in annotations:
                hints.update(id(part) for part in ast.walk(annotation))
        return hints

    def _record_member_access(
        self,
        module: str,
        node: ast.Attribute,
        member_refs: set[tuple[str, str]],
        dynamic_refs: set[str],
    ) -> None:
        if not isinstance(node.value, ast.Name):
            return
        for target in self._targets_for_name(module, node.value.id):
            members = self.graph.enum_members.get(target)
            if members is None:
                continue
            if node.attr in members:
                member_refs.add((target, node.attr))
            else:
                dynamic_refs.add(target)

    def _attribute_targets(self, module: str, node: ast.Attribute) -> set[str]:
        targets: set[str] = set()
        if isinstance(node.value, ast.Name):
            imported = self.imports[module].get(node.value.id)
            if imported is not None:
                resolved = self._resolve_import(f"{imported}.{node.attr}")
                if resolved is not None:
                    targets.add(resolved)
        return targets

    def _class_entry_points(self, qualname: str) -> set[str]:
        entry: set[str] = set()
        bases = self.class_bases[qualname]
        is_protocol = "Protocol" in bases
        framework_bases = {
            base
            for base in bases
            if base not in ENUM_BASES | {"BaseModel", "Protocol"}
            and not self._is_project_class(base)
        }
        for symbol in self.graph.symbols.values():
            if symbol.owner != qualname or symbol.kind is not SymbolKind.METHOD:
                continue
            method_name = symbol.qualname.rpartition(".")[2]
            node = self.bodies[symbol.qualname][0]
            assert isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
            if (
                method_name in DUNDER_ENTRY_POINTS
                or _decorator_names(node) & REGISTRATION_DECORATORS
                or is_protocol
                or (framework_bases and not method_name.startswith("_"))
            ):
                entry.add(symbol.qualname)
        return entry

    def _is_project_class(self, base: str) -> bool:
        return any(
            qualname.endswith(f".{base}") and symbol.kind is SymbolKind.CLASS
            for qualname, symbol in self.graph.symbols.items()
        )


def build_call_graph_from_sources(
    sources: Mapping[str, tuple[str, str]], entry_points: tuple[str, ...] = ()
) -> CallGraph:
    return _Indexer(sources, entry_points).build()


def build_call_graph(
    package_root: Path, files: tuple[Path, ...], entry_points: tuple[str, ...] = ()
) -> CallGraph:
    sources = {
        module_name(package_root, path): (str(path), path.read_text(encoding="utf-8"))
        for path in files
    }
    return build_call_graph_from_sources(sources, entry_points)


def _reach_from(graph: CallGraph, root: str) -> tuple[dict[str, int], dict[str, str]]:
    distances = {root: 0}
    predecessors: dict[str, str] = {}
    queue = deque([root])
    seen_attributes: set[str] = set()

    def add(target: str, source: str) -> None:
        if target in distances:
            return
        distances[target] = distances[source] + 1
        predecessors[target] = source
        queue.append(target)
        symbol = graph.symbols.get(target)
        if symbol is not None and symbol.kind is SymbolKind.CLASS:
            for attribute in sorted(seen_attributes):
                for method in sorted(graph.method_index.get(attribute, ())):
                    if graph.symbols[method].owner == target:
                        add(method, target)

    while queue:
        current = queue.popleft()
        for target in sorted(graph.edges.get(current, ())):
            add(target, current)
        for attribute in sorted(graph.attributes.get(current, ())):
            if attribute in seen_attributes:
                continue
            seen_attributes.add(attribute)
            for method in sorted(graph.method_index.get(attribute, ())):
                if graph.symbols[method].owner in distances:
                    add(method, current)
    return distances, predecessors


def reachability(graph: CallGraph) -> Reachability:
    distance_by_root: dict[str, dict[str, int]] = {}
    predecessor_by_root: dict[str, dict[str, str]] = {}
    for name, root in graph.roots.items():
        distance_by_root[name], predecessor_by_root[name] = _reach_from(graph, root)
    return Reachability(distance_by_root, predecessor_by_root)


def callable_records(graph: CallGraph, reached: Reachability) -> list[CallableRecord]:
    descendants: dict[str, int] = {}
    for qualname in graph.symbols:
        seen = {qualname}
        queue = deque([qualname])
        while queue:
            for target in graph.edges.get(queue.popleft(), ()):
                if target not in seen and graph.symbols[target].kind is not SymbolKind.MODULE:
                    seen.add(target)
                    queue.append(target)
        descendants[qualname] = len(seen) - 1
    return [
        CallableRecord(
            qualname,
            symbol.path,
            symbol.line,
            symbol.kind,
            graph.callers(qualname),
            reached.roots_reaching(qualname),
            reached.nearest_root_distance(qualname),
            descendants[qualname],
        )
        for qualname, symbol in sorted(graph.symbols.items())
        if symbol.kind is not SymbolKind.MODULE
    ]


def orphans(graph: CallGraph, reached: Reachability) -> list[str]:
    found: list[str] = []
    for record in callable_records(graph, reached):
        if not record.cli_roots:
            found.append(
                f"{record.path}:{record.line} {record.kind.value} {record.qualname} is "
                f"unreachable from CLI roots; direct callers: "
                f"{', '.join(record.direct_callers) or 'none'}; "
                f"nearest root distance: {record.nearest_distance}"
            )
    return found


def dead_enum_members(graph: CallGraph, reached: Reachability) -> list[str]:
    live = reached.reached()
    used: set[tuple[str, str]] = set()
    dynamic: set[str] = set()
    for qualname in live:
        used |= graph.enum_member_refs.get(qualname, set())
        dynamic |= graph.dynamic_enum_refs.get(qualname, set())
    found: list[str] = []
    for enum, members in sorted(graph.enum_members.items()):
        if enum not in live or enum in dynamic:
            continue
        for member in members:
            if (enum, member) not in used:
                symbol = graph.symbols[enum]
                found.append(f"{symbol.path}:{symbol.line} enum member {enum}.{member} is unused")
    return found


def command_metrics(graph: CallGraph, reached: Reachability) -> list[CommandMetrics]:
    metrics: list[CommandMetrics] = []
    for command, root in sorted(graph.roots.items()):
        seen = reached.distance_by_root[command]
        callables = {
            name
            for name in seen
            if graph.symbols[name].kind in {SymbolKind.FUNCTION, SymbolKind.METHOD}
        }
        leaves = {
            name
            for name in callables
            if not any(
                target in callables for target in graph.edges.get(name, ()) if target != name
            )
        }
        direct = {
            target
            for target in graph.edges.get(root, ())
            if target in graph.symbols
            and graph.symbols[target].kind in {SymbolKind.FUNCTION, SymbolKind.METHOD}
        }
        metrics.append(
            CommandMetrics(
                command,
                len(direct),
                len(callables),
                max((seen[name] for name in callables), default=0),
                len(leaves),
                len({graph.symbols[name].module for name in callables}),
            )
        )
    return metrics
