import ast
import io
import re
import tokenize
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

RAW_TYPE_NAMES = frozenset({"Any", "bool", "bytes", "complex", "float", "int", "object", "str"})
CONTAINER_NAMES = frozenset(
    {
        "Counter",
        "DefaultDict",
        "Dict",
        "FrozenSet",
        "List",
        "Mapping",
        "MutableMapping",
        "MutableSequence",
        "MutableSet",
        "Sequence",
        "Set",
        "defaultdict",
        "dict",
        "frozenset",
        "list",
        "set",
        "tuple",
    }
)
MAPPING_NAMES = frozenset(
    {"DefaultDict", "Dict", "Mapping", "MutableMapping", "defaultdict", "dict", "Counter"}
)
FORBIDDEN_GENERIC_TYPES = frozenset(
    {
        "AtLeastOneFloat",
        "FiniteFloat",
        "NonBlankText",
        "NonNegativeFloat",
        "NonNegativeInt",
        "OpenUnitFloat",
        "OpenUnitInterval",
        "PositiveFloat",
        "PositiveInt",
        "Sha256Hex",
        "SignedInt",
        "SignedUnitFloat",
        "UnitFloat",
        "UnitInterval",
    }
)
SCALAR_CONVERSIONS = frozenset({"bool", "bytes", "complex", "float", "int", "str"})
ESCAPE_HATCH_CALLS = frozenset(
    {"cast", "delattr", "eval", "exec", "getattr", "hasattr", "setattr", "vars", "__import__"}
)
SUPPRESSION_PATTERN = re.compile(
    r"#\s*(type:\s*ignore|noqa|pyright:|pylint:|fmt:|nosec|pragma)", re.IGNORECASE
)
LOG_METHODS = frozenset({"debug", "info", "warning", "error", "exception", "critical"})
LOG_SETUP_CALLS = frozenset({"get_logger", "configure_logging", "execute_with_failure_logging"})
FILESYSTEM_METHODS = frozenset(
    {
        "mkdir",
        "open",
        "read_bytes",
        "read_text",
        "rmdir",
        "unlink",
        "write_bytes",
        "write_text",
    }
)
ENUM_BASE_NAMES = frozenset({"Enum", "IntEnum", "StrEnum", "Flag", "IntFlag"})
LIBRARY_STORAGE_GENERICS = frozenset({"IO", "array", "NDArray", "ndarray", "dtype"})
PREDICATE_PREFIXES = ("is_", "has_", "can_", "should_", "_is_", "_has_")


@dataclass(frozen=True)
class Finding:
    rule: str
    location: str
    detail: str
    symbol: str = ""

    def __str__(self) -> str:
        scope = f" in {self.symbol}" if self.symbol else ""
        return f"[{self.rule}] {self.location}{scope}: {self.detail}"


@dataclass(frozen=True)
class SourceUnit:
    label: str
    text: str
    tree: ast.Module
    is_type_module: bool = False

    @staticmethod
    def parse(label: str, text: str, is_type_module: bool = False) -> "SourceUnit":
        return SourceUnit(label, text, ast.parse(text, filename=label), is_type_module)

    def at(self, node: ast.AST) -> str:
        return f"{self.label}:{getattr(node, 'lineno', 0)}"

    def symbol_at(self, node: ast.AST) -> str:
        line = getattr(node, "lineno", 0)
        best = ""
        best_span = 10**9
        for qualname, definition in definitions(self):
            end = definition.end_lineno or definition.lineno
            if definition.lineno <= line <= end and end - definition.lineno < best_span:
                best, best_span = qualname, end - definition.lineno
        return best

    def finding(self, rule: str, node: ast.AST, detail: str) -> Finding:
        return Finding(rule, self.at(node), detail, self.symbol_at(node))


def definitions(
    unit: SourceUnit,
) -> Iterator[tuple[str, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef]]:
    def walk(
        body: list[ast.stmt], prefix: str
    ) -> Iterator[tuple[str, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef]]:
        for node in body:
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                qualname = f"{prefix}{node.name}"
                yield qualname, node
                yield from walk(node.body, f"{qualname}.")

    yield from walk(unit.tree.body, "")


def function_nodes(
    unit: SourceUnit,
) -> Iterator[tuple[str, ast.FunctionDef | ast.AsyncFunctionDef]]:
    for qualname, node in definitions(unit):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            yield qualname, node


def _name_of(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _storage_slice_ids(node: ast.AST) -> set[int]:
    skipped: set[int] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Subscript) and _name_of(child.value) in LIBRARY_STORAGE_GENERICS:
            skipped.update(id(part) for part in ast.walk(child.slice))
    return skipped


def _names_in(node: ast.AST) -> Iterator[tuple[str, ast.AST]]:
    for child in ast.walk(node):
        if isinstance(child, ast.Name):
            yield child.id, child
        elif isinstance(child, ast.Attribute):
            yield child.attr, child
        elif isinstance(child, ast.Constant) and isinstance(child.value, str):
            try:
                inner = ast.parse(child.value, mode="eval")
            except SyntaxError:
                continue
            for name, _ in _names_in(inner):
                yield name, child


@dataclass(frozen=True)
class AnnotationSite:
    expression: ast.expr
    owner: str
    role: str


def annotation_sites(tree: ast.Module) -> Iterator[AnnotationSite]:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            arguments = (
                *node.args.posonlyargs,
                *node.args.args,
                node.args.vararg,
                *node.args.kwonlyargs,
                node.args.kwarg,
            )
            for argument in arguments:
                if argument is not None and argument.annotation is not None:
                    yield AnnotationSite(argument.annotation, node.name, "parameter")
            if node.returns is not None:
                yield AnnotationSite(node.returns, node.name, "return")
        elif isinstance(node, ast.AnnAssign):
            yield AnnotationSite(node.annotation, ast.unparse(node.target), "field")
        elif isinstance(node, ast.TypeAlias):
            yield AnnotationSite(node.value, ast.unparse(node.name), "alias")
        elif isinstance(node, ast.TypeVar) and node.bound is not None:
            yield AnnotationSite(node.bound, node.name, "type-parameter")
        elif isinstance(node, ast.ClassDef):
            for base in node.bases:
                yield AnnotationSite(base, node.name, "base")


def _is_predicate_return(site: AnnotationSite) -> bool:
    return (
        site.role == "return"
        and site.owner.startswith(PREDICATE_PREFIXES)
        and isinstance(site.expression, ast.Name)
        and site.expression.id == "bool"
    )


def type_alias_statements(tree: ast.Module) -> Iterator[tuple[str, ast.expr, ast.stmt]]:
    for node in tree.body:
        if isinstance(node, ast.TypeAlias):
            yield ast.unparse(node.name), node.value, node
        elif (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and _looks_like_type_name(node.targets[0].id)
            and _looks_like_type_expression(node.value)
        ):
            yield node.targets[0].id, node.value, node
        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.value is not None
            and _looks_like_type_name(node.target.id)
            and "TypeAlias" in ast.unparse(node.annotation)
        ):
            yield node.target.id, node.value, node


def _looks_like_type_name(name: str) -> bool:
    stripped = name.lstrip("_")
    return stripped[:1].isupper() and not stripped.isupper()


def _looks_like_type_expression(value: ast.expr) -> bool:
    if isinstance(value, ast.Name | ast.Attribute | ast.Subscript | ast.BinOp):
        return True
    return isinstance(value, ast.Call) and _name_of(value.func) in {"NewType", "TypeVar"}


def laundered_alias_names(type_module: SourceUnit) -> frozenset[str]:
    laundered: set[str] = set()
    for name, value, _ in type_alias_statements(type_module.tree):
        if _bare_primitive_expression(value, laundered):
            laundered.add(name)
    return frozenset(laundered)


def _bare_primitive_expression(value: ast.expr, laundered: set[str]) -> bool:
    if isinstance(value, ast.Name):
        return value.id in RAW_TYPE_NAMES or value.id in laundered
    if isinstance(value, ast.Attribute):
        return False
    if isinstance(value, ast.BinOp):
        return _bare_primitive_expression(value.left, laundered) or _bare_primitive_expression(
            value.right, laundered
        )
    if isinstance(value, ast.Subscript):
        head = _name_of(value.value)
        if head in {"Annotated", "NewType"}:
            return False
        if head in CONTAINER_NAMES | {"Callable", "Optional", "Union"}:
            return any(
                name in RAW_TYPE_NAMES or name in laundered for name, _ in _names_in(value.slice)
            )
    return False


def rule_forbidden_generic_types(unit: SourceUnit) -> list[Finding]:
    if unit.is_type_module:
        return []
    findings = [
        unit.finding("generic-constrained-type", node, name)
        for name, node in _names_in(unit.tree)
        if name in FORBIDDEN_GENERIC_TYPES and not isinstance(node, ast.Constant)
    ]
    for node in ast.walk(unit.tree):
        if isinstance(node, ast.ImportFrom):
            findings.extend(
                unit.finding("generic-constrained-type", node, alias.name)
                for alias in node.names
                if alias.name in FORBIDDEN_GENERIC_TYPES
            )
    return findings


def rule_primitive_annotations(unit: SourceUnit, laundered: frozenset[str]) -> list[Finding]:
    findings: list[Finding] = []
    flagged = RAW_TYPE_NAMES | laundered
    for site in annotation_sites(unit.tree):
        if _is_predicate_return(site):
            continue
        storage = _storage_slice_ids(site.expression)
        for name, node in _names_in(site.expression):
            if id(node) in storage:
                continue
            if name in flagged:
                kind = "laundered primitive alias" if name in laundered else "primitive type"
                findings.append(
                    unit.finding("primitive-annotation", node, f"{kind} {name} in {site.owner}")
                )
            elif name == "Literal":
                findings.append(
                    unit.finding("primitive-annotation", node, "Literal is a string enum")
                )
        findings.extend(_anonymous_container_findings(unit, site.expression))
    return findings


def _anonymous_container_findings(unit: SourceUnit, annotation: ast.expr) -> list[Finding]:
    heads = {id(node.value) for node in ast.walk(annotation) if isinstance(node, ast.Subscript)}
    return [
        unit.finding("anonymous-container", node, f"bare {node.id} annotation")
        for node in ast.walk(annotation)
        if isinstance(node, ast.Name)
        and node.id in MAPPING_NAMES | {"list", "set"}
        and id(node) not in heads
    ]


def rule_alias_ownership(unit: SourceUnit) -> list[Finding]:
    if unit.is_type_module:
        return []
    return [
        unit.finding("type-alias-outside-types", node, name)
        for name, _, node in type_alias_statements(unit.tree)
    ]


def rule_type_module_laundering(unit: SourceUnit) -> list[Finding]:
    if not unit.is_type_module:
        return []
    laundered: set[str] = set()
    findings: list[Finding] = []
    for name, value, node in type_alias_statements(unit.tree):
        if _bare_primitive_expression(value, laundered):
            laundered.add(name)
            findings.append(
                unit.finding("type-alias-laundering", node, f"{name} = {ast.unparse(value)}")
            )
    return findings


def rule_value_access(unit: SourceUnit) -> list[Finding]:
    return [
        unit.finding("enum-value-access", node, ast.unparse(node))
        for node in ast.walk(unit.tree)
        if isinstance(node, ast.Attribute) and node.attr in {"value", "_value_"}
    ]


def rule_scalar_conversions(unit: SourceUnit) -> list[Finding]:
    return [
        unit.finding("scalar-conversion", node, ast.unparse(node)[:80])
        for node in ast.walk(unit.tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in SCALAR_CONVERSIONS
    ]


def _call_target(node: ast.Call) -> str | None:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
        if func.value.id == "typing" and func.attr == "cast":
            return "cast"
        if func.value.id == "object" and func.attr == "__setattr__":
            return "__setattr__"
        if func.value.id == "builtins":
            return func.attr
    return None


def rule_escape_hatches(unit: SourceUnit) -> list[Finding]:
    findings: list[Finding] = []
    for node in ast.walk(unit.tree):
        if isinstance(node, ast.Call):
            target = _call_target(node)
            if target in ESCAPE_HATCH_CALLS | {"__setattr__"}:
                findings.append(unit.finding("escape-hatch", node, ast.unparse(node)[:80]))
        elif isinstance(node, ast.Name | ast.Attribute):
            if _name_of(node) in {"Any", "object"}:
                findings.append(unit.finding("escape-hatch", node, ast.unparse(node)))
        elif isinstance(node, ast.ImportFrom):
            findings.extend(
                unit.finding("escape-hatch", node, f"import {alias.name}")
                for alias in node.names
                if alias.name in {"Any", "cast"}
            )
    return findings


def rule_print(unit: SourceUnit) -> list[Finding]:
    return [
        unit.finding("print", node, "print() is neither observability nor CLI rendering")
        for node in ast.walk(unit.tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "print"
    ]


def _comment_lines(text: str) -> Iterator[tuple[int, str]]:
    try:
        for token in tokenize.generate_tokens(io.StringIO(text).readline):
            if token.type == tokenize.COMMENT:
                yield token.start[0], token.string
    except (tokenize.TokenError, IndentationError):
        yield 0, "<untokenizable source>"


def rule_suppressions(unit: SourceUnit) -> list[Finding]:
    findings: list[Finding] = []
    for number, comment in _comment_lines(unit.text):
        rule = "suppression" if SUPPRESSION_PATTERN.search(comment) else "comment"
        findings.append(Finding(rule, f"{unit.label}:{number}", comment))
    return findings


def _is_string_literal(node: ast.expr) -> bool:
    return isinstance(node, ast.Constant) and isinstance(node.value, str | bytes)


def _contains_string_literal(node: ast.expr) -> bool:
    if _is_string_literal(node):
        return True
    if isinstance(node, ast.Tuple | ast.Set | ast.List):
        return any(_contains_string_literal(item) for item in node.elts)
    if isinstance(node, ast.Call) and _name_of(node.func) in {"frozenset", "set", "tuple", "list"}:
        return any(_contains_string_literal(argument) for argument in node.args)
    return False


def _parents(tree: ast.AST) -> dict[ast.AST, ast.AST]:
    parents: dict[ast.AST, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[child] = parent
    return parents


def _is_member_name(node: ast.expr) -> bool:
    return isinstance(node, ast.Attribute) and node.attr == "name"


def _call_on_member_name(node: ast.AST, method: str) -> ast.Call | None:
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == method
        and _is_member_name(node.func.value)
    ):
        return node
    return None


def _inside_display_text(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> bool:
    current: ast.AST | None = node
    while current is not None:
        current = parents.get(current)
        if isinstance(current, ast.Call) and _name_of(current.func) == "DisplayText":
            return True
    return False


def _call_contains_string_literal(node: ast.Call) -> bool:
    return any(
        _contains_string_literal(item) for item in (*node.args, *(kw.value for kw in node.keywords))
    )


_ATTACK_VOCABULARY_METHODS = frozenset({"attack_type", "attack_type_name"})
_ANONYMOUS_MAPPING_KEYS = frozenset({"Any", "object", "str"})


def _owns_attack_vocabulary(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> bool:
    current: ast.AST | None = node
    while current is not None:
        current = parents.get(current)
        if isinstance(current, ast.FunctionDef | ast.AsyncFunctionDef):
            if current.name in _ATTACK_VOCABULARY_METHODS:
                return True
            continue
        if isinstance(current, ast.ClassDef):
            return any(
                isinstance(item, ast.FunctionDef | ast.AsyncFunctionDef)
                and item.name in _ATTACK_VOCABULARY_METHODS
                for item in current.body
            )
    return False


def _expr_is_anonymous_key(node: ast.expr) -> bool:
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        return _expr_is_anonymous_key(node.left) or _expr_is_anonymous_key(node.right)
    if isinstance(node, ast.Tuple):
        return any(_expr_is_anonymous_key(item) for item in node.elts)
    if isinstance(node, ast.Subscript):
        sliced = node.slice
        if isinstance(sliced, ast.Tuple):
            return any(_expr_is_anonymous_key(item) for item in sliced.elts)
        return _expr_is_anonymous_key(sliced)
    if isinstance(node, ast.Name | ast.Attribute):
        return _name_of(node) in _ANONYMOUS_MAPPING_KEYS
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value in _ANONYMOUS_MAPPING_KEYS
    return False


def _mapping_key_expression(node: ast.Subscript) -> ast.expr | None:
    if _name_of(node.value) not in MAPPING_NAMES:
        return None
    if isinstance(node.slice, ast.Tuple) and node.slice.elts:
        return node.slice.elts[0]
    return node.slice


def rule_string_domains(unit: SourceUnit) -> list[Finding]:
    findings: list[Finding] = []
    parents = _parents(unit.tree)
    for node in ast.walk(unit.tree):
        if isinstance(node, ast.Compare):
            if any(_contains_string_literal(item) for item in (node.left, *node.comparators)):
                findings.append(unit.finding("string-comparison", node, ast.unparse(node)[:80]))
        elif isinstance(node, ast.match_case):
            findings.extend(
                unit.finding("string-match-case", pattern, ast.unparse(pattern))
                for pattern in ast.walk(node.pattern)
                if isinstance(pattern, ast.MatchValue) and _is_string_literal(pattern.value)
            )
        elif isinstance(node, ast.Subscript) and _is_string_literal(node.slice):
            findings.append(unit.finding("string-key-lookup", node, ast.unparse(node)[:80]))
        elif isinstance(node, ast.Subscript) and _is_member_name(node.slice):
            findings.append(unit.finding("enum-name-lookup", node, ast.unparse(node)[:80]))
        elif isinstance(node, ast.Dict):
            keys = [key for key in node.keys if key is not None and _is_string_literal(key)]
            if keys:
                findings.append(
                    unit.finding(
                        "anonymous-structured-dict",
                        node,
                        f"keys {[ast.unparse(key) for key in keys][:3]}",
                    )
                )
        prefix = _call_on_member_name(node, "startswith")
        if prefix is not None and _call_contains_string_literal(prefix):
            findings.append(unit.finding("enum-name-prefix", prefix, ast.unparse(prefix)[:80]))
        rewritten = _call_on_member_name(node, "replace")
        if rewritten is not None and not _inside_display_text(rewritten, parents):
            findings.append(
                unit.finding("enum-name-rewrite", rewritten, ast.unparse(rewritten)[:80])
            )
        if (
            isinstance(node, ast.Call)
            and _name_of(node.func) == "AttackTypeName"
            and any(_is_member_name(argument) for argument in node.args)
        ):
            findings.append(unit.finding("enum-name-rewrite", node, ast.unparse(node)[:80]))
        if isinstance(node, ast.Call) and _name_of(node.func) == "AttackTypeName":
            arguments = (*node.args, *(keyword.value for keyword in node.keywords))
            literal = any(_is_string_literal(argument) for argument in arguments)
            if literal and not _owns_attack_vocabulary(node, parents):
                findings.append(unit.finding("attack-type-literal", node, ast.unparse(node)[:80]))
        if isinstance(node, ast.Subscript):
            key = _mapping_key_expression(node)
            if key is not None and _expr_is_anonymous_key(key):
                findings.append(
                    unit.finding("anonymous-structured-dict", node, ast.unparse(node)[:80])
                )
    return findings


def _is_semantic_name(name: str) -> bool:
    stripped = name.lstrip("_")
    return len(stripped) > 1 and stripped.isupper() and name != "__all__"


def _is_regular_expression(value: ast.expr) -> bool:
    return (
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Attribute)
        and value.func.attr == "compile"
        and isinstance(value.func.value, ast.Name)
        and value.func.value.id == "re"
        and bool(value.args)
        and _is_string_literal(value.args[0])
    )


def _constant_value_kind(value: ast.expr) -> str:
    if isinstance(value, ast.Constant):
        return type(value.value).__name__
    if isinstance(value, ast.UnaryOp) and isinstance(value.operand, ast.Constant):
        return type(value.operand.value).__name__
    if isinstance(value, ast.Tuple | ast.Set | ast.List | ast.Dict):
        return "collection"
    if isinstance(value, ast.BinOp):
        return "expression"
    if isinstance(value, ast.Call):
        callee = _name_of(value.func)
        if callee in {"frozenset", "tuple", "set", "list", "dict", "range"}:
            return "collection"
        return f"call:{callee}"
    return "reference"


def rule_loose_constants(unit: SourceUnit) -> list[Finding]:
    if unit.is_type_module:
        return []
    findings: list[Finding] = []

    def scan(body: list[ast.stmt], scope: str) -> None:
        for node in body:
            targets: list[ast.expr] = []
            value: ast.expr | None = None
            if isinstance(node, ast.Assign):
                targets, value = node.targets, node.value
            elif isinstance(node, ast.AnnAssign) and node.value is not None:
                targets, value = [node.target], node.value
                if "Final" in ast.unparse(node.annotation) and isinstance(node.target, ast.Name):
                    findings.append(
                        unit.finding("final-constant", node, f"{scope}{node.target.id}")
                    )
                    continue
            if value is not None and not _is_regular_expression(value):
                for target in targets:
                    if isinstance(target, ast.Name) and _is_semantic_name(target.id):
                        findings.append(
                            unit.finding(
                                "loose-constant",
                                node,
                                f"{scope}{target.id} ({_constant_value_kind(value)})",
                            )
                        )
            if isinstance(node, ast.ClassDef) and not any(
                _name_of(base) in ENUM_BASE_NAMES for base in node.bases
            ):
                scan(node.body, f"{node.name}.")

    scan(unit.tree.body, "")
    return findings


def project_callable_names(units: tuple[SourceUnit, ...]) -> frozenset[str]:
    names: set[str] = set()
    for unit in units:
        for node in ast.walk(unit.tree):
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                names.add(node.name)
            elif isinstance(node, ast.Assign):
                names.update(target.id for target in node.targets if isinstance(target, ast.Name))
    return frozenset(names)


def _imported_project_names(tree: ast.Module) -> frozenset[str]:
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("pace"):
            names.update(alias.asname or alias.name for alias in node.names)
    return frozenset(names)


def _is_library_call(
    node: ast.Call, project_names: frozenset[str], imported: frozenset[str]
) -> bool:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id not in project_names and func.id not in imported
    if isinstance(func, ast.Attribute):
        return func.attr not in project_names
    return False


def _constants_within(*nodes: ast.AST | None) -> set[int]:
    return {
        id(part)
        for node in nodes
        if node is not None
        for part in ast.walk(node)
        if isinstance(part, ast.Constant)
    }


def _literal_strings_in_code(
    unit: SourceUnit, project_names: frozenset[str]
) -> Iterator[tuple[str, ast.Constant]]:
    skipped: set[int] = set()
    imported = _imported_project_names(unit.tree)
    for node in ast.walk(unit.tree):
        if isinstance(node, ast.JoinedStr):
            skipped |= _constants_within(node)
        elif isinstance(node, ast.Raise):
            skipped |= _constants_within(node.exc)
        elif isinstance(node, ast.Call):
            if _name_of(node.func) in {"BadParameter", "echo"} or _is_library_call(
                node, project_names, imported
            ):
                skipped |= _constants_within(*node.args, *(item.value for item in node.keywords))
        elif isinstance(node, ast.ClassDef) and any(
            _name_of(base) in ENUM_BASE_NAMES for base in node.bases
        ):
            for statement in node.body:
                if isinstance(statement, ast.Assign):
                    skipped |= _constants_within(statement.value)
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            skipped |= _constants_within(
                node.returns, *(argument.annotation for argument in node.args.args)
            )
        elif isinstance(node, ast.AnnAssign):
            skipped |= _constants_within(node.annotation)
    for node in ast.walk(unit.tree):
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and id(node) not in skipped
            and node.value
        ):
            yield node.value, node


def repeated_string_literals(units: tuple[SourceUnit, ...], minimum: int) -> list[Finding]:
    occurrences: dict[str, list[str]] = defaultdict(list)
    project_names = project_callable_names(units)
    for unit in units:
        for text, node in _literal_strings_in_code(unit, project_names):
            occurrences[text].append(unit.at(node))
    return [
        Finding(
            "repeated-string-literal",
            sites[0],
            f"{text!r} appears {len(sites)} times: {', '.join(sites[:4])}",
        )
        for text, sites in sorted(occurrences.items())
        if len(sites) >= minimum
    ]


def rule_enum_hosting(unit: SourceUnit) -> list[Finding]:
    if unit.is_type_module:
        return []
    return [
        unit.finding("enum-outside-types", node, node.name)
        for node in ast.walk(unit.tree)
        if isinstance(node, ast.ClassDef)
        and any(_name_of(base) in ENUM_BASE_NAMES for base in node.bases)
    ]


def _is_inert(statement: ast.stmt) -> bool:
    if isinstance(statement, ast.Pass | ast.Continue | ast.Break):
        return True
    return isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant)


def rule_exception_handling(unit: SourceUnit) -> list[Finding]:
    findings: list[Finding] = []
    for node in ast.walk(unit.tree):
        if not isinstance(node, ast.ExceptHandler):
            continue
        if node.type is None or _name_of(node.type) in {"Exception", "BaseException"}:
            findings.append(unit.finding("broad-except", node, "catch-all handler"))
        if all(_is_inert(statement) for statement in node.body):
            findings.append(
                unit.finding("swallowed-exception", node, "handler discards the failure")
            )
    return findings


def contains_logging(node: ast.AST) -> bool:
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            func = child.func
            if isinstance(func, ast.Attribute) and func.attr in LOG_METHODS:
                return True
            if _name_of(func) in LOG_SETUP_CALLS:
                return True
    return False


def logged_events(node: ast.AST) -> set[str]:
    return {
        child.attr
        for child in ast.walk(node)
        if isinstance(child, ast.Attribute)
        and isinstance(child.value, ast.Name)
        and child.value.id == "LogEvent"
    }


def performs_filesystem_io(node: ast.AST) -> bool:
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            func = child.func
            if isinstance(func, ast.Attribute) and func.attr in FILESYSTEM_METHODS:
                return True
            if isinstance(func, ast.Name) and func.id == "open":
                return True
    return False


def cyclomatic_complexity(node: ast.FunctionDef | ast.AsyncFunctionDef) -> int:
    score = 1
    for child in ast.walk(node):
        if isinstance(
            child,
            ast.If
            | ast.For
            | ast.While
            | ast.ExceptHandler
            | ast.With
            | ast.IfExp
            | ast.comprehension
            | ast.match_case
            | ast.Assert,
        ):
            score += 1
        elif isinstance(child, ast.BoolOp):
            score += len(child.values) - 1
    return score


def function_length(node: ast.FunctionDef | ast.AsyncFunctionDef) -> int:
    return (node.end_lineno or node.lineno) - node.lineno + 1


def argument_count(node: ast.FunctionDef | ast.AsyncFunctionDef) -> int:
    arguments = (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs)
    counted = len([item for item in arguments if item.arg not in {"self", "cls"}])
    return counted + int(node.args.vararg is not None) + int(node.args.kwarg is not None)


def import_cycles(edges: dict[str, set[str]]) -> list[tuple[str, ...]]:
    index: dict[str, int] = {}
    low: dict[str, int] = {}
    stack: list[str] = []
    on_stack: set[str] = set()
    cycles: list[tuple[str, ...]] = []
    counter = 0

    def visit(vertex: str) -> None:
        nonlocal counter
        index[vertex] = low[vertex] = counter
        counter += 1
        stack.append(vertex)
        on_stack.add(vertex)
        for neighbor in sorted(edges.get(vertex, ())):
            if neighbor not in index:
                visit(neighbor)
                low[vertex] = min(low[vertex], low[neighbor])
            elif neighbor in on_stack:
                low[vertex] = min(low[vertex], index[neighbor])
        if low[vertex] == index[vertex]:
            component: list[str] = []
            while True:
                member = stack.pop()
                on_stack.discard(member)
                component.append(member)
                if member == vertex:
                    break
            if len(component) > 1:
                cycles.append(tuple(sorted(component)))

    for vertex in sorted(edges):
        if vertex not in index:
            visit(vertex)
    return cycles


def read_units(paths: tuple[Path, ...], root: Path, type_module: Path) -> tuple[SourceUnit, ...]:
    return tuple(
        SourceUnit.parse(
            str(path.relative_to(root)),
            path.read_text(encoding="utf-8"),
            is_type_module=path == type_module,
        )
        for path in paths
    )


def domain_findings(units: tuple[SourceUnit, ...], repeated_minimum: int = 2) -> list[Finding]:
    laundered: frozenset[str] = frozenset()
    for unit in units:
        if unit.is_type_module:
            laundered = laundered_alias_names(unit)
    findings: list[Finding] = []
    for unit in units:
        findings.extend(rule_forbidden_generic_types(unit))
        findings.extend(rule_primitive_annotations(unit, laundered))
        findings.extend(rule_alias_ownership(unit))
        findings.extend(rule_type_module_laundering(unit))
        findings.extend(rule_value_access(unit))
        findings.extend(rule_scalar_conversions(unit))
        findings.extend(rule_escape_hatches(unit))
        findings.extend(rule_print(unit))
        findings.extend(rule_suppressions(unit))
        findings.extend(rule_string_domains(unit))
        findings.extend(rule_loose_constants(unit))
        findings.extend(rule_enum_hosting(unit))
        findings.extend(rule_exception_handling(unit))
    findings.extend(repeated_string_literals(units, repeated_minimum))
    return findings


def called_names(node: ast.AST) -> set[str]:
    names: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            name = _name_of(child.func)
            if name is not None:
                names.add(name)
        elif isinstance(child, ast.Name):
            names.add(child.id)
    return names
