import textwrap
from collections.abc import Callable
from functools import cache

import pytest

from tests.architecture.exemptions import (
    EXEMPTIONS,
    Exemption,
    apply_exemptions,
    regression_test_exists,
)
from tests.architecture.rules import (
    Finding,
    SourceUnit,
    domain_findings,
    laundered_alias_names,
    read_units,
    repeated_string_literals,
    rule_alias_ownership,
    rule_enum_hosting,
    rule_escape_hatches,
    rule_exception_handling,
    rule_forbidden_generic_types,
    rule_loose_constants,
    rule_primitive_annotations,
    rule_print,
    rule_scalar_conversions,
    rule_string_domains,
    rule_suppressions,
    rule_type_module_laundering,
    rule_value_access,
)
from tests.architecture.source_index import (
    REPOSITORY_ROOT,
    SOURCE_ROOT,
    TYPE_MODULE,
    scan_production,
)


@cache
def _production_units() -> tuple[SourceUnit, ...]:
    report = scan_production()
    return read_units(report.parsed, REPOSITORY_ROOT, TYPE_MODULE)


@cache
def _remaining() -> tuple[list[Finding], list[Exemption]]:
    remaining, unused = apply_exemptions(domain_findings(_production_units()))
    return remaining, unused


def _of(*rules: str) -> list[str]:
    return [str(finding) for finding in _remaining()[0] if finding.rule in rules]


def _unit(source: str, label: str = "src/pace/example.py", types: bool = False) -> SourceUnit:
    return SourceUnit.parse(label, textwrap.dedent(source), is_type_module=types)


def _laundered(types_source: str) -> frozenset[str]:
    return laundered_alias_names(_unit(types_source, "src/pace/types.py", types=True))


def test_production_scan_covers_every_python_file_under_the_package_root() -> None:
    report = scan_production()
    assert set(report.parsed) == set(SOURCE_ROOT.rglob("*.py"))


def test_no_generic_constrained_types_outside_the_type_module() -> None:
    assert not _of("generic-constrained-type")


def test_no_primitive_or_laundered_annotations_in_production() -> None:
    assert not _of("primitive-annotation", "anonymous-container")


def test_no_type_alias_laundering_or_alias_ownership_leaks() -> None:
    assert not _of("type-alias-laundering", "type-alias-outside-types")


def test_enum_values_are_never_read_through_dot_value() -> None:
    assert not _of("enum-value-access")


def test_scalar_conversions_exist_only_at_exact_external_boundaries() -> None:
    assert not _of("scalar-conversion")


def test_no_type_escape_hatches_remain() -> None:
    assert not _of("escape-hatch")


def test_no_print_calls_in_production() -> None:
    assert not _of("print")


def test_no_comments_or_suppressions_in_production() -> None:
    assert not _of("comment", "suppression")


def test_no_string_backed_finite_domains_in_production() -> None:
    assert not _of(
        "string-comparison",
        "string-match-case",
        "string-key-lookup",
        "repeated-string-literal",
        "enum-name-prefix",
        "enum-name-rewrite",
        "enum-name-lookup",
        "attack-type-literal",
    )


def test_no_anonymous_structured_dictionaries_in_production() -> None:
    assert not _of("anonymous-structured-dict")


def test_no_loose_semantic_constants_in_production() -> None:
    assert not _of("loose-constant", "final-constant")


def test_project_enums_live_in_the_type_module() -> None:
    assert not _of("enum-outside-types")


def test_no_failure_is_silently_swallowed() -> None:
    assert not _of("swallowed-exception", "broad-except")


def test_every_exemption_is_exact_used_and_regression_tested() -> None:
    unused = _remaining()[1]
    assert not unused, unused
    for exemption in EXEMPTIONS:
        assert exemption.path.startswith("src/pace/"), exemption
        assert "*" not in exemption.path, exemption
        assert "*" not in exemption.symbol, exemption
        assert exemption.detail, exemption
        assert regression_test_exists(exemption), exemption
    assert len(EXEMPTIONS) <= 12


MUTATIONS: dict[str, tuple[str, str, Callable[[SourceUnit], list[Finding]]]] = {
    "generic-int": (
        "generic-constrained-type",
        "from pace.types import PositiveInt\ncount: PositiveInt = 1\n",
        rule_forbidden_generic_types,
    ),
    "generic-qualified": (
        "generic-constrained-type",
        "import pace.types as t\ndef f(a: t.NonNegativeFloat) -> None: ...\n",
        rule_forbidden_generic_types,
    ),
    "generic-renamed-import": (
        "generic-constrained-type",
        "from pace.types import PositiveInt as Count\n",
        rule_forbidden_generic_types,
    ),
    "value-access": (
        "enum-value-access",
        "selected = option.value\n",
        rule_value_access,
    ),
    "scalar-float": ("scalar-conversion", "x = float(raw)\n", rule_scalar_conversions),
    "scalar-int": ("scalar-conversion", "x = int(raw)\n", rule_scalar_conversions),
    "scalar-str": ("scalar-conversion", "x = str(raw)\n", rule_scalar_conversions),
    "scalar-bool": ("scalar-conversion", "x = bool(raw)\n", rule_scalar_conversions),
    "escape-any": (
        "escape-hatch",
        "from typing import Any\ndef f(x: Any) -> None: ...\n",
        rule_escape_hatches,
    ),
    "escape-object": ("escape-hatch", "def f(x: object) -> None: ...\n", rule_escape_hatches),
    "escape-cast": ("escape-hatch", "y = cast(int, x)\n", rule_escape_hatches),
    "escape-typing-cast": ("escape-hatch", "y = typing.cast(int, x)\n", rule_escape_hatches),
    "escape-getattr": ("escape-hatch", "y = getattr(x, name)\n", rule_escape_hatches),
    "escape-setattr": (
        "escape-hatch",
        "object.__setattr__(self, 'a', 1)\n",
        rule_escape_hatches,
    ),
    "print": ("print", "def run() -> None:\n    print('x')\n", rule_print),
    "type-ignore": ("suppression", "x = 1  # type: ignore\n", rule_suppressions),
    "noqa": ("suppression", "import os  # noqa\n", rule_suppressions),
    "pyright-ignore": ("suppression", "x = 1  # pyright: ignore\n", rule_suppressions),
    "plain-comment": ("comment", "x = 1  # explanation\n", rule_suppressions),
    "string-policy-comparison": (
        "string-comparison",
        "if policy == 'local':\n    pass\n",
        rule_string_domains,
    ),
    "string-membership-set": (
        "string-comparison",
        "ok = strategy in {'a', 'b', 'c'}\n",
        rule_string_domains,
    ),
    "string-membership-tuple": (
        "string-comparison",
        "ok = status in ('done', 'failed')\n",
        rule_string_domains,
    ),
    "string-match": (
        "string-match-case",
        "match status:\n    case 'done':\n        pass\n",
        rule_string_domains,
    ),
    "string-key": ("string-key-lookup", "x = record['ts']\n", rule_string_domains),
    "enum-name-prefix": (
        "enum-name-prefix",
        "if self.name.startswith('FEATURE_'):\n    return Family.FEATURE\n",
        rule_string_domains,
    ),
    "enum-name-rewrite": (
        "enum-name-rewrite",
        "return AttackTypeName(self.name.replace('_', '/', 1))\n",
        rule_string_domains,
    ),
    "enum-name-as-identifier": (
        "enum-name-rewrite",
        "return AttackTypeName(self.name)\n",
        rule_string_domains,
    ),
    "enum-name-lookup": (
        "enum-name-lookup",
        "return StudyStratum[dataset.name]\n",
        rule_string_domains,
    ),
    "anonymous-dict": (
        "anonymous-structured-dict",
        "payload = {'scientific': 1, 'splitting': 2}\n",
        rule_string_domains,
    ),
    "anonymous-dict-annotation": (
        "anonymous-structured-dict",
        "groups: dict[str, int] = {}\n",
        rule_string_domains,
    ),
    "anonymous-dict-union-key": (
        "anonymous-structured-dict",
        "groups: dict[str | None, int] = {}\n",
        rule_string_domains,
    ),
    "anonymous-defaultdict": (
        "anonymous-structured-dict",
        "counts: defaultdict[Any, int]\n",
        rule_string_domains,
    ),
    "anonymous-tuple-key": (
        "anonymous-structured-dict",
        "groups: dict[tuple[str, int], int] = {}\n",
        rule_string_domains,
    ),
    "attack-type-literal": (
        "attack-type-literal",
        "def normalize(label: SourceCell) -> AttackTypeName:\n    return AttackTypeName('MITM')\n",
        rule_string_domains,
    ),
    "attack-type-empty-literal": (
        "attack-type-literal",
        "class Key:\n"
        "    def sort_key(self) -> AttackTypeName:\n"
        "        return AttackTypeName('')\n",
        rule_string_domains,
    ),
    "loose-string-constant": (
        "loose-constant",
        "LOCAL_THRESHOLD = 'local_threshold'\n",
        rule_loose_constants,
    ),
    "loose-integer-constant": ("loose-constant", "ROW_CAP = 5_000\n", rule_loose_constants),
    "loose-float-constant": ("loose-constant", "FRACTION = 0.25\n", rule_loose_constants),
    "loose-choice-collection": (
        "loose-constant",
        "MODES = ('fast', 'slow')\n",
        rule_loose_constants,
    ),
    "loose-final-constant": (
        "final-constant",
        "from typing import Final\nLIMIT: Final = 3\n",
        rule_loose_constants,
    ),
    "class-level-constant": (
        "loose-constant",
        "class Policy:\n    DEFAULT = 'local'\n",
        rule_loose_constants,
    ),
    "enum-outside-types": (
        "enum-outside-types",
        "from enum import Enum\nclass Mode(Enum):\n    A = 1\n",
        rule_enum_hosting,
    ),
    "swallowed": (
        "swallowed-exception",
        "try:\n    run()\nexcept ValueError:\n    pass\n",
        rule_exception_handling,
    ),
    "broad-except": (
        "broad-except",
        "try:\n    run()\nexcept Exception as error:\n    raise\n",
        rule_exception_handling,
    ),
    "alias-outside-types": ("type-alias-outside-types", "ClientLike = int\n", rule_alias_ownership),
    "private-alias-outside-types": (
        "type-alias-outside-types",
        "_Key = tuple[str, int]\n",
        rule_alias_ownership,
    ),
    "type-statement-outside-types": (
        "type-alias-outside-types",
        "type Key = int\n",
        rule_alias_ownership,
    ),
}


@pytest.mark.parametrize("name", sorted(MUTATIONS))
def test_rule_rejects_its_mutation(name: str) -> None:
    rule, source, check = MUTATIONS[name]
    findings = check(_unit(source))
    assert any(finding.rule == rule for finding in findings), (name, findings)


PRIMITIVE_MUTATIONS = {
    "parameter": "def f(count: int) -> None: ...\n",
    "return": "def f() -> float: ...\n",
    "optional": "def f(x: Optional[int]) -> None: ...\n",
    "union-none": "def f(x: int | None) -> None: ...\n",
    "list-float": "def f(x: list[float]) -> None: ...\n",
    "tuple-str": "def f(x: tuple[str, ...]) -> None: ...\n",
    "sequence-float": "def f(x: Sequence[float]) -> None: ...\n",
    "mapping-object": "def f(x: Mapping[str, object]) -> None: ...\n",
    "dict-any": "def f(x: dict[str, Any]) -> None: ...\n",
    "callable": "def f(x: Callable[[str, int], float]) -> None: ...\n",
    "annotated": "def f(x: Annotated[str, Field()]) -> None: ...\n",
    "dataclass-field": "@dataclass\nclass A:\n    seed: int\n",
    "model-field": "class M(BaseModel):\n    alpha: float\n",
    "local-variable": "def f() -> None:\n    total: int = 0\n",
    "forward-reference": "def f(x: 'list[int]') -> None: ...\n",
    "literal": "def f(x: Literal['a', 'b']) -> None: ...\n",
    "bare-dict": "def f(x: dict) -> None: ...\n",
    "protocol-method": "class P(Protocol):\n    def run(self, count: int) -> None: ...\n",
    "type-statement": "type Payload = dict[str, int]\n",
    "bool-parameter": "def f(flag: bool) -> None: ...\n",
}


@pytest.mark.parametrize("name", sorted(PRIMITIVE_MUTATIONS))
def test_primitive_annotation_rule_rejects_nested_forms(name: str) -> None:
    findings = rule_primitive_annotations(_unit(PRIMITIVE_MUTATIONS[name]), frozenset())
    assert findings, name


@pytest.mark.parametrize(
    ("alias_source", "alias"),
    [
        ("ClientLike = int\n", "ClientLike"),
        ("ThresholdLike = float\n", "ThresholdLike"),
        ("PolicyLike = str\n", "PolicyLike"),
        ("Payload = dict[str, Any]\n", "Payload"),
        ("Payload: TypeAlias = dict[str, object]\n", "Payload"),
        ("type Text = str\n", "Text"),
        ("Renamed = ClientLike\nClientLike = int\n", "ClientLike"),
        ("Sequenced = Sequence[float]\n", "Sequenced"),
        ("Either = int | None\n", "Either"),
    ],
)
def test_alias_laundering_is_detected_and_propagates_to_consumers(
    alias_source: str, alias: str
) -> None:
    laundered = _laundered(alias_source)
    assert alias in laundered
    assert rule_type_module_laundering(_unit(alias_source, "src/pace/types.py", types=True))
    consumer = _unit(f"def f(x: {alias}) -> None: ...\n")
    assert rule_primitive_annotations(consumer, laundered)


VALID_CONTROLS: dict[str, tuple[str, Callable[[SourceUnit], list[Finding]]]] = {
    "domain-annotations": (
        "def f(count: ClientCount, rate: DetectionRate) -> Threshold: ...\n",
        lambda unit: rule_primitive_annotations(unit, frozenset()),
    ),
    "predicate-return": (
        "def is_available() -> bool: ...\ndef has_rows() -> bool: ...\n",
        lambda unit: rule_primitive_annotations(unit, frozenset()),
    ),
    "library-storage-generics": (
        "def f(x: IO[bytes], y: array[float], z: NDArray[np.float64]) -> None: ...\n",
        lambda unit: rule_primitive_annotations(unit, frozenset()),
    ),
    "enum-comparison": (
        "if policy is ThresholdPolicy.LOCAL_THRESHOLD:\n    pass\n",
        rule_string_domains,
    ),
    "enum-match": (
        "match policy:\n    case ThresholdPolicy.LOCAL_THRESHOLD:\n        pass\n",
        rule_string_domains,
    ),
    "display-label-from-member-name": (
        "return DisplayText(self.name.replace('_', ' '))\n",
        rule_string_domains,
    ),
    "display-label-capitalized": (
        "return DisplayText(self.name.replace('_', ' ').capitalize())\n",
        rule_string_domains,
    ),
    "path-name-prefix": (
        "stale = entry.name.startswith(CacheFileName.STAGING)\n",
        rule_string_domains,
    ),
    "typed-record": ("record = Settings(alpha=1)\n", rule_string_domains),
    "domain-keyed-dict": (
        "groups: dict[StudyStratum, list[Record]] = {}\n",
        rule_string_domains,
    ),
    "attack-vocabulary-literal": (
        "class Family:\n"
        "    def attack_type(self) -> AttackType:\n"
        "        return AttackType(AttackTypeName('ARP_SPOOF'))\n",
        rule_string_domains,
    ),
    "attack-vocabulary-member": (
        "class Variation:\n"
        "    COMBO = AttackTypeName('GAFGYT/COMBO')\n"
        "    def attack_type(self) -> AttackType:\n"
        "        return AttackType(self.COMBO)\n",
        rule_string_domains,
    ),
    "numpy-item": ("x = np.mean(values).item()\n", rule_scalar_conversions),
    "polars-cast": ("expr = column.cast(pl.Float64)\n", rule_escape_hatches),
    "torch-eval": ("model.eval()\n", rule_escape_hatches),
    "dunder-all": ("__all__ = ['a']\n", rule_loose_constants),
    "regular-expression": ("FIELD = re.compile(r'x+')\n", rule_loose_constants),
    "enum-members": (
        "from enum import StrEnum\nclass Mode(StrEnum):\n    FAST = 'fast'\n",
        rule_loose_constants,
    ),
    "lower-case-binding": ("limit = 5\n", rule_loose_constants),
    "translated-failure": (
        "try:\n    run()\nexcept ValueError as error:\n    raise TypeError('x') from error\n",
        rule_exception_handling,
    ),
    "explicit-result": (
        "def f():\n    try:\n        return run()\n    except ValueError:\n        return None\n",
        rule_exception_handling,
    ),
    "logged-failure": (
        "try:\n    run()\nexcept ValueError:\n    logger.warning('x')\n",
        rule_exception_handling,
    ),
    "no-comments": ("x = 1\n", rule_suppressions),
    "string-in-comment-free-literal": ("x = '# not a comment'\n", rule_suppressions),
    "print-name-attribute": ("console.print('x')\n", rule_print),
}


@pytest.mark.parametrize("name", sorted(VALID_CONTROLS))
def test_rules_do_not_flag_valid_constructs(name: str) -> None:
    source, check = VALID_CONTROLS[name]
    assert not check(_unit(source)), name


def test_repeated_literal_rule_flags_project_vocabularies_but_not_library_options() -> None:
    defining = _unit("def process(mode):\n    return mode\n", "src/pace/a.py")
    repeated = _unit(
        "from pace.a import process\nprocess('local')\nprocess('local')\nprocess('local')\n",
        "src/pace/b.py",
    )
    findings = repeated_string_literals((defining, repeated), minimum=3)
    assert any("'local'" in finding.detail for finding in findings)
    library = _unit(
        "path.open('rb')\npath.open('rb')\npath.open('rb')\nnp.quantile(x, 0.1, method='linear')\n"
        "np.quantile(x, 0.1, method='linear')\nnp.quantile(x, 0.1, method='linear')\n",
        "src/pace/c.py",
    )
    assert not repeated_string_literals((defining, library), minimum=3)
    messages = _unit(
        "def f():\n    raise ValueError('same text')\ndef g():\n    raise ValueError('same text')\n"
        "def h():\n    raise ValueError('same text')\n",
        "src/pace/d.py",
    )
    assert not repeated_string_literals((defining, messages), minimum=3)


def test_exemption_matching_is_exact_site_and_detects_stale_entries() -> None:
    unit = _unit("def helper(value: object) -> None: ...\n", "src/pace/example.py")
    findings = rule_primitive_annotations(unit, frozenset())
    exemption = Exemption(
        "primitive-annotation",
        "src/pace/example.py",
        "helper",
        "primitive type object in helper",
        "demonstration",
        "tests/architecture/test_domain_rules.py::test_exemption_matching_is_exact_site_and_detects_stale_entries",
    )
    remaining, unused = apply_exemptions(findings, (exemption,))
    assert not remaining
    assert not unused
    other_symbol = Exemption(
        exemption.rule,
        exemption.path,
        "other",
        exemption.detail,
        "demonstration",
        exemption.regression_test,
    )
    remaining, unused = apply_exemptions(findings, (other_symbol,))
    assert remaining
    assert unused == [other_symbol]
    assert regression_test_exists(exemption)
    assert not regression_test_exists(
        Exemption("r", "p", "s", "d", "r", "tests/architecture/test_domain_rules.py::missing")
    )
