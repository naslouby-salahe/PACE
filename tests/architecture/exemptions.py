import ast
from dataclasses import dataclass

from tests.architecture.rules import Finding
from tests.architecture.source_index import REPOSITORY_ROOT


@dataclass(frozen=True)
class Exemption:
    rule: str
    path: str
    symbol: str
    detail: str
    reason: str
    regression_test: str


TORCH_REASON = (
    "torch is only partially typed; its objects are narrowed once by isinstance at this boundary"
)
SOURCE_TEST = "tests/datasets/test_common.py::"

EXEMPTIONS: tuple[Exemption, ...] = (
    Exemption(
        "primitive-annotation",
        "src/pace/types.py",
        "_reject_blank_location",
        "primitive type object in _reject_blank_location",
        "pydantic before-validators receive arbitrary decoded configuration input",
        "tests/test_config.py::test_yaml_values_are_validated_strictly",
    ),
    Exemption(
        "escape-hatch",
        "src/pace/types.py",
        "_reject_blank_location",
        "object",
        "pydantic before-validators receive arbitrary decoded configuration input",
        "tests/test_config.py::test_yaml_values_are_validated_strictly",
    ),
    Exemption(
        "primitive-annotation",
        "src/pace/datasets/common.py",
        "_ReadAhead.readable",
        "primitive type bool in readable",
        "io.RawIOBase fixes the return type of readable() as bool",
        "tests/datasets/test_common.py::"
        "test_read_ahead_streams_bytes_in_order_and_propagates_errors",
    ),
    Exemption(
        "scalar-conversion",
        "src/pace/datasets/common.py",
        "parse_decimal",
        "float(text)",
        "external CSV text is parsed into a numeric feature value exactly here",
        SOURCE_TEST + "test_decimal_text_is_parsed_at_the_boundary",
    ),
    Exemption(
        "scalar-conversion",
        "src/pace/datasets/common.py",
        "parse_integer",
        "int(text, base)",
        "external CSV text is parsed into a source integer exactly here",
        SOURCE_TEST + "test_integer_text_is_parsed_at_the_boundary",
    ),
    Exemption(
        "scalar-conversion",
        "src/pace/datasets/common.py",
        "_parse_integer_literal",
        "float(parse_integer(text, IntegerBase.AUTO))",
        "integer-valued source text is widened to a feature value exactly here",
        SOURCE_TEST + "test_integer_literals_widen_to_feature_values",
    ),
    Exemption(
        "primitive-annotation",
        "src/pace/experiment/models.py",
        "adapt_library_result",
        "primitive type object in adapt_library_result",
        TORCH_REASON,
        "tests/experiment/test_models.py::"
        "test_library_adapter_rejects_an_unexpected_implementation",
    ),
    Exemption(
        "escape-hatch",
        "src/pace/experiment/models.py",
        "adapt_library_result",
        "object",
        TORCH_REASON,
        "tests/experiment/test_models.py::"
        "test_library_adapter_rejects_an_unexpected_implementation",
    ),
    Exemption(
        "escape-hatch",
        "src/pace/experiment/models.py",
        "adapt_library_result",
        "cast(LibraryType, value)",
        TORCH_REASON,
        "tests/experiment/test_models.py::"
        "test_library_adapter_rejects_an_unexpected_implementation",
    ),
    Exemption(
        "escape-hatch",
        "src/pace/experiment/models.py",
        "",
        "import cast",
        TORCH_REASON,
        "tests/experiment/test_models.py::"
        "test_library_adapter_rejects_an_unexpected_implementation",
    ),
)


def _matches(exemption: Exemption, finding: Finding) -> bool:
    return (
        exemption.rule == finding.rule
        and finding.location.startswith(f"{exemption.path}:")
        and exemption.symbol == finding.symbol
        and finding.detail.startswith(exemption.detail)
    )


def apply_exemptions(
    findings: list[Finding], exemptions: tuple[Exemption, ...] = EXEMPTIONS
) -> tuple[list[Finding], list[Exemption]]:
    used: set[Exemption] = set()
    remaining: list[Finding] = []
    for finding in findings:
        matching = [item for item in exemptions if _matches(item, finding)]
        if matching:
            used.update(matching)
        else:
            remaining.append(finding)
    return remaining, [item for item in exemptions if item not in used]


def regression_test_exists(exemption: Exemption) -> bool:
    path_text, _, function = exemption.regression_test.partition("::")
    path = REPOSITORY_ROOT / path_text
    if not path.is_file():
        return False
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return any(
        isinstance(node, ast.FunctionDef) and node.name == function for node in ast.walk(tree)
    )
