import shutil
from collections.abc import Iterator
from pathlib import Path

import pytest

import pace
from pace.experiment import implementation
from pace.experiment.implementation import digest_implementation
from pace.experiment.scientific_types import scientific_types_payload
from pace.types import ArtifactPayload, ImplementationScope, ScientificDigest

PACKAGE_ROOT = Path(pace.__file__).resolve().parent
CAMPAIGN = ImplementationScope.CAMPAIGN
REPORT = ImplementationScope.REPORT


@pytest.fixture
def package(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    destination = tmp_path / "pace"
    shutil.copytree(PACKAGE_ROOT, destination, ignore=shutil.ignore_patterns("__pycache__"))
    monkeypatch.setattr(
        implementation, "__file__", str(destination / "experiment" / "implementation.py")
    )
    yield destination
    implementation._implementation_snapshot.cache_clear()  # pyright: ignore[reportPrivateUsage]


def _digest(scope: ImplementationScope) -> ScientificDigest:
    implementation._implementation_snapshot.cache_clear()  # pyright: ignore[reportPrivateUsage]
    return digest_implementation(scope)


def _edit(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    assert text.count(old) == 1, old
    path.write_text(text.replace(old, new), encoding="utf-8")


@pytest.mark.parametrize(
    ("relative", "old", "new"),
    [
        (
            "types.py",
            'SHARE_CONSERVING = "SHARE_CONSERVING"',
            'SHARE_CONSERVING = "SHARE_CONSERVED"',
        ),
        (
            "types.py",
            'SHARE_CONSERVING = "SHARE_CONSERVING"',
            'SHARE_CONSERVING = "SHARE_CONSERVING"\n    EXTRA_POLICY = "EXTRA_POLICY"',
        ),
        (
            "types.py",
            "CURRENT_ARTIFACT_SCHEMA: ArtifactSchemaVersion = 18",
            "CURRENT_ARTIFACT_SCHEMA: ArtifactSchemaVersion = 19",
        ),
        (
            "experiment/runner.py",
            "class MatrixRunResult:",
            "class MatrixRunResult:\n    pass_through = 1",
        ),
        ("experiment/sources.py", "def stamp_of(", "def stamp_of_changed("),
        ("config.py", "class AttackMinimumOverride(", "class AttackMinimumOverrideChanged("),
        ("experiment/inputs.py", "class PeerExpert:", "class PeerExpertChanged:"),
    ],
)
def test_scientific_edit_changes_digest(package: Path, relative: str, old: str, new: str) -> None:
    campaign = _digest(CAMPAIGN)
    _edit(package / relative, old, new)
    assert _digest(CAMPAIGN) != campaign


def test_new_module_changes_digest(package: Path) -> None:
    campaign = _digest(CAMPAIGN)
    (package / "experiment" / "added.py").write_text("VALUE = 1\n", encoding="utf-8")
    assert _digest(CAMPAIGN) != campaign


@pytest.mark.parametrize(
    ("relative", "old", "new"),
    [
        ("types.py", 'SATISFIED = "satisfied"', 'SATISFIED = "criterion_satisfied"'),
        ("types.py", 'DisplayText("Cross-fitted")', 'DisplayText("Cross-fitted calibration")'),
        (
            "reporting/figures.py",
            "logger = structlog.get_logger()",
            "logger = structlog.get_logger()\nNOTE = 1",
        ),
    ],
)
def test_presentation_edit_keeps_campaign_digest(
    package: Path, relative: str, old: str, new: str
) -> None:
    campaign, report = _digest(CAMPAIGN), _digest(REPORT)
    _edit(package / relative, old, new)
    assert _digest(CAMPAIGN) == campaign
    assert _digest(REPORT) != report


TYPES = b"""
class Kept(Enum):
    A = "a"

    @property
    def label(self):
        return Text("kept")

    @property
    def used(self):
        return Helper.B

    @property
    def unused(self):
        return Text("unused")


class Helper(Enum):
    B = "b"


class Orphan(Enum):
    C = "c"


class OnlyForOrphan(Enum):
    D = "d"

    @property
    def uses(self):
        return Orphan.C


class Annotated(Enum):
    E = "e"

SCALE = Hidden
"""


def _payload(types_source: bytes, *scope: bytes) -> bytes:
    return scientific_types_payload(
        ArtifactPayload(types_source), tuple(ArtifactPayload(s) for s in scope)
    )


def test_payload_keeps_only_reachable_types() -> None:
    scientific = b"x = Kept.A.used\ndef f(v: 'Annotated') -> None: ...\n"
    payload = _payload(TYPES, scientific)
    for kept in (b"Kept", b"Helper", b"Annotated", b"used"):
        assert b"name='" + kept + b"'" in payload
    for dropped in (b"Orphan", b"OnlyForOrphan", b"label", b"unused"):
        assert b"name='" + dropped + b"'" not in payload


DEPENDENCIES = b"""
class Probe(Enum):
    A = "a"

    @property
    def label(self):
        return Text("probe")

    @property
    def unused(self):
        return Text("unused")

    @cached_property
    def cached(self):
        return Text("cached")

    @computed_field
    @property
    def computed(self):
        return Text("computed")


class Wrapped(Enum):
    W = "w"


class Registered(Enum):
    R = "r"


class Defaulted(Enum):
    D = "d"


class Base:
    @property
    def inherited(self):
        return Text("inherited")


class Derived(Base):
    pass


class Unreferenced(Enum):
    U = "u"


Alias = Annotated[Wrapped, "alias"]
LIMIT = 3
"""

REACHED = {
    "class pattern keyword": (
        b"match value:\n    case Probe(label=text):\n        pass\n",
        b'Text("probe")',
        b'Text("edited")',
    ),
    "getattr literal": (
        b'member = Probe.A\ntext = getattr(member, "label")\n',
        b'Text("probe")',
        b'Text("edited")',
    ),
    "dotted path string": (
        b'member = Probe.A\nread = attrgetter("holder.label")\n',
        b'Text("probe")',
        b'Text("edited")',
    ),
    "forward reference": (
        b'Handle = TypeVar("Handle", bound="Registered")\n',
        b'R = "r"',
        b'R = "changed"',
    ),
    "string annotation": (
        b'def f(value: "dict[str, Registered]") -> None: ...\n',
        b'R = "r"',
        b'R = "changed"',
    ),
    "default argument": (
        b"def f(value=Defaulted.D): ...\n",
        b'D = "d"',
        b'D = "changed"',
    ),
    "decorator": (
        b"@Registered.register\ndef f(): ...\n",
        b'R = "r"',
        b'R = "changed"',
    ),
    "alias in annotation": (
        b"def f(value: Alias) -> None: ...\n",
        b'W = "w"',
        b'W = "changed"',
    ),
    "inherited property": (
        b"text = Derived().inherited\n",
        b'Text("inherited")',
        b'Text("edited")',
    ),
    "iteration": (
        b"members = list(Registered)\n",
        b'R = "r"',
        b'R = "changed"',
    ),
    "referenced member": (
        b"member = Probe.A\n",
        b'A = "a"',
        b'A = "z"',
    ),
    "referenced property": (
        b"text = Probe.A.label\n",
        b'Text("probe")',
        b'Text("edited")',
    ),
    "cached property": (
        b"member = Probe.A\n",
        b'Text("cached")',
        b'Text("edited")',
    ),
    "computed field": (
        b"member = Probe.A\n",
        b'Text("computed")',
        b'Text("edited")',
    ),
    "module constant": (
        b"member = Probe.A\n",
        b"LIMIT = 3",
        b"LIMIT = 4",
    ),
}


@pytest.mark.parametrize(("scope", "old", "new"), list(REACHED.values()), ids=list(REACHED))
def test_reached_dependency_changes_payload(scope: bytes, old: bytes, new: bytes) -> None:
    assert DEPENDENCIES.count(old) == 1
    assert _payload(DEPENDENCIES.replace(old, new), scope) != _payload(DEPENDENCIES, scope)


@pytest.mark.parametrize(
    ("old", "new"),
    [
        (b'Text("unused")', b'Text("edited")'),
        (b'U = "u"', b'U = "changed"'),
        (b'A = "a"\n', b"A = 'a'\n"),
        (b"LIMIT = 3", b"LIMIT = (3)"),
        (b"class Unreferenced(Enum):\n", b"# a comment\n\n\n\nclass Unreferenced(Enum):  # note\n"),
    ],
    ids=["unused property", "unused class", "quote style", "parentheses", "comments"],
)
def test_presentation_edit_keeps_payload(old: bytes, new: bytes) -> None:
    scope = b"member = Probe.A\ntext = Probe.A.label\n"
    assert DEPENDENCIES.count(old) == 1
    assert _payload(DEPENDENCIES.replace(old, new), scope) == _payload(DEPENDENCIES, scope)
