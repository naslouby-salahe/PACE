import ast
import textwrap
from functools import cache

from tests.architecture.call_graph import (
    CallGraph,
    build_call_graph,
    build_call_graph_from_sources,
)
from tests.architecture.logging_rules import (
    FunctionFact,
    commands_without_lifecycle_logging,
    function_facts,
    logging_inside_hot_loops,
    unobserved_orchestrators,
)
from tests.architecture.rules import SourceUnit, read_units, rule_exception_handling, rule_print
from tests.architecture.source_index import (
    REPOSITORY_ROOT,
    SOURCE_ROOT,
    TYPE_MODULE,
    production_files,
)

COMMANDS = ("doctor", "status", "preflight", "run", "report", "accept")
LAYERS = ("pace.datasets", "pace.experiment", "pace.reporting", "pace.cli")
PROGRESS_EVENTS = {
    "MATRIX_VARIANT_STARTED",
    "CAMPAIGN_SEED_COMPLETED",
    "PEER_EXPERTS_TRAINED",
    "LOCAL_ENSEMBLE_FITTED",
    "ADAPTER_CLIENT_BUILT",
    "ADAPTER_SOURCE_READ",
    "CLIENT_CACHE_STORED",
    "ARTIFACT_PERSISTED",
}


@cache
def _production() -> tuple[tuple[SourceUnit, ...], CallGraph, dict[str, FunctionFact]]:
    units = read_units(production_files(), REPOSITORY_ROOT, TYPE_MODULE)
    graph = build_call_graph(SOURCE_ROOT, production_files(), ("pace.cli.main",))
    return units, graph, function_facts(units, graph)


def _synthetic(*modules: tuple[str, str]) -> dict[str, FunctionFact]:
    units = tuple(
        SourceUnit.parse(f"{name.replace('.', '/')}.py", textwrap.dedent(text))
        for name, text in modules
    )
    graph = build_call_graph_from_sources(
        {name: (f"{name}.py", textwrap.dedent(text)) for name, text in modules}
    )
    return function_facts(units, graph)


def test_every_cli_command_runs_the_lifecycle_logger() -> None:
    _, graph, facts = _production()
    commands = {name: graph.roots[name] for name in COMMANDS}
    assert not commands_without_lifecycle_logging(facts, commands)


def test_public_orchestration_and_io_functions_are_observable() -> None:
    _, _, facts = _production()
    assert not unobserved_orchestrators(facts, LAYERS)


def test_no_logging_happens_inside_comprehensions_polling_or_per_sample_loops() -> None:
    _, _, facts = _production()
    assert not logging_inside_hot_loops(facts)


def test_progress_level_events_are_emitted_by_long_running_workflows() -> None:
    units, _, _ = _production()
    emitted = {
        node.attr
        for unit in units
        if not unit.is_type_module
        for node in ast.walk(unit.tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "LogEvent"
    }
    assert emitted >= PROGRESS_EVENTS


LIFECYCLE_SOURCE = textwrap.dedent(
    """
    def execute(operation):
        logger.info(LogEvent.COMMAND_STARTED)
        try:
            operation()
        except PaceError:
            logger.error(LogEvent.COMMAND_FAILED)
            raise
        logger.info(LogEvent.COMMAND_COMPLETED)
    """
)


def test_command_without_lifecycle_logging_is_rejected() -> None:
    facts = _synthetic(
        ("app.cli", LIFECYCLE_SOURCE + "\ndef serve():\n    return work()\n"),
    )
    problems = commands_without_lifecycle_logging(facts, {"serve": "app.cli.serve"})
    assert any("does not run the lifecycle logger" in problem for problem in problems)


def test_command_using_the_lifecycle_logger_is_accepted() -> None:
    facts = _synthetic(
        ("app.cli", LIFECYCLE_SOURCE + "\ndef serve():\n    return execute(work)\n"),
    )
    assert not commands_without_lifecycle_logging(facts, {"serve": "app.cli.serve"})


def test_missing_lifecycle_logger_is_rejected() -> None:
    facts = _synthetic(("app.cli", "def serve():\n    return 1\n"))
    assert commands_without_lifecycle_logging(facts, {"serve": "app.cli.serve"})


def test_expensive_workflow_without_logging_is_rejected() -> None:
    source = """
        def load_a(): return 1
        def load_b(): return 2
        def load_c(): return 3
        def load_d(): return 4
        def load_e(): return 5
        def load_f(): return 6
        def load_g(): return 7
        def load_h(): return 8

        def run_everything():
            return (
                load_a(), load_b(), load_c(), load_d(), load_e(), load_f(), load_g(), load_h()
            )
        """
    facts = _synthetic(("app.experiment.workflow", source))
    assert unobserved_orchestrators(facts, ("app.experiment",))


def test_workflow_that_logs_or_delegates_to_a_logging_function_is_accepted() -> None:
    logging_source = "def run_everything():\n    logger.info('x')\n    return step_a() + step_b()\n"
    helpers = "def step_a(): return 1\ndef step_b(): return 2\n"
    facts = _synthetic(("app.experiment.workflow", helpers + logging_source))
    assert not unobserved_orchestrators(facts, ("app.experiment",))
    delegating = (
        "def logged_run():\n    logger.info('x')\n    return 1\n"
        + "\n".join(f"def part_{index}(): return {index}" for index in range(8))
        + "\ndef public_run():\n    logged_run()\n    return ("
        + ", ".join(f"part_{index}()" for index in range(8))
        + ")\n"
    )
    facts = _synthetic(("app.experiment.delegate", delegating))
    assert not unobserved_orchestrators(facts, ("app.experiment",))


def test_unlogged_public_file_io_boundary_is_rejected() -> None:
    facts = _synthetic(
        ("app.experiment.io", "def load(path):\n    return path.read_text()\n"),
    )
    assert unobserved_orchestrators(facts, ("app.experiment",))


def test_tiny_pure_numerical_helpers_are_not_forced_to_log() -> None:
    facts = _synthetic(
        ("app.experiment.math", "def mean_of(values):\n    return sum(values) / len(values)\n"),
    )
    assert not unobserved_orchestrators(facts, ("app.experiment",))


def test_private_helpers_and_lower_layers_are_not_forced_to_log() -> None:
    source = "def _read(path):\n    return path.read_text()\n"
    assert not unobserved_orchestrators(
        _synthetic(("app.experiment.private", source)), ("app.experiment",)
    )
    pure_io = "def load(path):\n    return path.read_text()\n"
    assert not unobserved_orchestrators(_synthetic(("app.config", pure_io)), ("app.experiment",))


def test_exception_paths_that_swallow_failures_are_rejected() -> None:
    unit = SourceUnit.parse(
        "src/pace/experiment/example.py",
        "def run():\n    try:\n        work()\n    except OSError:\n        pass\n",
    )
    assert rule_exception_handling(unit)
    logged = SourceUnit.parse(
        "src/pace/experiment/example.py",
        "def run():\n    try:\n        work()\n    except OSError:\n        logger.error('x')\n",
    )
    assert not rule_exception_handling(logged)


def test_print_in_orchestration_is_rejected_but_logging_is_not() -> None:
    printing = SourceUnit.parse(
        "src/pace/experiment/example.py", "def run():\n    print('progress')\n"
    )
    assert rule_print(printing)
    logging = SourceUnit.parse(
        "src/pace/experiment/example.py", "def run():\n    logger.info('progress')\n"
    )
    assert not rule_print(logging)


def test_logging_inside_comprehensions_polling_and_per_sample_loops_is_rejected() -> None:
    nested = """
        def run(rows):
            for outer in rows:
                for inner in outer:
                    logger.info('row')
        """
    comprehension = """
        def run(rows):
            return [logger.info('row') for row in rows]
        """
    polling = """
        def run(queue):
            while queue:
                logger.info('poll')
        """
    per_sample = """
        def run(rows):
            for row in rows:
                logger.info('row')
        """
    allowed = """
        def run(sources):
            for source in sources:
                logger.info('source')
        """
    assert logging_inside_hot_loops(_synthetic(("app.hot2", comprehension)))
    assert logging_inside_hot_loops(_synthetic(("app.hot3", polling)))
    assert logging_inside_hot_loops(_synthetic(("app.hot4", per_sample)))
    assert not logging_inside_hot_loops(_synthetic(("app.ok", allowed)))
    assert not logging_inside_hot_loops(
        _synthetic(("app.nested", nested.replace("rows", "sources")))
    )
