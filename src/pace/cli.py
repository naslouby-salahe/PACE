import multiprocessing
import os
import time
from collections.abc import Callable, Iterable
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import structlog
import typer
from structlog.stdlib import BoundLogger

from pace.config import PowerSettings, ResolvedConfiguration, load_configuration
from pace.experiment.artifacts import execution_log_path
from pace.experiment.logging_setup import configure_logging
from pace.experiment.preflight import preflight_registered_dataset
from pace.experiment.runner import run_registered_dataset_matrix_workflow
from pace.experiment.smoke import run_synthetic_workflow
from pace.experiment.status import inspect_matrix_status, run_doctor
from pace.reporting.acceptance import acceptance_line
from pace.reporting.metrics import summarize_budget_power
from pace.reporting.service import (
    ResultFiles,
    report_acceptance,
    write_results,
    write_stratum_figures,
)
from pace.reporting.studies import write_calibration_study, write_compute_cost, write_data_roles
from pace.types import (
    AnalysisPower,
    CampaignCompletionState,
    CommandName,
    CommandTarget,
    ConfigurationFile,
    DisplayText,
    EnvironmentVariable,
    ExitStatus,
    LocalTrainingRowCount,
    LogEvent,
    OptionalTargetArgument,
    OverwriteOption,
    PaceError,
    PopulationOption,
    PopulationScope,
    ProgramName,
    RepeatsOption,
    ReplacementPolicy,
    SmokeOption,
    SourceInventoryStatus,
    StratumBudgetPlan,
    StudyStratum,
    TargetArgument,
    TargetBudgetSupport,
    WorkerCount,
    WorkersOption,
)

app = typer.Typer(add_completion=False, no_args_is_help=False, help="PACE experiment CLI")


def _status_detail(state: CampaignCompletionState) -> DisplayText:
    match state:
        case CampaignCompletionState.COMPLETE:
            return DisplayText("")
        case CampaignCompletionState.MISSING:
            return DisplayText("; artifact missing")
        case CampaignCompletionState.INVALID:
            return DisplayText("; artifact verification failed")
        case CampaignCompletionState.STALE:
            return DisplayText("; provenance is stale")


def _policy(overwrite: OverwriteOption) -> ReplacementPolicy:
    return ReplacementPolicy.REPLACE_REQUESTED if overwrite else ReplacementPolicy.PRESERVE_VALID


def _configuration_path() -> Path:
    return Path(os.environ.get(EnvironmentVariable.CONFIGURATION, ConfigurationFile.DEFAULT))


def _load(config: Path) -> ResolvedConfiguration:
    repository_root = Path.cwd()
    try:
        return load_configuration((repository_root / config).resolve(), repository_root)
    except PaceError as error:
        raise typer.BadParameter(f"{error}") from error


def _command_logger(
    command: CommandName, configuration: ResolvedConfiguration, config: Path
) -> BoundLogger:
    configure_logging(
        execution_log_path(configuration.runtime.output_root, command),
        configuration.runtime.log_level,
    )
    logger = structlog.get_logger().bind(
        command=command, configuration_digest=configuration.scientific_digest
    )
    logger.info(
        LogEvent.CONFIGURATION_LOADED,
        configuration_path=config.as_posix(),
        provenance_active=configuration.runtime.provenance_active,
    )
    return logger


def _execute(
    command: CommandName,
    operation: Callable[[ResolvedConfiguration, BoundLogger], ExitStatus],
) -> ExitStatus:
    config = _configuration_path()
    configuration = _load(config)
    logger = _command_logger(command, configuration, config)
    if not configuration.runtime.provenance_active:
        typer.echo(
            "Warning: provenance verification is disabled (runtime.provenance_active: false); "
            "do not report results produced or reused in this mode",
            err=True,
        )
    logger.info(LogEvent.COMMAND_STARTED)
    started = time.perf_counter()
    try:
        exit_status = operation(configuration, logger)
    except PaceError as error:
        logger.error(
            LogEvent.COMMAND_FAILED,
            failure_type=type(error).__name__,
            duration_seconds=time.perf_counter() - started,
        )
        raise typer.BadParameter(f"{error}") from error
    logger.info(
        LogEvent.COMMAND_COMPLETED,
        exit_status=exit_status.name,
        duration_seconds=time.perf_counter() - started,
    )
    return exit_status


def _target_support_line(
    target: TargetBudgetSupport, budgets: tuple[LocalTrainingRowCount, ...]
) -> DisplayText:
    support = ", ".join(
        f">={budget}: {'yes' if target.is_supported_at(budget) else 'no'}" for budget in budgets
    )
    issues = "".join(f" [{issue.name}]" for issue in target.eligibility.issues)
    return DisplayText(
        f"  target {target.client.name}: benign {target.benign_rows}, "
        f"train {target.training_rows}, available {target.training_rows}, "
        f"calibration {target.calibration_rows}, attack types {target.attack_type_count}, "
        f"{support}{issues}"
    )


def _barred(parts: Iterable[DisplayText]) -> DisplayText:
    return DisplayText(" | ".join(parts))


def _plan_summary(plan: StratumBudgetPlan) -> tuple[DisplayText, ...]:
    counts = _barred(
        DisplayText(f">={budget}: {plan.supported_count(budget)}")
        for budget in plan.requested_budgets
    )
    lines = [DisplayText(f"  summary targets_total={len(plan.eligible)} | {counts}")]
    lines.extend(
        DisplayText(
            f"  analysis {analysis.kind.name.lower()}: budgets "
            f"{'/'.join(f'{budget}' for budget in analysis.budgets)} on "
            f"{len(analysis.targets)} targets"
        )
        for analysis in plan.analyses
    )
    if plan.unavailable_budgets:
        lines.append(
            DisplayText(
                "  unavailable for full-stratum inference: "
                + "/".join(f"{budget}" for budget in plan.unavailable_budgets)
            )
        )
    if plan.unobserved_budgets:
        lines.append(
            DisplayText(
                "  no valid observations: "
                + "/".join(f"{budget}" for budget in plan.unobserved_budgets)
            )
        )
    return tuple(lines)


def _power_lines(
    power: tuple[AnalysisPower, ...], settings: PowerSettings
) -> tuple[DisplayText, ...]:
    lines = [
        DisplayText(
            "Outcome-independent power for the pooled paired mean difference "
            f"(meaningful effect {settings.minimum_meaningful_effect:.3f}, "
            f"alpha {settings.significance_level:.2f}, target power {settings.power:.2f})"
        )
    ]
    for analysis in power:
        lines.append(DisplayText(f"  {analysis.kind.title}: {analysis.target_count} targets"))
        lines.extend(
            DisplayText(
                f"    assumed SD of target-level differences {point.difference_sd:.2f}: "
                f"minimum detectable effect {point.minimum_detectable_effect:.4f}, "
                f"power at the meaningful effect {point.power_at_meaningful_effect:.3f}"
            )
            for point in analysis.points
        )
    return tuple(lines)


def doctor(smoke: SmokeOption = False) -> ExitStatus:
    def operation(configuration: ResolvedConfiguration, _logger: BoundLogger) -> ExitStatus:
        report = run_doctor(configuration)
        for source in report.sources:
            typer.echo(
                f"{source.dataset.name}: {source.availability.name.lower()} "
                f"({source.file_count} files, {source.byte_count} bytes)"
            )
        typer.echo(
            SourceInventoryStatus.READY if report.is_complete else SourceInventoryStatus.INCOMPLETE
        )
        if smoke:
            result = run_synthetic_workflow(configuration, ReplacementPolicy.PRESERVE_VALID)
            typer.echo(
                f"Synthetic evidence {result.reuse_state.name.lower()}: {result.artifact_path.path}"
            )
        return ExitStatus.SUCCESS if report.is_complete else ExitStatus.FAILURE

    return _execute(CommandName.DOCTOR, operation)


def status(target: OptionalTargetArgument = CommandTarget.ALL) -> ExitStatus:
    def operation(configuration: ResolvedConfiguration, _logger: BoundLogger) -> ExitStatus:
        incomplete = 0
        for stratum in target.strata:
            cells = inspect_matrix_status(stratum, configuration)
            complete = sum(cell.state is CampaignCompletionState.COMPLETE for cell in cells)
            incomplete += len(cells) - complete
            typer.echo(f"{stratum.display_label}: {complete} of {len(cells)} matrix cells complete")
            for cell in cells:
                if cell.state is not CampaignCompletionState.COMPLETE:
                    typer.echo(f"  {cell.artifact_path.path}{_status_detail(cell.state)}")
        return ExitStatus.SUCCESS if incomplete == 0 else ExitStatus.FAILURE

    return _execute(CommandName.STATUS, operation)


def preflight(target: OptionalTargetArgument = CommandTarget.ALL) -> ExitStatus:
    def operation(configuration: ResolvedConfiguration, _logger: BoundLogger) -> ExitStatus:
        plans: list[StratumBudgetPlan] = []
        for stratum in target.strata:
            plan = preflight_registered_dataset(stratum.dataset, configuration, stratum)
            plans.append(plan)
            typer.echo(f"{plan.stratum.display_label} budget feasibility")
            for candidate in plan.candidates:
                typer.echo(_target_support_line(candidate, plan.requested_budgets))
            for line in _plan_summary(plan):
                typer.echo(line)
        protocol = configuration.protocol_matrix
        for line in _power_lines(summarize_budget_power(tuple(plans), protocol), protocol.power):
            typer.echo(line)
        return ExitStatus.SUCCESS

    return _execute(CommandName.PREFLIGHT, operation)


def _run_stratum(
    stratum: StudyStratum,
    configuration: ResolvedConfiguration,
    policy: ReplacementPolicy,
    logger: BoundLogger,
) -> None:
    matrix_results = run_registered_dataset_matrix_workflow(
        stratum.dataset, configuration, stratum, policy
    )
    for matrix_result in matrix_results:
        variant, result = matrix_result.variant, matrix_result.result
        logger.info(
            LogEvent.MATRIX_VARIANT_COMPLETED,
            analysis=variant.kind.name,
            alpha=variant.alpha.name,
            local_training_rows=variant.local_training_rows,
            target_count=result.target_count,
        )
        typer.echo(
            f"{stratum.display_label} campaign {variant.kind.name.lower()} "
            f"alpha={variant.alpha.fraction:.3f}, "
            f"local rows={variant.local_training_rows}, "
            f"{result.reuse_state.name.lower()}: "
            f"{result.target_count} targets at {result.artifact_path.path}"
        )
    figures = write_stratum_figures(stratum, configuration, policy)
    typer.echo(f"{stratum.display_label} figures: {len(figures)} files written")


def _limit_numeric_threads(threads: WorkerCount) -> None:
    for variable in (
        EnvironmentVariable.OMP_THREADS,
        EnvironmentVariable.MKL_THREADS,
        EnvironmentVariable.OPENBLAS_THREADS,
    ):
        os.environ[variable] = f"{threads}"


def _stratum_job(stratum: StudyStratum, policy: ReplacementPolicy) -> StudyStratum:
    configuration = _load(_configuration_path())
    logger = _command_logger(CommandName.RUN, configuration, _configuration_path())
    logger.info(LogEvent.STRATUM_WORKER_STARTED, study_stratum=stratum.name)
    _run_stratum(stratum, configuration, policy, logger)
    return stratum


def _run_in_parallel(
    strata: tuple[StudyStratum, ...], policy: ReplacementPolicy, workers: WorkerCount
) -> None:
    pool_size = min(workers, len(strata))
    _limit_numeric_threads(max(1, -(-(os.cpu_count() or 1) // pool_size)))
    with ProcessPoolExecutor(
        max_workers=pool_size, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        for finished in pool.map(_stratum_job, strata, [policy] * len(strata)):
            typer.echo(f"{finished.display_label} worker finished")


def run(
    target: TargetArgument, overwrite: OverwriteOption = False, workers: WorkersOption = 1
) -> ExitStatus:
    def operation(configuration: ResolvedConfiguration, logger: BoundLogger) -> ExitStatus:
        if workers > 1 and len(target.strata) > 1:
            _run_in_parallel(target.strata, _policy(overwrite), workers)
            return ExitStatus.SUCCESS
        for stratum in target.strata:
            _run_stratum(stratum, configuration, _policy(overwrite), logger)
        return ExitStatus.SUCCESS

    return _execute(CommandName.RUN, operation)


def _results_line(files: ResultFiles) -> DisplayText:
    return DisplayText(
        f"Results written: {len(files.tables)} table files, {len(files.figures)} figure files"
    )


def report(population: PopulationOption = PopulationScope.PRIMARY) -> ExitStatus:
    def operation(configuration: ResolvedConfiguration, _logger: BoundLogger) -> ExitStatus:
        files = write_results(configuration, ReplacementPolicy.PRESERVE_VALID, population)
        typer.echo(_results_line(files))
        typer.echo("Calibration sensitivity is refreshed separately by pace calibration-study.")
        return ExitStatus.SUCCESS

    return _execute(CommandName.REPORT, operation)


def benchmark(
    target: OptionalTargetArgument = CommandTarget.ALL, repeats: RepeatsOption = 3
) -> ExitStatus:
    def operation(configuration: ResolvedConfiguration, _logger: BoundLogger) -> ExitStatus:
        output = write_compute_cost(configuration, target.strata, repeats)
        typer.echo(f"Compute cost: {output.records} measurements at {output.path}")
        return ExitStatus.SUCCESS

    return _execute(CommandName.BENCHMARK, operation)


def audit(target: OptionalTargetArgument = CommandTarget.ALL) -> ExitStatus:
    def operation(configuration: ResolvedConfiguration, _logger: BoundLogger) -> ExitStatus:
        output, violations = write_data_roles(configuration, target.strata)
        typer.echo(f"Data roles: {output.records} clients audited at {output.path}")
        for item in violations:
            typer.echo(f"  role violation: {item.target.name}")
        return ExitStatus.FAILURE if violations else ExitStatus.SUCCESS

    return _execute(CommandName.AUDIT, operation)


def calibration_study(workers: WorkersOption = 1) -> ExitStatus:
    def operation(configuration: ResolvedConfiguration, _logger: BoundLogger) -> ExitStatus:
        output = write_calibration_study(configuration, workers)
        typer.echo(f"Calibration study: {output.records} records at {output.path}")
        return ExitStatus.SUCCESS

    return _execute(CommandName.CALIBRATION_STUDY, operation)


def accept(population: PopulationOption = PopulationScope.PRIMARY) -> ExitStatus:
    def operation(configuration: ResolvedConfiguration, _logger: BoundLogger) -> ExitStatus:
        acceptance = report_acceptance(configuration, ReplacementPolicy.PRESERVE_VALID, population)
        typer.echo(
            f"Official protocol; population: {population}; "
            "comparator: same-calibration mean-ensemble local branch; "
            "detection gain at comparable realised FPR"
        )
        if population is PopulationScope.CONFIRMATION:
            typer.echo("Confirmation cohort: detection may be saturated by ceiling effects.")
        for cell in acceptance.cells:
            typer.echo(acceptance_line(cell))
        heading = (
            "OFFICIAL SCIENTIFIC ACCEPTANCE (primary)"
            if population is PopulationScope.PRIMARY
            else "CONFIRMATION COHORT ASSESSMENT"
        )
        typer.echo(
            f"{heading} against "
            f"{acceptance.comparator.name}: "
            f"{'ACCEPTED' if acceptance.is_accepted else 'REJECTED'}"
        )
        return ExitStatus.SUCCESS if acceptance.is_accepted else ExitStatus.FAILURE

    return _execute(CommandName.ACCEPT, operation)


for _command in (
    doctor,
    status,
    preflight,
    run,
    report,
    accept,
    benchmark,
    audit,
    calibration_study,
):
    app.command()(_command)


def main() -> ExitStatus:
    try:
        exit_status = typer.main.get_command(app).main(
            prog_name=ProgramName.PACE, standalone_mode=False
        )
    except typer.TyperException as error:
        typer.echo(f"Error: {error}", err=True)
        raise SystemExit(ExitStatus.USAGE_ERROR) from error
    return ExitStatus(exit_status) if exit_status else ExitStatus.SUCCESS
