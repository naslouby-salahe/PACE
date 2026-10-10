import os
import time
from collections.abc import Callable, Iterable
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
from pace.reporting.metrics import summarize_budget_power
from pace.reporting.service import (
    ResultFiles,
    report_acceptance,
    write_results,
    write_stratum_figures,
)
from pace.types import (
    AcceptanceCell,
    AcceptanceDesign,
    AcceptanceOutcomeWord,
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
    ProgramName,
    RateConfidenceInterval,
    RateDifference,
    ReplacementPolicy,
    SmokeOption,
    SourceInventoryStatus,
    StratumBudgetPlan,
    StudyStratum,
    TargetArgument,
    TargetBudgetSupport,
    UnavailableReading,
)

app = typer.Typer(add_completion=False, no_args_is_help=False, help="PACE experiment CLI")


def _interval(interval: RateConfidenceInterval) -> DisplayText:
    return DisplayText(
        f"{interval.confidence_level:.0%} CI "
        f"[{interval.lower_bound:.4f}, {interval.upper_bound:.4f}]"
    )


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


def _acceptance_line(cell: AcceptanceCell) -> DisplayText:
    def interval(gain: RateDifference | None, bounds: RateConfidenceInterval | None) -> DisplayText:
        if gain is None or bounds is None:
            return DisplayText(UnavailableReading.NOT_AVAILABLE)
        return DisplayText(f"{gain:+.4f} {_interval(bounds)}")

    failed = _barred(
        DisplayText(verdict.criterion.name.lower())
        for verdict in cell.verdicts
        if not verdict.satisfied
    )
    outcome = AcceptanceOutcomeWord.ACCEPTED if cell.is_accepted else AcceptanceOutcomeWord.FAILED
    gating = f"{AcceptanceOutcomeWord.GATING} " if cell.is_gating else ""
    detail = "" if cell.is_accepted else f" [{failed}]"
    return DisplayText(
        f"{cell.kind.name.lower()} alpha={cell.alpha.fraction:.2f} rows={cell.local_training_rows} "
        f"targets={cell.target_count}: "
        f"nominal {interval(cell.nominal_gain, cell.nominal_interval)}, "
        f"matched {interval(cell.matched_gain, cell.matched_interval)}, "
        f"without {AcceptanceDesign.PROTOCOL.excluded_dataset.name.lower()} "
        f"{interval(cell.gain_without_excluded, cell.gain_without_excluded_interval)}, "
        f"harmed {cell.harmed_target_count}, severe {cell.severe_target_count}, "
        f"worst {cell.worst_target_difference:+.4f}, FPR {cell.false_alert_increase:+.4f}, "
        f"{gating}{outcome}{detail}"
    )


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


def run(target: TargetArgument, overwrite: OverwriteOption = False) -> ExitStatus:
    def operation(configuration: ResolvedConfiguration, logger: BoundLogger) -> ExitStatus:
        for stratum in target.strata:
            _run_stratum(stratum, configuration, _policy(overwrite), logger)
        return ExitStatus.SUCCESS

    return _execute(CommandName.RUN, operation)


def _results_line(files: ResultFiles) -> DisplayText:
    return DisplayText(
        f"Results written: {len(files.tables)} table files, {len(files.figures)} figure files"
    )


def report() -> ExitStatus:
    def operation(configuration: ResolvedConfiguration, _logger: BoundLogger) -> ExitStatus:
        files = write_results(configuration, ReplacementPolicy.PRESERVE_VALID)
        typer.echo(_results_line(files))
        return ExitStatus.SUCCESS

    return _execute(CommandName.REPORT, operation)


def accept() -> ExitStatus:
    def operation(configuration: ResolvedConfiguration, _logger: BoundLogger) -> ExitStatus:
        acceptance = report_acceptance(configuration, ReplacementPolicy.PRESERVE_VALID)
        for cell in acceptance.cells:
            typer.echo(_acceptance_line(cell))
        typer.echo(
            f"acceptance against {acceptance.comparator.name}: "
            f"{'ACCEPTED' if acceptance.is_accepted else 'REJECTED'}"
        )
        return ExitStatus.SUCCESS if acceptance.is_accepted else ExitStatus.FAILURE

    return _execute(CommandName.ACCEPT, operation)


for _command in (doctor, status, preflight, run, report, accept):
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
