from dataclasses import dataclass
from pathlib import Path

import structlog
from pydantic import ValidationError
from structlog.stdlib import BoundLogger

from pace.config import (
    ProtocolMatrixSettings,
    ResolvedConfiguration,
    SplitSettings,
)
from pace.datasets.cache import resolve_clients, validate_dataset_clients
from pace.datasets.registry import (
    dataset_source_files,
    load_dataset,
    resolve_study_stratum,
)
from pace.experiment.artifacts import (
    digest_source_files,
    execution_log_path,
    write_bytes_atomically,
)
from pace.experiment.logging_setup import configure_logging
from pace.experiment.splitting import assess_target_eligibility, plan_client_split
from pace.types import (
    AlphaLevel,
    ArtifactFileName,
    ArtifactPayload,
    ArtifactValidationError,
    BudgetAnalysis,
    BudgetAnalysisKind,
    BudgetPlanSummary,
    ClientName,
    CommandName,
    DatasetClientData,
    DatasetName,
    DatasetValidationError,
    FrozenStrictModel,
    LocalTrainingRowCount,
    LogEvent,
    MinimumPeerCount,
    ObservationCount,
    OutputDirectory,
    ScientificDigest,
    StratumBudgetPlan,
    StudyStratum,
    TargetBudgetSupport,
    TargetCohortRole,
    TargetEligibilityIssue,
    TargetTrainingRows,
    canonical_json,
)

logger = structlog.get_logger()


@dataclass(frozen=True, slots=True)
class MatrixVariant:
    kind: BudgetAnalysisKind
    alpha: AlphaLevel
    local_training_rows: LocalTrainingRowCount
    targets: tuple[ClientName, ...]


class PersistedTargetSupport(FrozenStrictModel, frozen=True):
    client: ClientName
    benign_rows: ObservationCount
    training_rows: ObservationCount
    calibration_rows: ObservationCount
    attack_type_count: ObservationCount
    eligibility_issues: tuple[TargetEligibilityIssue, ...]


class PersistedBudgetAnalysis(FrozenStrictModel, frozen=True):
    kind: BudgetAnalysisKind
    budgets: tuple[LocalTrainingRowCount, ...]
    targets: tuple[ClientName, ...]


class PersistedBudgetPlan(FrozenStrictModel, frozen=True):
    stratum: StudyStratum
    input_digest: ScientificDigest
    requested_budgets: tuple[LocalTrainingRowCount, ...]
    candidates: tuple[PersistedTargetSupport, ...]
    eligible_targets: ObservationCount
    analyses: tuple[PersistedBudgetAnalysis, ...]
    unavailable_budgets: tuple[LocalTrainingRowCount, ...]
    unobserved_budgets: tuple[LocalTrainingRowCount, ...]


def assess_target_support(
    clients: tuple[DatasetClientData, ...],
    splitting: SplitSettings,
    minimum_peer_count: MinimumPeerCount,
) -> tuple[TargetBudgetSupport, ...]:
    validate_dataset_clients(clients)
    pool = tuple(
        client
        for client in sorted(clients, key=lambda item: item.client.name)
        if client.target_cohort_role is not TargetCohortRole.OUTSIDE_PEER_POOL
    )
    plans = tuple(
        plan_client_split(
            client.client,
            client.study_stratum,
            client.benign_source_order,
            client.attack_captures,
            splitting,
        )
        for client in pool
    )
    expert_clients = {plan.client for plan in plans if plan.attack_splits}
    support: list[TargetBudgetSupport] = []
    for client, plan in zip(pool, plans, strict=True):
        if client.target_cohort_role is TargetCohortRole.PEER_ONLY:
            continue
        peer_count = len(expert_clients - {plan.client})
        support.append(
            TargetBudgetSupport(
                client=client.client,
                benign_rows=client.benign_features.shape[0],
                training_rows=len(plan.benign_roles.training),
                calibration_rows=len(plan.benign_roles.calibration),
                attack_type_count=len(plan.attack_splits),
                eligibility=assess_target_eligibility(
                    len(plan.attack_splits),
                    len(plan.benign_roles.calibration),
                    peer_count,
                    splitting.minimum_calibration_rows,
                    minimum_peer_count,
                ),
            )
        )
    return tuple(support)


def plan_budget_analyses(
    stratum: StudyStratum,
    candidates: tuple[TargetBudgetSupport, ...],
    protocol: ProtocolMatrixSettings,
) -> StratumBudgetPlan:
    budgets = protocol.local_training_row_counts
    eligible = tuple(target for target in candidates if target.eligibility.is_eligible)
    if not eligible:
        raise DatasetValidationError("No target satisfies the frozen eligibility rules")
    analyses = tuple(
        BudgetAnalysis(spec.kind, spec.budgets, cohort)
        for spec in protocol.analyses
        if (
            cohort := tuple(
                target.client.name
                for target in eligible
                if target.is_supported_at(spec.budgets[-1])
            )
        )
    )
    if not analyses:
        raise DatasetValidationError("No frozen analysis has a supported target cohort")
    supported = {
        budget: sum(target.is_supported_at(budget) for target in eligible) for budget in budgets
    }
    return StratumBudgetPlan(
        stratum,
        budgets,
        candidates,
        analyses,
        tuple(budget for budget in budgets if supported[budget] != len(eligible)),
        tuple(budget for budget in budgets if supported[budget] == 0),
    )


def plan_stratum_budgets(
    clients: tuple[DatasetClientData, ...], configuration: ResolvedConfiguration
) -> StratumBudgetPlan:
    candidates = assess_target_support(
        clients, configuration.splitting, configuration.scientific.minimum_peer_count
    )
    return plan_budget_analyses(clients[0].study_stratum, candidates, configuration.protocol_matrix)


def matrix_variants(
    analyses: tuple[BudgetAnalysis, ...], alpha_levels: tuple[AlphaLevel, ...]
) -> tuple[MatrixVariant, ...]:
    variants: dict[
        tuple[AlphaLevel, LocalTrainingRowCount, tuple[ClientName, ...]], MatrixVariant
    ] = {}
    for analysis in analyses:
        for alpha in alpha_levels:
            for budget in analysis.budgets:
                variants.setdefault(
                    (alpha, budget, analysis.targets),
                    MatrixVariant(analysis.kind, alpha, budget, analysis.targets),
                )
    return tuple(variants.values())


def budget_plan_path(output_root: Path, stratum: StudyStratum) -> Path:
    return (
        output_root
        / OutputDirectory.PREFLIGHT
        / stratum.name.lower()
        / ArtifactFileName.BUDGET_PLAN
    )


def _persisted(plan: StratumBudgetPlan, input_digest: ScientificDigest) -> PersistedBudgetPlan:
    return PersistedBudgetPlan(
        stratum=plan.stratum,
        input_digest=input_digest,
        requested_budgets=plan.requested_budgets,
        candidates=tuple(
            PersistedTargetSupport(
                client=target.client.name,
                benign_rows=target.benign_rows,
                training_rows=target.training_rows,
                calibration_rows=target.calibration_rows,
                attack_type_count=target.attack_type_count,
                eligibility_issues=target.eligibility.issues,
            )
            for target in plan.candidates
        ),
        eligible_targets=len(plan.eligible),
        analyses=tuple(
            PersistedBudgetAnalysis(kind=item.kind, budgets=item.budgets, targets=item.targets)
            for item in plan.analyses
        ),
        unavailable_budgets=plan.unavailable_budgets,
        unobserved_budgets=plan.unobserved_budgets,
    )


def persist_budget_plan(
    output_root: Path, plan: StratumBudgetPlan, input_digest: ScientificDigest
) -> Path:
    destination = budget_plan_path(output_root, plan.stratum)
    persisted = _persisted(plan, input_digest)

    def validate_written(path: Path) -> None:
        if _read_persisted(path) != persisted:
            raise ArtifactValidationError("Serialized budget plan changed during validation")

    write_bytes_atomically(
        destination, ArtifactPayload(canonical_json(persisted) + b"\n"), validate_written
    )
    logger.info(
        LogEvent.BUDGET_PLAN_PERSISTED,
        study_stratum=plan.stratum.name,
        artifact_path=destination.as_posix(),
    )
    return destination


def _read_persisted(path: Path) -> PersistedBudgetPlan:
    try:
        return PersistedBudgetPlan.model_validate_json(path.read_bytes(), strict=True)
    except (OSError, ValidationError, ValueError) as error:
        raise ArtifactValidationError(f"Budget plan is missing or invalid: {path}") from error


def read_budget_plan_summary(
    configuration: ResolvedConfiguration, stratum: StudyStratum, input_digest: ScientificDigest
) -> BudgetPlanSummary:
    persisted = _read_persisted(budget_plan_path(configuration.runtime.output_root, stratum))
    protocol = configuration.protocol_matrix
    if (
        persisted.stratum is not stratum
        or persisted.requested_budgets != protocol.local_training_row_counts
        or (configuration.runtime.provenance_active and persisted.input_digest != input_digest)
    ):
        raise ArtifactValidationError(
            f"Budget plan for {stratum.name} is stale; run pace preflight again"
        )
    return BudgetPlanSummary(
        tuple(BudgetAnalysis(item.kind, item.budgets, item.targets) for item in persisted.analyses),
        persisted.eligible_targets,
        tuple(
            TargetTrainingRows(item.client, item.training_rows, item.attack_type_count)
            for item in persisted.candidates
            if not item.eligibility_issues
        ),
    )


def preflight_registered_dataset(
    dataset: DatasetName,
    configuration: ResolvedConfiguration,
    stratum: StudyStratum | None = None,
) -> StratumBudgetPlan:
    selected = resolve_study_stratum(dataset, stratum)
    configure_logging(
        execution_log_path(configuration.runtime.output_root, CommandName.PREFLIGHT),
        configuration.runtime.log_level,
    )
    bound: BoundLogger = structlog.get_logger().bind(
        command=CommandName.PREFLIGHT, dataset=dataset.name, study_stratum=selected.name
    )
    raw_data_root = configuration.runtime.raw_data_root
    source_files = dataset_source_files(raw_data_root, dataset, selected)
    input_digest = digest_source_files(source_files, raw_data_root)
    clients = resolve_clients(
        lambda: (
            load_dataset(
                raw_data_root, configuration.runtime.lab_network, dataset, selected
            ).clients
        ),
        configuration,
        (dataset, selected),
        input_digest,
        bound,
    )
    plan = plan_stratum_budgets(clients, configuration)
    persist_budget_plan(configuration.runtime.output_root, plan, input_digest)
    bound.info(
        LogEvent.PREFLIGHT_COMPLETED,
        eligible_targets=len(plan.eligible),
        analysis_count=len(plan.analyses),
        unavailable_budgets=plan.unavailable_budgets,
        unobserved_budgets=plan.unobserved_budgets,
    )
    return plan
