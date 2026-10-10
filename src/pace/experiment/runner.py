from collections.abc import Callable
from dataclasses import dataclass, field, replace
from math import isclose
from pathlib import Path

import numpy as np
import structlog
from structlog.stdlib import BoundLogger

from pace.config import (
    LocalBankSettings,
    LocalDetectorSettings,
    PeerTrainingSettings,
    ResolvedConfiguration,
    ScientificSettings,
    SplitSettings,
    configuration_for_protocol_variant,
)
from pace.core.decision import (
    apply_bonferroni_local_peer_rule,
    apply_max_fusion_rule,
)
from pace.datasets.cache import resolve_clients, validate_dataset_clients
from pace.datasets.registry import (
    dataset_source_files,
    load_dataset,
    resolve_study_stratum,
)
from pace.experiment.arms import ArmContext, evaluate_arms, protected_channel_decisions
from pace.experiment.artifacts import (
    CampaignOutcome,
    campaign_artifact_path,
    campaign_provenance,
    derive_learning_seed,
    digest_source_files,
    execution_log_path,
    require_unchanged_sources,
    reuse_or_write_artifact,
    reused_campaign,
    workflow_result,
)
from pace.experiment.inputs import (
    LocalEvidence,
    PeerEvidence,
    PeerExpert,
    PreparedDevice,
    TargetFrame,
    build_target_frame,
    compute_local_evidence,
    compute_peer_evidence,
    prepare_device,
    train_peer_mlp,
)
from pace.experiment.logging_setup import configure_logging, execute_with_failure_logging
from pace.experiment.preflight import (
    MatrixVariant,
    matrix_variants,
    persist_budget_plan,
    plan_stratum_budgets,
)
from pace.experiment.splitting import (
    assess_target_eligibility,
    digest_split_plan,
    fit_pooled_standardizer,
    plan_client_split,
)
from pace.types import (
    ArtifactPath,
    AttackTypePerformance,
    BudgetMultiplier,
    BudgetSweepPerformance,
    CampaignWorkflowResult,
    ChannelDecisions,
    ClientName,
    CommandName,
    DatasetClientData,
    DatasetClientId,
    DatasetName,
    DatasetValidationError,
    ExecutionDevice,
    LearningSeed,
    LogEvent,
    MaxFusionDecisions,
    ModelSeedProvenance,
    ReplacementPolicy,
    ScientificDigest,
    SeedDomain,
    StudyStratum,
    TargetCohortRole,
    TargetPerformance,
)

logger = structlog.get_logger()


@dataclass(frozen=True, slots=True)
class _LocalKey:
    client: DatasetClientId
    detectors: LocalDetectorSettings
    bank: LocalBankSettings
    device: ExecutionDevice


@dataclass(slots=True)
class _SeedModels:
    experts: tuple[PeerExpert, ...]
    frames: dict[DatasetClientId, TargetFrame]
    peer_evidence: dict[DatasetClientId, PeerEvidence] = field(
        default_factory=dict[DatasetClientId, PeerEvidence]
    )
    local_evidence: dict[_LocalKey, LocalEvidence] = field(
        default_factory=dict[_LocalKey, LocalEvidence]
    )


@dataclass(slots=True)
class MatrixTrainingCache:
    clients: tuple[DatasetClientData, ...]
    peer_training: PeerTrainingSettings
    splitting: SplitSettings
    prepared: tuple[PreparedDevice, ...] | None = None
    input_digest: ScientificDigest | None = None
    frames: dict[DatasetClientId, TargetFrame] = field(
        default_factory=dict[DatasetClientId, TargetFrame]
    )
    models_by_seed: dict[LearningSeed, _SeedModels] = field(
        default_factory=dict[LearningSeed, _SeedModels]
    )


def prepare_clients(
    devices: tuple[DatasetClientData, ...], configuration: ResolvedConfiguration
) -> tuple[PreparedDevice, ...]:
    validate_dataset_clients(devices)
    peer_pool = sorted(
        (
            device
            for device in devices
            if device.target_cohort_role is not TargetCohortRole.OUTSIDE_PEER_POOL
        ),
        key=lambda item: item.client.name,
    )
    if not peer_pool:
        raise DatasetValidationError("Dataset campaign has no clients in its frozen peer pool")
    ordered = tuple(
        replace(
            device,
            attack_captures=tuple(
                sorted(device.attack_captures, key=lambda capture: capture.attack_type.identifier)
            ),
        )
        for device in peer_pool
    )
    plans = tuple(
        plan_client_split(
            device.client,
            device.study_stratum,
            device.benign_source_order,
            device.attack_captures,
            configuration.splitting,
        )
        for device in ordered
    )
    standardizer = fit_pooled_standardizer(
        tuple(
            device.benign_features[plan.benign_roles.training]
            for device, plan in zip(ordered, plans, strict=True)
        )
    )
    return tuple(
        prepare_device(device, plan, standardizer)
        for device, plan in zip(ordered, plans, strict=True)
    )


def fit_peer_experts(
    prepared: tuple[PreparedDevice, ...],
    configuration: ResolvedConfiguration,
    seed: LearningSeed,
) -> tuple[PeerExpert, ...]:
    experts: list[PeerExpert] = []
    for device in prepared:
        if not device.development:
            continue
        peer_seed = derive_learning_seed(seed, device.client, SeedDomain.PEER_MODEL)
        model = train_peer_mlp(
            device.training,
            np.concatenate([features for _, features in device.development], axis=0),
            peer_seed,
            configuration.scientific.peer_training,
        )
        experts.append(PeerExpert(device.client, peer_seed, model))
    logger.info(LogEvent.PEER_EXPERTS_TRAINED, expert_count=len(experts), seed=seed)
    return tuple(experts)


def _sweep_performance(
    decisions: ChannelDecisions, frame: TargetFrame, multiplier: BudgetMultiplier
) -> BudgetSweepPerformance:
    return BudgetSweepPerformance(
        multiplier=multiplier,
        detection_rate=np.mean(frame.detection_rates(decisions.combined_alerts)).item(),
        false_alert_rate=frame.false_alert_rate(decisions.combined_alerts),
    )


def _target_inputs(
    target: PreparedDevice,
    peer_experts: tuple[PeerExpert, ...],
    models: _SeedModels,
    configuration: ResolvedConfiguration,
    seed: LearningSeed,
) -> tuple[TargetFrame, PeerEvidence, LocalEvidence]:
    client = target.client
    frame = models.frames.get(client)
    if frame is None:
        frame = models.frames[client] = build_target_frame(target)
    peers = models.peer_evidence.get(client)
    if peers is None:
        peers = models.peer_evidence[client] = compute_peer_evidence(peer_experts, frame)
    local_key = _LocalKey(
        client,
        configuration.scientific.local_detectors,
        configuration.scientific.local_bank,
        configuration.runtime.execution_device,
    )
    local = models.local_evidence.get(local_key)
    if local is None:
        local = models.local_evidence[local_key] = compute_local_evidence(
            target, frame, configuration, seed, peers
        )
    return frame, peers, local


@dataclass(frozen=True, slots=True)
class _CampaignDecisions:
    protected: ChannelDecisions
    max_fusion: MaxFusionDecisions
    bonferroni: ChannelDecisions
    budget_sweep: tuple[BudgetSweepPerformance, ...]


def _campaign_decisions(
    frame: TargetFrame,
    peers: PeerEvidence,
    local: LocalEvidence,
    scientific: ScientificSettings,
) -> _CampaignDecisions:
    local_budget = scientific.alpha.fraction * scientific.reserve_fraction
    peer_budget = scientific.alpha.fraction * (1.0 - scientific.reserve_fraction)

    def protected(multiplier: BudgetMultiplier) -> ChannelDecisions:
        return protected_channel_decisions(
            local,
            peers.calibration_evidence,
            peers.query_evidence,
            scientific,
            local_budget * multiplier,
            peer_budget * multiplier,
        )

    decisions = protected(1.0)
    budget_sweep = tuple(
        _sweep_performance(
            decisions if isclose(multiplier, 1.0) else protected(multiplier), frame, multiplier
        )
        for multiplier in scientific.budget_sweep_multipliers.for_alpha(scientific.alpha)
    )
    max_fusion = apply_max_fusion_rule(
        local.calibration,
        local.query,
        peers.max_fusion_calibration_evidence,
        peers.max_fusion_query_evidence,
        scientific.calibration_block_size,
        local_budget,
        peer_budget,
    )
    bonferroni = apply_bonferroni_local_peer_rule(
        local.calibration,
        local.query,
        peers.calibration_evidence,
        peers.query_evidence,
        scientific.calibration_block_size,
        local_budget,
        peer_budget,
    )
    return _CampaignDecisions(decisions, max_fusion, bonferroni, budget_sweep)


def _attack_type_performances(
    frame: TargetFrame, decisions: _CampaignDecisions
) -> tuple[AttackTypePerformance, ...]:
    rates = tuple(
        map(
            frame.detection_rates,
            (
                decisions.protected.combined_alerts,
                decisions.protected.local_alerts,
                decisions.max_fusion.combined_alerts,
                decisions.protected.peer_alerts,
                decisions.bonferroni.combined_alerts,
            ),
        )
    )
    return tuple(
        AttackTypePerformance(
            attack_type=attack_type,
            test_rows=features.shape[0],
            detection_rate=rates[0][index],
            local_detection_rate=rates[1][index],
            max_fusion_detection_rate=rates[2][index],
            peer_union_detection_rate=rates[3][index],
            bonferroni_detection_rate=rates[4][index],
        )
        for index, (attack_type, features) in enumerate(frame.held_out)
    )


def _target_performance(
    target: PreparedDevice,
    models: _SeedModels,
    configuration: ResolvedConfiguration,
    seed: LearningSeed,
) -> TargetPerformance:
    client = target.client
    peer_experts = tuple(expert for expert in models.experts if expert.client != client)
    if len(peer_experts) < configuration.scientific.minimum_peer_count:
        raise DatasetValidationError("Target has fewer eligible peers than the configured minimum")
    if not target.split.attack_splits:
        raise DatasetValidationError("Target has no eligible held-out attack types")
    frame, peers, local = _target_inputs(target, peer_experts, models, configuration, seed)
    decisions = _campaign_decisions(frame, peers, local, configuration.scientific)
    return TargetPerformance(
        target=client,
        seed=seed,
        split_digest=digest_split_plan(target.split),
        peer_model_seeds=tuple(
            ModelSeedProvenance(expert.client, expert.seed, SeedDomain.PEER_MODEL)
            for expert in peer_experts
        ),
        local_model_seed=ModelSeedProvenance(client, local.seed, SeedDomain.LOCAL_MODEL),
        benign_test_rows=frame.benign_count,
        realized_false_alert_rate=frame.false_alert_rate(decisions.protected.combined_alerts),
        attack_types=_attack_type_performances(frame, decisions),
        local_false_alert_rate=frame.false_alert_rate(decisions.protected.local_alerts),
        max_fusion_false_alert_rate=frame.false_alert_rate(decisions.max_fusion.combined_alerts),
        peer_union_false_alert_rate=frame.false_alert_rate(decisions.protected.peer_alerts),
        bonferroni_false_alert_rate=frame.false_alert_rate(decisions.bonferroni.combined_alerts),
        peer_count=len(peer_experts),
        budget_sweep=decisions.budget_sweep,
        study_stratum=target.study_stratum,
        arms=evaluate_arms(ArmContext(configuration, target, frame, peers, local)),
    )


def _cohort_targets(
    eligible: tuple[PreparedDevice, ...], cohort: tuple[ClientName, ...]
) -> tuple[PreparedDevice, ...]:
    if not cohort:
        return eligible
    unknown = set(cohort) - {target.client.name for target in eligible}
    if unknown:
        raise DatasetValidationError(
            f"Target cohort names clients outside the eligible study targets: {sorted(unknown)}"
        )
    return tuple(target for target in eligible if target.client.name in cohort)


def _run_seed(
    cache: MatrixTrainingCache, configuration: ResolvedConfiguration, seed: LearningSeed
) -> tuple[TargetPerformance, ...]:
    if cache.prepared is None:
        cache.prepared = prepare_clients(cache.clients, configuration)
        logger.info(LogEvent.CAMPAIGN_PREPARED, client_count=len(cache.prepared))
    prepared = cache.prepared
    models = cache.models_by_seed.get(seed)
    if models is None:
        models = cache.models_by_seed[seed] = _SeedModels(
            fit_peer_experts(prepared, configuration, seed), cache.frames
        )
    declared_targets = tuple(
        target
        for target in prepared
        if target.target_cohort_role is TargetCohortRole.REPORTED_TARGET
    )
    eligible_targets = tuple(
        target
        for target in prepared
        if target.target_cohort_role
        not in {TargetCohortRole.PEER_ONLY, TargetCohortRole.OUTSIDE_PEER_POOL}
        and assess_target_eligibility(
            len(target.split.attack_splits),
            len(target.split.benign_roles.calibration),
            sum(expert.client != target.client for expert in models.experts),
            configuration.splitting.minimum_calibration_rows,
            configuration.scientific.minimum_peer_count,
        ).is_eligible
    )
    if declared_targets and len(eligible_targets) != len(declared_targets):
        eligible_ids = {target.client for target in eligible_targets}
        excluded_ids = tuple(
            target.client.name for target in declared_targets if target.client not in eligible_ids
        )
        raise DatasetValidationError(
            f"Reported study targets fail configured eligibility: {excluded_ids}"
        )
    if not eligible_targets:
        raise DatasetValidationError("No target satisfies the frozen eligibility rules")
    selected_targets = _cohort_targets(eligible_targets, configuration.scientific.target_clients)
    performances = tuple(
        _target_performance(target, models, configuration, seed) for target in selected_targets
    )
    logger.info(LogEvent.CAMPAIGN_SEED_COMPLETED, seed=seed, target_count=len(performances))
    return performances


def run_dataset_study(
    clients: tuple[DatasetClientData, ...],
    configuration: ResolvedConfiguration,
    matrix_cache: MatrixTrainingCache | None = None,
) -> tuple[TargetPerformance, ...]:
    if matrix_cache is None:
        matrix_cache = MatrixTrainingCache(
            clients, configuration.scientific.peer_training, configuration.splitting
        )
    elif matrix_cache.clients is not clients:
        raise DatasetValidationError("Matrix training cache belongs to different client data")
    elif (
        matrix_cache.peer_training != configuration.scientific.peer_training
        or matrix_cache.splitting != configuration.splitting
    ):
        raise DatasetValidationError("Matrix training cache has incompatible peer training inputs")
    return tuple(
        result
        for seed in configuration.scientific.learning_seeds
        for result in _run_seed(matrix_cache, configuration, seed)
    )


def _logged_workflow(
    dataset: DatasetName,
    stratum: StudyStratum,
    configuration: ResolvedConfiguration,
    operation: Callable[[ArtifactPath, BoundLogger], CampaignWorkflowResult],
) -> CampaignWorkflowResult:
    destination = campaign_artifact_path(
        configuration.runtime.output_root, stratum, configuration.scientific_digest
    )
    configure_logging(
        execution_log_path(configuration.runtime.output_root, CommandName.RUN),
        configuration.runtime.log_level,
    )
    logger = structlog.get_logger().bind(
        command=CommandName.RUN,
        dataset=dataset.name,
        study_stratum=stratum.name,
        execution_device=configuration.runtime.execution_device.name,
        configuration_digest=configuration.scientific_digest,
        artifact_path=destination.path.as_posix(),
    )
    return execute_with_failure_logging(
        lambda: operation(destination, logger),
        logger,
        LogEvent.CAMPAIGN_FAILED,
        destination.path,
    )


def run_dataset_workflow(
    clients: tuple[DatasetClientData, ...],
    source_files: tuple[Path, ...],
    configuration: ResolvedConfiguration,
    replacement_policy: ReplacementPolicy = ReplacementPolicy.PRESERVE_VALID,
    matrix_cache: MatrixTrainingCache | None = None,
) -> CampaignWorkflowResult:
    dataset = validate_dataset_clients(clients)
    return _logged_workflow(
        dataset,
        clients[0].study_stratum,
        configuration,
        lambda destination, logger: _execute_dataset_workflow(
            clients,
            source_files,
            configuration,
            replacement_policy,
            destination,
            logger,
            (dataset, clients[0].study_stratum),
            matrix_cache,
        ),
    )


@dataclass(frozen=True, slots=True)
class MatrixRunResult:
    variant: MatrixVariant
    result: CampaignWorkflowResult


def run_dataset_matrix_workflow(
    clients: tuple[DatasetClientData, ...],
    source_files: tuple[Path, ...],
    configuration: ResolvedConfiguration,
    replacement_policy: ReplacementPolicy = ReplacementPolicy.PRESERVE_VALID,
) -> tuple[MatrixRunResult, ...]:
    matrix_cache = MatrixTrainingCache(
        clients, configuration.scientific.peer_training, configuration.splitting
    )
    raw_data_root = configuration.runtime.raw_data_root
    matrix_cache.input_digest = digest_source_files(source_files, raw_data_root)
    plan = plan_stratum_budgets(clients, configuration)
    persist_budget_plan(configuration.runtime.output_root, plan, matrix_cache.input_digest)
    results: list[MatrixRunResult] = []
    for variant in matrix_variants(plan.analyses, configuration.protocol_matrix.alpha_levels):
        logger.info(
            LogEvent.MATRIX_VARIANT_STARTED,
            analysis=variant.kind.name,
            alpha=variant.alpha.name,
            local_training_rows=variant.local_training_rows,
            target_count=len(variant.targets),
        )
        results.append(
            MatrixRunResult(
                variant,
                run_dataset_workflow(
                    clients,
                    source_files,
                    configuration_for_protocol_variant(
                        configuration, variant.alpha, variant.local_training_rows, variant.targets
                    ),
                    replacement_policy,
                    matrix_cache,
                ),
            )
        )
        if digest_source_files(source_files, raw_data_root) != matrix_cache.input_digest:
            raise DatasetValidationError("Dataset sources changed during matrix execution")
    return tuple(results)


def run_registered_dataset_matrix_workflow(
    dataset: DatasetName,
    configuration: ResolvedConfiguration,
    stratum: StudyStratum | None = None,
    replacement_policy: ReplacementPolicy = ReplacementPolicy.PRESERVE_VALID,
) -> tuple[MatrixRunResult, ...]:
    selected_stratum = resolve_study_stratum(dataset, stratum)
    raw_data_root = configuration.runtime.raw_data_root
    source_files = dataset_source_files(raw_data_root, dataset, selected_stratum)
    clients = resolve_clients(
        lambda: (
            load_dataset(
                raw_data_root, configuration.runtime.lab_network, dataset, selected_stratum
            ).clients
        ),
        configuration,
        (dataset, selected_stratum),
        digest_source_files(source_files, raw_data_root),
        logger,
    )
    return run_dataset_matrix_workflow(clients, source_files, configuration, replacement_policy)


def _execute_dataset_workflow(
    clients: tuple[DatasetClientData, ...] | Callable[[], tuple[DatasetClientData, ...]],
    source_files: tuple[Path, ...],
    configuration: ResolvedConfiguration,
    replacement_policy: ReplacementPolicy,
    destination: ArtifactPath,
    initial_logger: BoundLogger,
    identity: tuple[DatasetName, StudyStratum],
    matrix_cache: MatrixTrainingCache | None = None,
) -> CampaignWorkflowResult:
    dataset, stratum = identity
    shared_digest = None if matrix_cache is None else matrix_cache.input_digest
    input_digest = shared_digest or digest_source_files(
        source_files, configuration.runtime.raw_data_root
    )
    provenance = campaign_provenance(configuration, input_digest, identity, clients)
    logger = initial_logger.bind(
        input_digest=input_digest,
        implementation_digest=provenance.implementation_digest,
        preprocessing_digest=provenance.preprocessing_digest,
    )
    if destination.path.exists() and replacement_policy is ReplacementPolicy.PRESERVE_VALID:
        return reused_campaign(destination, provenance, logger)
    logger.info(
        LogEvent.CAMPAIGN_STARTED, dataset=dataset.name, source_file_count=len(source_files)
    )
    resolved_clients = resolve_clients(
        clients, configuration, (dataset, stratum), input_digest, logger
    )
    targets = (
        run_dataset_study(resolved_clients, configuration)
        if matrix_cache is None
        else run_dataset_study(resolved_clients, configuration, matrix_cache)
    )
    require_unchanged_sources(source_files, configuration, input_digest)
    outcome = CampaignOutcome.from_targets(targets)
    artifact, reuse_state = reuse_or_write_artifact(
        destination.path, outcome, provenance, replacement_policy
    )
    logger.info(
        LogEvent.CAMPAIGN_COMPLETED,
        target_count=outcome.target_count,
        artifact_path=destination.path.as_posix(),
        content_digest=artifact.content_digest,
        state=reuse_state.name,
    )
    return workflow_result(destination, artifact.content_digest, reuse_state, outcome)
