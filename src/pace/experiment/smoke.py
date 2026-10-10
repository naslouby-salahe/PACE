from dataclasses import dataclass

import numpy as np
import structlog
from structlog.stdlib import BoundLogger

from pace.config import ResolvedConfiguration, SplitSettings
from pace.core.decision import apply_protected_local_peer_rule
from pace.core.evidence import local_ensemble_evidence, target_relative_evidence
from pace.experiment.artifacts import (
    Provenance,
    SyntheticOutcome,
    artifact_path,
    digest_implementation,
    digest_inputs,
    execution_log_path,
    reuse_or_write_artifact,
)
from pace.experiment.inputs import PeerMLP, train_peer_mlp
from pace.experiment.logging_setup import configure_logging, execute_with_failure_logging
from pace.experiment.models import fit_local_ensemble, select_local_training_rows
from pace.experiment.splitting import chronological_indices, fit_pooled_standardizer
from pace.types import (
    AlertDecision,
    ArtifactPath,
    ArtifactPayload,
    ArtifactReuseState,
    ArtifactVerification,
    BooleanArray,
    CommandName,
    DetectionOutcome,
    DetectorScoreMatrix,
    FixtureKind,
    FloatArray,
    FloatMatrix,
    ImplementationScope,
    InputDigestMaterial,
    LearningSeed,
    LogEvent,
    MinimumPeerCount,
    ReplacementPolicy,
    ScientificDigest,
    SyntheticFixtureShape,
    SyntheticWorkflowResult,
    VerificationState,
)


@dataclass(frozen=True, slots=True)
class _SyntheticScores:
    peer_models: tuple[PeerMLP, ...]
    peer_reference_margins: FloatArray
    peer_calibration_margins: FloatArray
    peer_evaluation_margins: FloatArray
    local_reference_scores: DetectorScoreMatrix
    local_calibration_scores: DetectorScoreMatrix
    local_evaluation_scores: DetectorScoreMatrix
    attack_labels: BooleanArray


@dataclass(frozen=True, slots=True)
class _SyntheticTarget:
    training: FloatMatrix
    reference: FloatMatrix
    calibration: FloatMatrix
    test_benign: FloatMatrix
    attacks: FloatMatrix


def _synthetic_target(
    generator: np.random.Generator, shape: SyntheticFixtureShape, splitting: SplitSettings
) -> _SyntheticTarget:
    epoch = np.datetime64("1970-01-01", "s")
    benign_timestamps = epoch + np.arange(shape.benign_rows).astype("timedelta64[s]")
    target_benign = generator.normal(0.0, 1.0, (shape.benign_rows, shape.feature_count))
    roles = chronological_indices(benign_timestamps, splitting)
    attack_dimension = np.arange(shape.attack_rows) % shape.feature_count
    attacks = generator.normal(0.0, 1.0, (shape.attack_rows, shape.feature_count))
    attacks[np.arange(shape.attack_rows), attack_dimension] += shape.attack_shift
    return _SyntheticTarget(
        target_benign[roles.training],
        target_benign[roles.reference],
        target_benign[roles.calibration],
        target_benign[roles.test],
        attacks,
    )


def _synthetic_peer_models(
    generator: np.random.Generator,
    seed: LearningSeed,
    peer_count: MinimumPeerCount,
    shape: SyntheticFixtureShape,
    configuration: ResolvedConfiguration,
) -> tuple[PeerMLP, ...]:
    peer_seed_sequences = np.random.SeedSequence(seed).spawn(peer_count)
    return tuple(
        train_peer_mlp(
            generator.normal(0.0, 1.0, (shape.peer_benign_rows, shape.feature_count)),
            generator.normal(
                np.eye(shape.feature_count)[peer_index % shape.feature_count] * shape.attack_shift,
                1.0,
                (shape.peer_attack_rows, shape.feature_count),
            ),
            seed_sequence.generate_state(1)[0].item(),
            configuration.scientific.peer_training,
        )
        for peer_index, seed_sequence in enumerate(peer_seed_sequences)
    )


def _local_scores(
    target: _SyntheticTarget, seed: LearningSeed, configuration: ResolvedConfiguration
) -> tuple[DetectorScoreMatrix, DetectorScoreMatrix, DetectorScoreMatrix]:
    standardizer = fit_pooled_standardizer((target.training,))
    local_ensemble = fit_local_ensemble(
        select_local_training_rows(
            standardizer.transform(target.training),
            configuration.scientific.local_detectors.local_training_rows,
        ),
        configuration.scientific.local_detectors,
        seed,
        configuration.runtime.execution_device,
    )
    reference = local_ensemble.scores(standardizer.transform(target.reference))
    calibration = local_ensemble.scores(standardizer.transform(target.calibration))
    benign = local_ensemble.scores(standardizer.transform(target.test_benign))
    attacks = local_ensemble.scores(standardizer.transform(target.attacks))
    evaluation = DetectorScoreMatrix(np.concatenate((benign.values, attacks.values), axis=1))
    return reference, calibration, evaluation


def _synthetic_scores(
    seed: LearningSeed,
    peer_count: MinimumPeerCount,
    configuration: ResolvedConfiguration,
) -> _SyntheticScores:
    generator = np.random.default_rng(seed)
    shape = SyntheticFixtureShape.standard()
    target = _synthetic_target(generator, shape, configuration.splitting)
    peer_models = _synthetic_peer_models(generator, seed, peer_count, shape, configuration)
    peer_evaluation = np.concatenate(
        (
            np.stack(tuple(model.margins(target.test_benign) for model in peer_models)),
            np.stack(tuple(model.margins(target.attacks) for model in peer_models)),
        ),
        axis=1,
    )
    local_reference, local_calibration, local_evaluation = _local_scores(
        target, seed, configuration
    )
    labels = np.concatenate(
        (
            np.zeros(target.test_benign.shape[0], dtype=np.bool_),
            np.ones(shape.attack_rows, dtype=np.bool_),
        )
    )
    return _SyntheticScores(
        peer_models,
        np.stack(tuple(model.margins(target.reference) for model in peer_models)),
        np.stack(tuple(model.margins(target.calibration) for model in peer_models)),
        peer_evaluation,
        local_reference,
        local_calibration,
        local_evaluation,
        labels,
    )


def _peer_evidence(reference_margins: FloatArray, query_margins: FloatArray) -> FloatArray:
    return np.stack(
        tuple(
            target_relative_evidence(reference_margins[index], query_margins[index])
            for index in range(reference_margins.shape[0])
        ),
        axis=0,
    )


def _outcome_record(
    outcome: DetectionOutcome,
) -> SyntheticOutcome:
    return SyntheticOutcome(
        fixture_kind=FixtureKind.DETERMINISTIC_SYNTHETIC,
        row_count=outcome.evaluation_count,
        detected_count=outcome.alert_count,
        local_alert_count=sum(alert is AlertDecision.ALERT for alert in outcome.local_alerts),
        peer_alert_count=sum(alert is AlertDecision.ALERT for alert in outcome.peer_alerts),
        benign_alert_rate=outcome.benign_alert_rate,
        attack_detection_rate=outcome.attack_detection_rate,
        local_threshold=outcome.local_threshold,
        expert_thresholds=outcome.expert_thresholds,
        joint_multiplier=outcome.joint_multiplier,
        peer_block_alert_rates=outcome.peer_block_alert_rates,
        alerts=outcome.alerts,
    )


def _alert_decisions(mask: BooleanArray) -> tuple[AlertDecision, ...]:
    return tuple(AlertDecision.ALERT if alert else AlertDecision.PASS for alert in mask)


def _compute_outcome(
    configuration: ResolvedConfiguration,
    scores: _SyntheticScores,
) -> DetectionOutcome:
    scientific = configuration.scientific
    peer_calibration_evidence = _peer_evidence(
        scores.peer_reference_margins, scores.peer_calibration_margins
    )
    peer_evaluation_evidence = _peer_evidence(
        scores.peer_reference_margins, scores.peer_evaluation_margins
    )
    local_calibration_evidence = local_ensemble_evidence(
        scores.local_reference_scores, scores.local_calibration_scores
    )
    local_evaluation_evidence = local_ensemble_evidence(
        scores.local_reference_scores, scores.local_evaluation_scores
    )
    nominal_budget = scientific.alpha.fraction
    local_budget = nominal_budget * scientific.reserve_fraction
    peer_budget = nominal_budget * (1.0 - scientific.reserve_fraction)
    channel_decisions = apply_protected_local_peer_rule(
        local_calibration_evidence,
        local_evaluation_evidence,
        peer_calibration_evidence,
        peer_evaluation_evidence,
        scientific.calibration_block_size,
        local_budget,
        peer_budget,
        scientific.joint_scale_cap,
        scientific.joint_scale_iterations,
    )
    combined_alert_mask = channel_decisions.combined_alerts
    benign_labels = np.logical_not(scores.attack_labels)
    benign_rate = np.mean(combined_alert_mask[benign_labels]).item()
    attack_rate = np.mean(combined_alert_mask[scores.attack_labels]).item()
    local_alerts = _alert_decisions(channel_decisions.local_alerts)
    peer_decisions = _alert_decisions(channel_decisions.peer_alerts)
    combined_alerts = _alert_decisions(combined_alert_mask)
    return DetectionOutcome(
        expert_ids=tuple(
            f"peer-{index + 1}" for index in range(scores.peer_reference_margins.shape[0])
        ),
        local_threshold=channel_decisions.local_threshold,
        expert_thresholds=channel_decisions.expert_thresholds,
        joint_multiplier=channel_decisions.joint_multiplier,
        peer_block_alert_rates=channel_decisions.peer_block_alert_rates,
        evaluation_count=len(combined_alerts),
        alert_count=sum(alert is AlertDecision.ALERT for alert in combined_alerts),
        benign_alert_rate=benign_rate,
        attack_detection_rate=attack_rate,
        local_alerts=local_alerts,
        peer_alerts=peer_decisions,
        alerts=combined_alerts,
    )


def run_synthetic_workflow(
    configuration: ResolvedConfiguration,
    replacement_policy: ReplacementPolicy = ReplacementPolicy.PRESERVE_VALID,
) -> SyntheticWorkflowResult:
    output_root = configuration.runtime.output_root
    configure_logging(
        execution_log_path(output_root, CommandName.SMOKE), configuration.runtime.log_level
    )
    implementation_digest = digest_implementation(ImplementationScope.SYNTHETIC)
    destination = artifact_path(output_root, configuration.scientific_digest)
    logger = structlog.get_logger().bind(
        command=CommandName.SMOKE,
        execution_device=configuration.runtime.execution_device.name,
        configuration_digest=configuration.scientific_digest,
        implementation_digest=implementation_digest,
        seed=configuration.scientific.seed,
    )
    return execute_with_failure_logging(
        lambda: _execute_synthetic_workflow(
            configuration,
            replacement_policy,
            implementation_digest,
            destination,
            logger,
        ),
        logger,
        LogEvent.RUN_FAILED,
        destination.path,
    )


def _synthetic_input_digest(scores: _SyntheticScores) -> ScientificDigest:
    return digest_inputs(
        InputDigestMaterial(
            tuple(
                ArtifactPayload(array.tobytes(order="C"))
                for array in (
                    scores.peer_reference_margins,
                    scores.peer_calibration_margins,
                    scores.peer_evaluation_margins,
                    scores.local_reference_scores.values,
                    scores.local_calibration_scores.values,
                    scores.local_evaluation_scores.values,
                    scores.attack_labels,
                )
            )
            + tuple(
                ArtifactPayload(parameter.tobytes(order="C"))
                for model in scores.peer_models
                for layer in model.layers
                for parameter in (layer.weights, layer.bias)
            )
        )
    )


def _execute_synthetic_workflow(
    configuration: ResolvedConfiguration,
    replacement_policy: ReplacementPolicy,
    implementation_digest: ScientificDigest,
    destination: ArtifactPath,
    logger: BoundLogger,
) -> SyntheticWorkflowResult:
    logger.info(
        LogEvent.RUN_STARTED,
        fixture_kind=FixtureKind.DETERMINISTIC_SYNTHETIC.name,
    )
    scores = _synthetic_scores(
        configuration.scientific.seed,
        configuration.scientific.minimum_peer_count,
        configuration,
    )
    input_digest = _synthetic_input_digest(scores)
    outcome = _compute_outcome(configuration, scores)
    logger.info(
        LogEvent.CALIBRATION_COMPLETED,
        expert_count=len(outcome.expert_ids),
        joint_multiplier=outcome.joint_multiplier,
        block_count=len(outcome.peer_block_alert_rates),
    )
    artifact, reuse_state = reuse_or_write_artifact(
        destination.path,
        _outcome_record(outcome),
        Provenance(
            configuration.scientific,
            configuration.splitting,
            configuration.scientific_digest,
            input_digest,
            implementation_digest,
            execution_device=configuration.runtime.execution_device,
            provenance_active=configuration.runtime.provenance_active,
        ),
        replacement_policy,
    )
    logger.info(
        LogEvent.RUN_REUSED
        if reuse_state is ArtifactReuseState.REUSED
        else LogEvent.ARTIFACT_WRITTEN,
        artifact_path=destination.path.as_posix(),
        content_digest=artifact.content_digest,
        state=reuse_state.name,
    )
    return SyntheticWorkflowResult(
        artifact_path=destination,
        verification=ArtifactVerification(
            VerificationState.VERIFIED,
            artifact.content_digest,
        ),
        reuse_state=reuse_state,
    )
