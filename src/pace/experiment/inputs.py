from collections.abc import Callable
from dataclasses import dataclass
from math import sqrt

import numpy as np
import structlog
from scipy.special import expit

from pace.config import PeerTrainingSettings, ResolvedConfiguration
from pace.core.decision import TieredLocalEvidence
from pace.core.evidence import (
    local_ensemble_evidence,
    probability_tail_evidence,
    target_relative_evidence,
)
from pace.experiment.artifacts import derive_learning_seed
from pace.experiment.models import (
    LocalBankDetectorScores,
    LocalBankEvidence,
    LocalBankFeatures,
    LocalBankResponses,
    LocalDetectorEnsemble,
    build_local_bank,
    fit_local_ensemble,
    select_local_training_rows,
)
from pace.experiment.splitting import PooledStandardizer
from pace.types import (
    AttackHalfIndices,
    AttackType,
    BooleanArray,
    ClassLabelArray,
    DatasetClientData,
    DatasetClientId,
    DatasetSplitPlan,
    DetectionRate,
    FalseAlertRate,
    FeatureCount,
    FloatArray,
    FloatMatrix,
    LearningSeed,
    LocalDetectorSeed,
    LogEvent,
    ObservationCount,
    RowIndexArray,
    SeedDomain,
    StudyStratum,
    TargetCohortRole,
)

logger = structlog.get_logger()


@dataclass(frozen=True, slots=True)
class DenseLayer:
    weights: FloatArray
    bias: FloatArray


@dataclass(frozen=True, slots=True)
class PeerMLP:
    layers: tuple[DenseLayer, DenseLayer, DenseLayer]
    feature_count: FeatureCount

    def __post_init__(self) -> None:
        expected_width = self.feature_count
        for layer in self.layers:
            if (
                layer.weights.ndim != 2
                or layer.weights.shape[0] != expected_width
                or layer.weights.shape[1] != layer.bias.size
                or layer.bias.ndim != 1
                or layer.bias.size == 0
                or not np.isfinite(layer.weights).all()
                or not np.isfinite(layer.bias).all()
            ):
                raise ValueError("Peer MLP layer dimensions and parameters must be valid")
            layer.weights.setflags(write=False)
            layer.bias.setflags(write=False)
            expected_width = layer.bias.size
        if len(self.layers) != 3 or self.layers[-1].bias.size != 2:
            raise ValueError("Peer MLP must contain two hidden layers and two output logits")

    def margins(self, features: FloatArray) -> FloatArray:
        if features.ndim != 2 or features.shape[0] == 0:
            raise ValueError("Peer inference requires a non-empty feature matrix")
        if features.shape[1] != self.feature_count:
            raise ValueError("Peer inference features differ from the trained feature schema")
        if not np.isfinite(features).all():
            raise ValueError("Peer inference features must be finite")
        activations = features
        for layer in self.layers[:-1]:
            activations = np.maximum(activations @ layer.weights + layer.bias, 0.0)
        logits = activations @ self.layers[-1].weights + self.layers[-1].bias
        return logits[:, 1] - logits[:, 0]


def attack_probabilities_from_margins(margins: FloatArray) -> FloatArray:
    return np.clip(expit(margins), np.finfo(np.float64).tiny, 1.0)


def _validate_training_features(
    benign_features: FloatArray, attack_features: FloatArray
) -> FeatureCount:
    if (
        benign_features.ndim != 2
        or attack_features.ndim != 2
        or benign_features.shape[0] == 0
        or attack_features.shape[0] == 0
        or benign_features.shape[1] == 0
        or benign_features.shape[1] != attack_features.shape[1]
    ):
        raise ValueError("Peer training requires aligned non-empty benign and attack matrices")
    if not np.isfinite(benign_features).all() or not np.isfinite(attack_features).all():
        raise ValueError("Peer training features must be finite")
    return benign_features.shape[1]


def _initial_parameters(
    feature_count: FeatureCount,
    settings: PeerTrainingSettings,
    generator: np.random.Generator,
) -> list[FloatArray]:
    widths = (
        feature_count,
        settings.hidden_layer_widths[0],
        settings.hidden_layer_widths[1],
        2,
    )
    parameters: list[FloatArray] = []
    for input_width, output_width in zip(widths[:-1], widths[1:], strict=True):
        initialization_bound = 1.0 / sqrt(input_width)
        parameters.append(
            generator.uniform(
                -initialization_bound, initialization_bound, (input_width, output_width)
            )
        )
        parameters.append(
            generator.uniform(-initialization_bound, initialization_bound, output_width)
        )
    return parameters


def _batch_gradients(
    parameters: list[FloatArray],
    features: FloatArray,
    labels: ClassLabelArray,
    sample_weights: FloatArray,
) -> list[FloatArray]:
    first_preactivation = features @ parameters[0] + parameters[1]
    first_activation = np.maximum(first_preactivation, 0.0)
    second_preactivation = first_activation @ parameters[2] + parameters[3]
    second_activation = np.maximum(second_preactivation, 0.0)
    logits = second_activation @ parameters[4] + parameters[5]
    logits -= np.max(logits, axis=1, keepdims=True)
    exponentials = np.exp(logits)
    probabilities = exponentials / np.sum(exponentials, axis=1, keepdims=True)
    probabilities[np.arange(labels.size), labels] -= 1.0
    probabilities *= sample_weights[:, np.newaxis] / labels.size
    output_weight_gradient = second_activation.T @ probabilities
    output_bias_gradient = np.sum(probabilities, axis=0)
    second_gradient = (probabilities @ parameters[4].T) * (second_preactivation > 0.0)
    second_weight_gradient = first_activation.T @ second_gradient
    second_bias_gradient = np.sum(second_gradient, axis=0)
    first_gradient = (second_gradient @ parameters[2].T) * (first_preactivation > 0.0)
    first_weight_gradient = features.T @ first_gradient
    first_bias_gradient = np.sum(first_gradient, axis=0)
    return [
        first_weight_gradient,
        first_bias_gradient,
        second_weight_gradient,
        second_bias_gradient,
        output_weight_gradient,
        output_bias_gradient,
    ]


def _balanced_sample_weights(
    benign_count: ObservationCount, attack_count: ObservationCount
) -> FloatArray:
    total_count = benign_count + attack_count
    return np.concatenate(
        (
            np.full(
                benign_count,
                total_count / (2 * benign_count),
                dtype=np.float64,
            ),
            np.full(
                attack_count,
                total_count / (2 * attack_count),
                dtype=np.float64,
            ),
        )
    )


def train_peer_mlp(
    benign_features: FloatArray,
    attack_features: FloatArray,
    seed: LearningSeed,
    settings: PeerTrainingSettings,
) -> PeerMLP:
    feature_count = _validate_training_features(benign_features, attack_features)
    training_features = np.concatenate((benign_features, attack_features), axis=0)
    labels = np.concatenate(
        (
            np.zeros(benign_features.shape[0], dtype=np.int64),
            np.ones(attack_features.shape[0], dtype=np.int64),
        )
    )
    sample_weights = _balanced_sample_weights(
        benign_features.shape[0],
        attack_features.shape[0],
    )
    generator = np.random.default_rng(seed)
    parameters = _initial_parameters(feature_count, settings, generator)
    first_moments = [np.zeros_like(parameter) for parameter in parameters]
    second_moments = [np.zeros_like(parameter) for parameter in parameters]
    optimizer_step = 0
    for _ in range(settings.epochs):
        shuffled_rows = generator.permutation(labels.size)
        for batch_start in range(0, labels.size, settings.batch_size):
            batch_rows = shuffled_rows[batch_start : batch_start + settings.batch_size]
            gradients = _batch_gradients(
                parameters,
                training_features[batch_rows],
                labels[batch_rows],
                sample_weights[batch_rows],
            )
            optimizer_step += 1
            for index, gradient in enumerate(gradients):
                first_moments[index] = 0.9 * first_moments[index] + 0.1 * gradient
                second_moments[index] = 0.999 * second_moments[index] + 0.001 * np.square(gradient)
                corrected_first = first_moments[index] / (1.0 - 0.9**optimizer_step)
                corrected_second = second_moments[index] / (1.0 - 0.999**optimizer_step)
                parameters[index] -= (
                    settings.learning_rate * corrected_first / (np.sqrt(corrected_second) + 1e-8)
                )
    layers = (
        DenseLayer(parameters[0], parameters[1]),
        DenseLayer(parameters[2], parameters[3]),
        DenseLayer(parameters[4], parameters[5]),
    )
    return PeerMLP(layers, feature_count)


@dataclass(frozen=True, slots=True)
class PreparedDevice:
    client: DatasetClientId
    study_stratum: StudyStratum
    target_cohort_role: TargetCohortRole
    split: DatasetSplitPlan
    training: FloatMatrix
    reference: FloatMatrix
    calibration: FloatMatrix
    test: FloatMatrix
    development: tuple[tuple[AttackType, FloatMatrix], ...]
    held_out: tuple[tuple[AttackType, FloatMatrix], ...]


def prepare_device(
    device: DatasetClientData, plan: DatasetSplitPlan, standardizer: PooledStandardizer
) -> PreparedDevice:
    roles = plan.benign_roles
    captures = {capture.attack_type: capture.features for capture in device.attack_captures}

    def attack_rows(
        select: Callable[[AttackHalfIndices], RowIndexArray],
    ) -> tuple[tuple[AttackType, FloatMatrix], ...]:
        return tuple(
            (
                split.attack_type,
                standardizer.transform(captures[split.attack_type][select(split.indices)]),
            )
            for split in plan.attack_splits
        )

    def benign_rows(indices: RowIndexArray) -> FloatMatrix:
        return standardizer.transform(device.benign_features[indices])

    return PreparedDevice(
        device.client,
        device.study_stratum,
        device.target_cohort_role,
        plan,
        benign_rows(roles.training),
        benign_rows(roles.reference),
        benign_rows(roles.calibration),
        benign_rows(roles.test),
        attack_rows(lambda indices: indices.peer_development),
        attack_rows(lambda indices: indices.held_out),
    )


@dataclass(frozen=True, slots=True)
class PeerExpert:
    client: DatasetClientId
    seed: LearningSeed
    model: PeerMLP


@dataclass(frozen=True, slots=True)
class TargetFrame:
    reference: FloatMatrix
    calibration: FloatMatrix
    query: FloatMatrix
    benign_count: ObservationCount
    held_out: tuple[tuple[AttackType, FloatMatrix], ...]

    def attack_slices(self) -> tuple[slice, ...]:
        slices: list[slice] = []
        offset = self.benign_count
        for _, features in self.held_out:
            slices.append(slice(offset, offset + features.shape[0]))
            offset += features.shape[0]
        return tuple(slices)

    def false_alert_rate(self, alerts: BooleanArray) -> FalseAlertRate:
        return np.mean(alerts[: self.benign_count]).item()

    def detection_rates(self, alerts: BooleanArray) -> tuple[DetectionRate, ...]:
        return tuple(np.mean(alerts[rows]).item() for rows in self.attack_slices())


def build_target_frame(target: PreparedDevice) -> TargetFrame:
    query = np.concatenate((target.test, *(features for _, features in target.held_out)), axis=0)
    return TargetFrame(
        target.reference, target.calibration, query, target.test.shape[0], target.held_out
    )


@dataclass(frozen=True, slots=True)
class PeerEvidence:
    experts: tuple[PeerExpert, ...]
    reference_margins: FloatMatrix
    calibration_margins: FloatMatrix
    query_margins: FloatMatrix
    reference_probabilities: FloatMatrix
    calibration_evidence: FloatMatrix
    query_evidence: FloatMatrix
    max_fusion_calibration_evidence: FloatMatrix
    max_fusion_query_evidence: FloatMatrix


def _stack_evidence(
    evidence: Callable[[FloatArray, FloatArray], FloatArray],
    reference: FloatMatrix,
    query: FloatMatrix,
) -> FloatMatrix:
    return np.stack([evidence(reference[row], query[row]) for row in range(reference.shape[0])])


def _margin_rows(
    experts: tuple[PeerExpert, ...],
    features: FloatMatrix,
    previous: PeerEvidence | None,
    previous_margins: Callable[[PeerEvidence], FloatMatrix],
) -> FloatMatrix:
    cached = (
        {}
        if previous is None
        else {
            expert.client: (expert, row)
            for expert, row in zip(previous.experts, previous_margins(previous), strict=True)
        }
    )
    rows: list[FloatArray] = []
    for expert in experts:
        hit = cached.get(expert.client)
        rows.append(
            hit[1] if hit is not None and hit[0] is expert else expert.model.margins(features)
        )
    return np.stack(rows)


def compute_peer_evidence(
    experts: tuple[PeerExpert, ...], frame: TargetFrame, previous: PeerEvidence | None = None
) -> PeerEvidence:
    reference_margins = _margin_rows(
        experts, frame.reference, previous, lambda item: item.reference_margins
    )
    calibration_margins = _margin_rows(
        experts, frame.calibration, previous, lambda item: item.calibration_margins
    )
    query_margins = _margin_rows(experts, frame.query, previous, lambda item: item.query_margins)
    reference_probabilities = attack_probabilities_from_margins(reference_margins)
    calibration_probabilities = attack_probabilities_from_margins(calibration_margins)
    query_probabilities = attack_probabilities_from_margins(query_margins)
    return PeerEvidence(
        experts,
        reference_margins,
        calibration_margins,
        query_margins,
        reference_probabilities,
        _stack_evidence(target_relative_evidence, reference_margins, calibration_margins),
        _stack_evidence(target_relative_evidence, reference_margins, query_margins),
        _stack_evidence(
            probability_tail_evidence, reference_probabilities, calibration_probabilities
        ),
        _stack_evidence(probability_tail_evidence, reference_probabilities, query_probabilities),
    )


@dataclass(frozen=True, slots=True)
class LocalEvidence:
    seed: LearningSeed
    calibration: FloatArray
    query: FloatArray
    bank: LocalBankEvidence

    def tiered(self, bank: LocalBankEvidence | None = None) -> TieredLocalEvidence:
        selected = self.bank if bank is None else bank
        return TieredLocalEvidence(
            selected.anchor_calibration,
            selected.anchor_query,
            selected.calibration,
            selected.query,
        )


def peer_response_vectors(peers: PeerEvidence, features: FloatMatrix) -> FloatMatrix:
    margins = np.stack([expert.model.margins(features) for expert in peers.experts])
    return np.asarray(_stack_evidence(target_relative_evidence, peers.reference_margins, margins).T)


def _bank_evidence(
    local: LocalDetectorEnsemble,
    target: PreparedDevice,
    frame: TargetFrame,
    peers: PeerEvidence,
    configuration: ResolvedConfiguration,
    local_seed: LearningSeed,
) -> LocalBankEvidence:
    training = select_local_training_rows(
        target.training, configuration.scientific.local_detectors.local_training_rows
    )
    return build_local_bank(
        LocalBankDetectorScores(
            local.scores(frame.reference),
            local.scores(frame.calibration),
            local.scores(frame.query),
        ),
        LocalBankFeatures(training, frame.reference, frame.calibration, frame.query),
        LocalBankResponses(
            peer_response_vectors(peers, training),
            peer_response_vectors(peers, frame.reference),
            peer_response_vectors(peers, frame.calibration),
            peer_response_vectors(peers, frame.query),
        ),
        configuration.scientific.local_bank,
        LocalDetectorSeed(local_seed),
    )


def compute_local_evidence(
    target: PreparedDevice,
    frame: TargetFrame,
    configuration: ResolvedConfiguration,
    seed: LearningSeed,
    peers: PeerEvidence,
) -> LocalEvidence:
    local_seed = derive_learning_seed(seed, target.client, SeedDomain.LOCAL_MODEL)
    local = fit_local_ensemble(
        select_local_training_rows(
            target.training, configuration.scientific.local_detectors.local_training_rows
        ),
        configuration.scientific.local_detectors,
        local_seed,
        configuration.runtime.execution_device,
    )
    reference = local.scores(frame.reference)
    logger.info(LogEvent.LOCAL_ENSEMBLE_FITTED, client=target.client.name, seed=local_seed)
    bank = _bank_evidence(local, target, frame, peers, configuration, local_seed)
    return LocalEvidence(
        local_seed,
        local_ensemble_evidence(reference, local.scores(frame.calibration)),
        local_ensemble_evidence(reference, local.scores(frame.query)),
        bank,
    )
