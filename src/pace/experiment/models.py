from contextlib import AbstractContextManager
from dataclasses import dataclass
from types import ModuleType
from typing import Protocol, cast

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from torch import Tensor, nn

from pace.config import LocalBankSettings, LocalDetectorSettings
from pace.core.evidence import (
    anchor_detector_rows,
    fit_ecod_sum,
    fit_pair_cooccurrence,
    local_detector_evidence,
    target_relative_evidence,
)
from pace.types import (
    CovarianceRegularization,
    CudaDeviceIndex,
    DetectorCount,
    DetectorSampleLimit,
    DetectorScoreMatrix,
    ExecutionDevice,
    ExecutionDeviceUnavailableError,
    FeatureCount,
    FloatArray,
    FloatMatrix,
    IntMatrix,
    LearningSeed,
    LocalDetectorError,
    LocalDetectorSeed,
    LocalExpertFamily,
    LocalExpertKind,
    LocalTrainingRowCount,
    PcaSolver,
)


class _TorchRandomModule(Protocol):
    def fork_rng(self, devices: list[CudaDeviceIndex]) -> AbstractContextManager[None]: ...

    def manual_seed(self, seed: LocalDetectorSeed) -> torch.Generator: ...


class TorchOptimizer(Protocol):
    def zero_grad(self) -> None: ...

    def step(self) -> None: ...


class _TorchLoss(Protocol):
    def backward(self) -> None: ...


def adapt_library_result[LibraryType](
    value: object, expected_implementation: type[object], _protocol: type[LibraryType]
) -> LibraryType:
    if not isinstance(value, expected_implementation):
        raise LocalDetectorError("Third-party library returned an unexpected implementation")
    return cast(LibraryType, value)


class ReconstructionNetwork(nn.Module):
    def __init__(self, feature_count: FeatureCount, settings: LocalDetectorSettings) -> None:
        super().__init__()
        first_width, second_width = settings.autoencoder_hidden_widths
        self.network = nn.Sequential(
            nn.Linear(feature_count, first_width),
            nn.ReLU(),
            nn.Linear(first_width, second_width),
            nn.ReLU(),
            nn.Linear(second_width, first_width),
            nn.ReLU(),
            nn.Linear(first_width, feature_count),
        )

    def forward(self, features: Tensor) -> Tensor:
        return self.network(features)


@dataclass(frozen=True, slots=True)
class LocalDetectorEnsemble:
    feature_count: FeatureCount
    device: torch.device
    autoencoder: ReconstructionNetwork
    pca: PCA
    isolation_forest: IsolationForest
    lof: LocalOutlierFactor
    neighbor_count: DetectorCount
    histogram_edges: FloatMatrix
    histogram_probabilities: FloatMatrix
    gaussian_mean: FloatMatrix
    gaussian_precision: FloatMatrix

    def scores(self, features: FloatMatrix) -> DetectorScoreMatrix:
        _validate_features(features, self.feature_count)
        ae_scores = _reconstruction_scores(features, self.autoencoder, self.device)
        projected = self.pca.transform(features)
        reconstructed = self.pca.inverse_transform(projected)
        if (
            projected.ndim != 2
            or projected.shape[0] != features.shape[0]
            or reconstructed.shape != features.shape
            or not np.isfinite(projected).all()
            or not np.isfinite(reconstructed).all()
        ):
            raise LocalDetectorError("PCA returned invalid transformed features")
        pca_scores = np.mean(np.square(features - reconstructed), axis=1)
        isolation_scores = -self.isolation_forest.score_samples(features)
        lof_scores = -self.lof.score_samples(features)
        if (
            isolation_scores.shape != (features.shape[0],)
            or lof_scores.shape != (features.shape[0],)
            or not np.isfinite(isolation_scores).all()
            or not np.isfinite(lof_scores).all()
        ):
            raise LocalDetectorError("Novelty detectors returned invalid score vectors")
        neighbor_distances, _ = self.lof.kneighbors(features, self.neighbor_count)
        if (
            neighbor_distances.shape != (features.shape[0], self.neighbor_count)
            or not np.isfinite(neighbor_distances).all()
            or (neighbor_distances < 0).any()
        ):
            raise LocalDetectorError("Nearest-neighbor detector returned invalid distances")
        neighbor_scores = np.mean(neighbor_distances, axis=1)
        histogram_scores = _histogram_scores(
            features, self.histogram_edges, self.histogram_probabilities
        )
        centered = features - self.gaussian_mean
        gaussian_scores = np.sum((centered @ self.gaussian_precision) * centered, axis=1)
        detector_scores = np.stack(
            (
                ae_scores,
                pca_scores,
                isolation_scores,
                lof_scores,
                neighbor_scores,
                histogram_scores,
                gaussian_scores,
            ),
            axis=0,
        )
        if not np.isfinite(detector_scores).all():
            raise LocalDetectorError("Local detector inference produced non-finite scores")
        return DetectorScoreMatrix(np.asarray(detector_scores, dtype=np.float64))


def resolve_torch_device(execution_device: ExecutionDevice) -> torch.device:
    match execution_device:
        case ExecutionDevice.CPU:
            return torch.device("cpu")
        case ExecutionDevice.CUDA:
            if not torch.cuda.is_available():
                raise ExecutionDeviceUnavailableError(
                    "CUDA was requested for local detector training but is unavailable"
                )
            return torch.device("cuda", torch.cuda.current_device())


def _validate_features(features: FloatMatrix, feature_count: FeatureCount | None = None) -> None:
    if (
        features.ndim != 2
        or features.shape[0] == 0
        or features.shape[1] == 0
        or (feature_count is not None and features.shape[1] != feature_count)
        or not np.isfinite(features).all()
    ):
        raise ValueError("Local detector features must be non-empty, finite, and schema-aligned")


def _bounded_fit_sample(
    features: FloatMatrix, limit: DetectorSampleLimit, seed: LocalDetectorSeed
) -> FloatMatrix:
    if features.shape[0] <= limit:
        return features
    rows = np.random.default_rng(seed).choice(features.shape[0], limit, replace=False)
    return features[np.sort(rows)]


def select_local_training_rows(
    chronological_training: FloatMatrix, row_count: LocalTrainingRowCount
) -> FloatMatrix:
    _validate_features(chronological_training)
    if chronological_training.shape[0] < row_count:
        raise ValueError("Chronological training role has fewer rows than the local budget")
    selected = np.array(chronological_training[:row_count], copy=True)
    selected.setflags(write=False)
    return selected


def _cuda_device_indices(execution_device: ExecutionDevice) -> list[CudaDeviceIndex]:
    match execution_device:
        case ExecutionDevice.CPU:
            return []
        case ExecutionDevice.CUDA:
            return [CudaDeviceIndex(torch.cuda.current_device())]


def _train_autoencoder(
    features: FloatMatrix,
    settings: LocalDetectorSettings,
    seed: LocalDetectorSeed,
    device: torch.device,
    execution_device: ExecutionDevice,
) -> ReconstructionNetwork:
    feature_count = features.shape[1]
    cuda_devices = _cuda_device_indices(execution_device)
    torch_random = adapt_library_result(torch.random, ModuleType, _TorchRandomModule)
    with torch_random.fork_rng(devices=cuda_devices):
        torch_random.manual_seed(seed)
        model = ReconstructionNetwork(feature_count, settings).to(device)
    optimizer = adapt_library_result(
        torch.optim.Adam(
            model.parameters(), lr=settings.autoencoder_learning_rate, weight_decay=0.0
        ),
        torch.optim.Optimizer,
        TorchOptimizer,
    )
    training = torch.tensor(features, dtype=torch.float32, device=device)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    model.train()
    for _ in range(settings.autoencoder_epochs):
        rows = torch.randperm(training.shape[0], generator=generator, device=device)
        for start in range(0, training.shape[0], settings.autoencoder_batch_size):
            batch = training[rows[start : start + settings.autoencoder_batch_size]]
            optimizer.zero_grad()
            loss = adapt_library_result(
                torch.mean(torch.square(model(batch) - batch)), Tensor, _TorchLoss
            )
            loss.backward()
            optimizer.step()
    model.eval()
    return model


def _reconstruction_scores(
    features: FloatMatrix, model: ReconstructionNetwork, device: torch.device
) -> FloatMatrix:
    model.eval()
    with torch.no_grad():
        values = torch.tensor(features, dtype=torch.float32, device=device)
        residuals = model(values) - values
        errors = torch.mean(torch.square(residuals), dim=1)
    return np.asarray(errors.cpu().numpy(), dtype=np.float64)


def _fit_histogram(features: FloatMatrix, bins: DetectorCount) -> tuple[FloatMatrix, FloatMatrix]:
    edges: list[FloatArray] = []
    probabilities: list[FloatArray] = []
    for column_index in range(features.shape[1]):
        column: FloatArray = features[:, column_index]
        minimum = np.min(column).item()
        maximum = np.max(column).item()
        if minimum == maximum:
            minimum -= 0.5
            maximum += 0.5
        feature_edges = np.linspace(minimum, maximum, bins + 1, dtype=np.float64)
        positions = np.searchsorted(feature_edges, column, side="right")
        indexes = np.minimum(positions - 1, bins - 1)
        counts = np.bincount(indexes, minlength=bins)
        edges.append(feature_edges)
        probabilities.append((counts.astype(np.float64) + 1.0) / (column.size + bins))
    return np.stack(edges), np.stack(probabilities)


def _histogram_scores(
    features: FloatMatrix, edges: FloatMatrix, probabilities: FloatMatrix
) -> FloatMatrix:
    scores = np.zeros(features.shape[0], dtype=np.float64)
    for feature_index, (feature_edges, feature_probabilities) in enumerate(
        zip(edges, probabilities, strict=True)
    ):
        indices: IntMatrix = np.digitize(
            features[:, feature_index], feature_edges[1:-1], right=False
        )
        indices = np.clip(indices, 0, feature_probabilities.size - 1)
        probabilities_for_rows = feature_probabilities[indices].copy()
        outside_training_range = (features[:, feature_index] < feature_edges[0]) | (
            features[:, feature_index] > feature_edges[-1]
        )
        probabilities_for_rows[outside_training_range] = np.min(feature_probabilities)
        scores += -np.log(probabilities_for_rows)
    return scores


def _fit_gaussian(
    features: FloatMatrix, regularization: CovarianceRegularization
) -> tuple[FloatMatrix, FloatMatrix]:
    mean = np.mean(features, axis=0, keepdims=True)
    covariance = np.atleast_2d(np.cov(features, rowvar=False, ddof=1))
    covariance += np.eye(features.shape[1], dtype=np.float64) * regularization
    precision = np.linalg.pinv(covariance, hermitian=True)
    return np.asarray(mean, dtype=np.float64), np.asarray(precision, dtype=np.float64)


def fit_local_ensemble(
    benign_training: FloatMatrix,
    settings: LocalDetectorSettings,
    seed: LearningSeed,
    execution_device: ExecutionDevice,
) -> LocalDetectorEnsemble:
    _validate_features(benign_training)
    maximum_neighbors = max(settings.lof_neighbors, settings.nearest_neighbors)
    if benign_training.shape[0] <= maximum_neighbors:
        raise ValueError("Benign training rows must exceed the configured neighborhood sizes")
    device = resolve_torch_device(execution_device)
    sample = _bounded_fit_sample(
        benign_training,
        settings.detector_sample_limit,
        LocalDetectorSeed(seed),
    )
    if sample.shape[0] <= maximum_neighbors:
        raise ValueError("Detector training sample is too small for configured neighborhoods")
    pca = PCA(
        n_components=settings.pca_explained_variance,
        svd_solver=PcaSolver.FULL,
        random_state=seed,
    )
    pca.fit(benign_training)
    isolation_forest = IsolationForest(
        n_estimators=settings.isolation_forest_trees, random_state=seed, n_jobs=1
    )
    isolation_forest.fit(sample)
    lof = LocalOutlierFactor(n_neighbors=settings.lof_neighbors, novelty=True, n_jobs=1)
    lof.fit(sample)
    histogram_edges, histogram_probabilities = _fit_histogram(
        benign_training, settings.histogram_bins
    )
    gaussian_mean, gaussian_precision = _fit_gaussian(
        benign_training, settings.covariance_regularization
    )
    autoencoder = _train_autoencoder(
        benign_training, settings, LocalDetectorSeed(seed), device, execution_device
    )
    return LocalDetectorEnsemble(
        benign_training.shape[1],
        device,
        autoencoder,
        pca,
        isolation_forest,
        lof,
        settings.nearest_neighbors,
        histogram_edges,
        histogram_probabilities,
        gaussian_mean,
        gaussian_precision,
    )


@dataclass(frozen=True, slots=True)
class LocalBankFeatures:
    training: FloatMatrix
    reference: FloatMatrix
    calibration: FloatMatrix
    query: FloatMatrix


@dataclass(frozen=True, slots=True)
class LocalBankResponses:
    training: FloatMatrix
    reference: FloatMatrix
    calibration: FloatMatrix
    query: FloatMatrix


@dataclass(frozen=True, slots=True)
class LocalBankDetectorScores:
    reference: DetectorScoreMatrix
    calibration: DetectorScoreMatrix
    query: DetectorScoreMatrix


@dataclass(frozen=True, slots=True)
class LocalBankEvidence:
    anchor_calibration: FloatArray
    anchor_query: FloatArray
    expert_kinds: tuple[LocalExpertKind, ...]
    calibration: FloatMatrix
    query: FloatMatrix

    def __post_init__(self) -> None:
        if (
            self.calibration.shape[0] != len(self.expert_kinds)
            or self.query.shape[0] != len(self.expert_kinds)
            or self.anchor_calibration.shape != (self.calibration.shape[1],)
            or self.anchor_query.shape != (self.query.shape[1],)
            or not np.isfinite(self.calibration).all()
            or not np.isfinite(self.query).all()
        ):
            raise ValueError("Local bank evidence must be finite and expert-aligned")

    def without_family(self, family: LocalExpertFamily) -> "LocalBankEvidence":
        kept = [index for index, kind in enumerate(self.expert_kinds) if kind.family is not family]
        return LocalBankEvidence(
            self.anchor_calibration,
            self.anchor_query,
            tuple(self.expert_kinds[index] for index in kept),
            self.calibration[kept],
            self.query[kept],
        )


def _response_scores(
    training: FloatMatrix,
    queries: tuple[FloatMatrix, ...],
    settings: LocalBankSettings,
    seed: LocalDetectorSeed,
) -> tuple[FloatMatrix, ...]:
    forest = IsolationForest(
        n_estimators=settings.response_forest_trees, random_state=seed, n_jobs=1
    )
    forest.fit(training)
    scored = tuple(np.stack((-forest.score_samples(rows),)) for rows in queries)
    if not all(np.isfinite(values).all() for values in scored):
        raise LocalDetectorError("Response experts produced non-finite scores")
    return scored


def _evidence_rows(
    reference: FloatMatrix, calibration: FloatMatrix, query: FloatMatrix
) -> tuple[FloatMatrix, FloatMatrix]:
    return (
        np.stack(
            [
                target_relative_evidence(reference[row], calibration[row])
                for row in range(reference.shape[0])
            ]
        ),
        np.stack(
            [
                target_relative_evidence(reference[row], query[row])
                for row in range(reference.shape[0])
            ]
        ),
    )


def build_local_bank(
    detectors: LocalBankDetectorScores,
    features: LocalBankFeatures,
    responses: LocalBankResponses,
    settings: LocalBankSettings,
    seed: LocalDetectorSeed,
) -> LocalBankEvidence:
    calibration_evidence = local_detector_evidence(detectors.reference, detectors.calibration)
    query_evidence = local_detector_evidence(detectors.reference, detectors.query)
    anchor_rows = anchor_detector_rows()
    ecod = fit_ecod_sum(features.training)
    pair = fit_pair_cooccurrence(
        features.training,
        settings.pair_count,
        settings.pair_bins,
        settings.pair_smoothing,
        seed,
    )
    response_reference, response_calibration, response_query = _response_scores(
        responses.training,
        (responses.reference, responses.calibration, responses.query),
        settings,
        seed,
    )
    response_cal, response_qry = _evidence_rows(
        response_reference, response_calibration, response_query
    )
    rarity_reference = np.stack((ecod.scores(features.reference), pair.scores(features.reference)))
    rarity_cal, rarity_qry = _evidence_rows(
        rarity_reference,
        np.stack((ecod.scores(features.calibration), pair.scores(features.calibration))),
        np.stack((ecod.scores(features.query), pair.scores(features.query))),
    )
    by_kind_cal: dict[LocalExpertKind, FloatArray] = {}
    by_kind_query: dict[LocalExpertKind, FloatArray] = {}
    for kind in LocalExpertKind:
        detector = kind.detector
        if detector is not None:
            by_kind_cal[kind] = calibration_evidence[detector.row]
            by_kind_query[kind] = query_evidence[detector.row]
    response_kinds = tuple(
        kind for kind in LocalExpertKind if kind.family is LocalExpertFamily.RESPONSE
    )
    for index, kind in enumerate(response_kinds):
        by_kind_cal[kind] = response_cal[index]
        by_kind_query[kind] = response_qry[index]
    by_kind_cal[LocalExpertKind.RARITY_ECOD_SUM] = rarity_cal[0]
    by_kind_query[LocalExpertKind.RARITY_ECOD_SUM] = rarity_qry[0]
    by_kind_cal[LocalExpertKind.RARITY_PAIR_COOCCURRENCE] = rarity_cal[1]
    by_kind_query[LocalExpertKind.RARITY_PAIR_COOCCURRENCE] = rarity_qry[1]
    kinds = tuple(LocalExpertKind)
    return LocalBankEvidence(
        np.mean(calibration_evidence[anchor_rows], axis=0),
        np.mean(query_evidence[anchor_rows], axis=0),
        kinds,
        np.stack([by_kind_cal[kind] for kind in kinds]),
        np.stack([by_kind_query[kind] for kind in kinds]),
    )


def margin_response_bank(
    bank: LocalBankEvidence,
    responses: LocalBankResponses,
    settings: LocalBankSettings,
    seed: LocalDetectorSeed,
) -> LocalBankEvidence:
    reference, calibration_scores, query_scores = _response_scores(
        responses.training,
        (responses.reference, responses.calibration, responses.query),
        settings,
        seed,
    )
    response_cal, response_query = _evidence_rows(reference, calibration_scores, query_scores)
    indices = [
        index
        for index, kind in enumerate(bank.expert_kinds)
        if kind.family is LocalExpertFamily.RESPONSE
    ]
    calibration = np.array(bank.calibration, copy=True)
    query = np.array(bank.query, copy=True)
    calibration[indices] = response_cal
    query[indices] = response_query
    return LocalBankEvidence(
        bank.anchor_calibration, bank.anchor_query, bank.expert_kinds, calibration, query
    )
