import math
from dataclasses import dataclass

import numpy as np
from scipy.cluster.hierarchy import ClusterNode, linkage, to_tree
from scipy.special import expit

from pace.config import KitNetSettings
from pace.types import (
    FeatureCount,
    FloatArray,
    FloatMatrix,
    HiddenRatio,
    KitNetClusterSize,
    LearningRate,
    LearningSeed,
    NumericGuard,
    RowIndexArray,
)


@dataclass(slots=True)
class _Autoencoder:
    weights: FloatMatrix
    hidden_bias: FloatArray
    visible_bias: FloatArray
    minimum: FloatArray
    maximum: FloatArray

    @classmethod
    def initial(
        cls, visible: FeatureCount, ratio: HiddenRatio, generator: np.random.Generator
    ) -> "_Autoencoder":
        hidden = math.ceil(visible * ratio)
        bound = 1.0 / visible
        return cls(
            generator.uniform(-bound, bound, (visible, hidden)),
            np.zeros(hidden),
            np.zeros(visible),
            np.full(visible, np.inf),
            np.full(visible, -np.inf),
        )

    def _normalise(self, values: FloatMatrix) -> FloatMatrix:
        return (values - self.minimum) / (
            self.maximum - self.minimum + NumericGuard.NORMALISATION.guard
        )

    def _reconstruct(self, normalised: FloatMatrix) -> FloatMatrix:
        hidden = expit(normalised @ self.weights + self.hidden_bias)
        return np.asarray(expit(hidden @ self.weights.T + self.visible_bias), dtype=np.float64)

    def train(self, row: FloatArray, rate: LearningRate) -> FloatArray:
        self.maximum = np.maximum(self.maximum, row)
        self.minimum = np.minimum(self.minimum, row)
        normalised = self._normalise(row)
        hidden = expit(normalised @ self.weights + self.hidden_bias)
        rebuilt = expit(hidden @ self.weights.T + self.visible_bias)
        visible_error = normalised - rebuilt
        hidden_error = (visible_error @ self.weights) * hidden * (1.0 - hidden)
        self.weights += rate * (
            np.outer(normalised, hidden_error) + np.outer(visible_error, hidden)
        )
        self.hidden_bias += rate * np.mean(hidden_error)
        self.visible_bias += rate * np.mean(visible_error)
        return np.sqrt(np.mean(np.square(visible_error)))

    def scores(self, rows: FloatMatrix) -> FloatArray:
        normalised = self._normalise(rows)
        return np.sqrt(np.mean(np.square(normalised - self._reconstruct(normalised)), axis=1))


def _correlation_distance(training: FloatMatrix) -> FloatMatrix:
    count = training.shape[1]
    linear_sum = np.zeros(count)
    residual_square = np.zeros(count)
    partial = np.zeros((count, count))
    for step, row in enumerate(training, start=1):
        linear_sum += row
        residual = row - linear_sum / step
        residual_square += np.square(residual)
        partial += np.outer(residual, residual)
    scale = np.outer(np.sqrt(residual_square), np.sqrt(residual_square))
    scale[np.equal(scale, 0.0)] = NumericGuard.ZERO_VARIANCE.guard
    return np.maximum(1.0 - partial / scale, 0.0)


def _break_clusters(node: ClusterNode | None, largest: KitNetClusterSize) -> list[RowIndexArray]:
    if node is None:
        return []
    if node.get_count() <= largest:
        return [np.asarray(node.pre_order(), dtype=np.int64)]
    return _break_clusters(node.get_left(), largest) + _break_clusters(node.get_right(), largest)


def feature_map(training: FloatMatrix, largest: KitNetClusterSize) -> tuple[RowIndexArray, ...]:
    count = training.shape[1]
    if count == 1:
        return (np.zeros(1, dtype=np.int64),)
    distance = _correlation_distance(training)
    tree = to_tree(linkage(distance[np.triu_indices(count, 1)]))
    bounded = min(max(largest, 1), count)
    return tuple(np.asarray(cluster, dtype=np.int64) for cluster in _break_clusters(tree, bounded))


@dataclass(frozen=True, slots=True)
class KitNetModel:
    mapping: tuple[RowIndexArray, ...]
    ensemble: tuple[_Autoencoder, ...]
    output: _Autoencoder

    def scores(self, features: FloatMatrix) -> FloatArray:
        layer = np.stack(
            [
                encoder.scores(features[:, columns])
                for encoder, columns in zip(self.ensemble, self.mapping, strict=True)
            ],
            axis=1,
        )
        return self.output.scores(layer)


def fit_kitnet(training: FloatMatrix, settings: KitNetSettings, seed: LearningSeed) -> KitNetModel:
    if training.ndim != 2 or training.shape[0] < 2 or not np.isfinite(training).all():
        raise ValueError("KitNET needs at least two finite training rows")
    generator = np.random.default_rng(seed)
    mapping = feature_map(training, settings.max_autoencoder_size)
    ensemble = tuple(
        _Autoencoder.initial(len(columns), settings.hidden_ratio, generator) for columns in mapping
    )
    output = _Autoencoder.initial(len(mapping), settings.hidden_ratio, generator)
    for _ in range(settings.epochs):
        for row in training:
            layer = np.asarray(
                [
                    encoder.train(row[columns], settings.learning_rate)
                    for encoder, columns in zip(ensemble, mapping, strict=True)
                ]
            )
            output.train(layer, settings.learning_rate)
    return KitNetModel(mapping, ensemble, output)
