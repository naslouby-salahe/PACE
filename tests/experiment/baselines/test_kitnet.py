"""KitNET tests. The oracle below transcribes the MIT-licensed reference KitNET-py
(Copyright (c) 2017 Yisroel Mirsky; dA portions adapted from Yusuke Sugomori) line by line."""

import math

import numpy as np
import pytest
from scipy.cluster.hierarchy import linkage, to_tree
from scipy.special import expit

from pace.config import KitNetSettings
from pace.experiment.baselines.kitnet import feature_map, fit_kitnet
from pace.types import FloatArray, FloatMatrix

SETTINGS = KitNetSettings(max_autoencoder_size=4, hidden_ratio=0.75, learning_rate=0.1, epochs=1)


class _ReferenceAutoencoder:
    def __init__(self, weights: FloatMatrix) -> None:
        self.weights = weights.copy()
        self.hbias: FloatArray = np.zeros(weights.shape[1])
        self.vbias: FloatArray = np.zeros(weights.shape[0])
        self.norm_max = np.full(weights.shape[0], -np.inf)
        self.norm_min = np.full(weights.shape[0], np.inf)
        self.lr = 0.1

    def _normalise(self, x: FloatArray) -> FloatArray:
        return (x - self.norm_min) / (self.norm_max - self.norm_min + 0.0000000000000001)

    def train(self, x: FloatArray) -> float:
        self.norm_max[x > self.norm_max] = x[x > self.norm_max]
        self.norm_min[x < self.norm_min] = x[x < self.norm_min]
        x = self._normalise(x)
        y: FloatArray = expit(x @ self.weights + self.hbias)
        z: FloatArray = expit(y @ self.weights.T + self.vbias)
        l_h2 = x - z
        l_h1 = (l_h2 @ self.weights) * y * (1 - y)
        l_w = np.outer(x, l_h1) + np.outer(l_h2, y)
        self.weights += self.lr * l_w
        self.hbias += self.lr * l_h1
        self.vbias += self.lr * l_h2
        return np.sqrt(np.mean(l_h2**2)).item()

    def execute(self, x: FloatArray) -> float:
        x = self._normalise(x)
        y: FloatArray = expit(x @ self.weights + self.hbias)
        z: FloatArray = expit(y @ self.weights.T + self.vbias)
        return np.sqrt(((x - z) ** 2).mean()).item()


def _reference_clusters(training: np.ndarray, largest: int) -> list[list[int]]:
    n = training.shape[1]
    c = np.zeros(n)
    c_rs = np.zeros(n)
    partial = np.zeros((n, n))
    for count, x in enumerate(training, start=1):
        c += x
        residual = x - c / count
        c_rs += residual**2
        partial += np.outer(residual, residual)
    root = np.sqrt(c_rs)
    scale = np.outer(root, root)
    scale[scale == 0] = 1e-100
    distance = 1 - partial / scale
    distance[distance < 0] = 0
    tree = to_tree(linkage(distance[np.triu_indices(n, 1)]))

    def split(node: object) -> list[list[int]]:
        if node.count <= largest:  # type: ignore[attr-defined]
            return [node.pre_order()]  # type: ignore[attr-defined]
        return split(node.get_left()) + split(node.get_right())  # type: ignore[attr-defined]

    return split(tree)


def _data(rows: int, features: int, seed: int) -> np.ndarray:
    generator = np.random.default_rng(seed)
    base = generator.normal(size=(rows, 3))
    mixing = generator.normal(size=(3, features))
    return base @ mixing + 0.1 * generator.normal(size=(rows, features))


def test_feature_map_matches_the_reference_clustering() -> None:
    training = _data(120, 11, 1)
    expected = _reference_clusters(training, 4)
    produced = [list(cluster) for cluster in feature_map(training, 4)]
    assert produced == expected
    assert sorted(column for cluster in produced for column in cluster) == list(range(11))
    assert all(len(cluster) <= 4 for cluster in produced)


def test_single_pass_training_and_scoring_match_the_reference_line_by_line() -> None:
    training, queries = _data(150, 9, 2), _data(40, 9, 3)
    model = fit_kitnet(training, SETTINGS, 7)
    generator = np.random.default_rng(7)
    mapping = _reference_clusters(training, SETTINGS.max_autoencoder_size)
    ensemble: list[_ReferenceAutoencoder] = []
    for columns in mapping:
        bound = 1.0 / len(columns)
        hidden = math.ceil(len(columns) * 0.75)
        ensemble.append(
            _ReferenceAutoencoder(generator.uniform(-bound, bound, (len(columns), hidden)))
        )
    bound = 1.0 / len(mapping)
    output = _ReferenceAutoencoder(
        generator.uniform(-bound, bound, (len(mapping), math.ceil(len(mapping) * 0.75)))
    )
    for row in training:
        output.train(np.array([a.train(row[c]) for a, c in zip(ensemble, mapping, strict=True)]))
    expected = np.array(
        [
            output.execute(
                np.array([a.execute(row[c]) for a, c in zip(ensemble, mapping, strict=True)])
            )
            for row in queries
        ]
    )
    np.testing.assert_allclose(model.scores(queries), expected, rtol=1e-9, atol=1e-12)


def test_bias_updates_are_per_unit_not_a_shared_scalar() -> None:
    model = fit_kitnet(_data(150, 9, 2), SETTINGS, 7)
    multi_unit = [a for a in (*model.ensemble, model.output) if a.hidden_bias.size > 1]
    assert multi_unit
    for autoencoder in multi_unit:
        assert np.ptp(autoencoder.hidden_bias) > 0.0
        assert np.ptp(autoencoder.visible_bias) > 0.0


def test_scores_are_deterministic_and_seed_dependent() -> None:
    training = _data(100, 8, 4)
    queries = _data(20, 8, 5)
    first = fit_kitnet(training, SETTINGS, 3).scores(queries)
    np.testing.assert_array_equal(first, fit_kitnet(training, SETTINGS, 3).scores(queries))
    assert not np.array_equal(first, fit_kitnet(training, SETTINGS, 4).scores(queries))


def test_shifted_traffic_scores_above_benign_traffic() -> None:
    training = _data(300, 10, 6)
    model = fit_kitnet(
        training,
        KitNetSettings(max_autoencoder_size=5, hidden_ratio=0.75, learning_rate=0.1, epochs=5),
        1,
    )
    benign = model.scores(_data(200, 10, 7))
    shifted = model.scores(_data(200, 10, 8) + 4.0)
    assert np.median(shifted) > np.percentile(benign, 95)


def test_more_epochs_change_the_fit() -> None:
    training = _data(100, 8, 9)
    queries = _data(20, 8, 10)
    one = fit_kitnet(training, SETTINGS, 2).scores(queries)
    many = fit_kitnet(
        training,
        KitNetSettings(max_autoencoder_size=4, hidden_ratio=0.75, learning_rate=0.1, epochs=4),
        2,
    ).scores(queries)
    assert not np.allclose(one, many)


def test_single_feature_input_forms_one_autoencoder() -> None:
    training = _data(60, 1, 11)
    model = fit_kitnet(training, SETTINGS, 1)
    assert len(model.mapping) == 1
    assert model.scores(training).shape == (60,)


def test_invalid_training_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least two finite"):
        fit_kitnet(np.ones((1, 4)), SETTINGS, 1)
    with pytest.raises(ValueError, match="at least two finite"):
        fit_kitnet(np.full((10, 4), np.nan), SETTINGS, 1)
