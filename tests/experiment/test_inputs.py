from pathlib import Path

import numpy as np
import pytest

from pace.config import PeerTrainingSettings, load_configuration
from pace.core.evidence import FloatArray
from pace.experiment.inputs import train_peer_mlp
from tests.support import REPOSITORY_ROOT


def _training_settings() -> PeerTrainingSettings:
    configuration = load_configuration(REPOSITORY_ROOT / "configs" / "default.yaml", Path.cwd())
    return configuration.scientific.peer_training


def test_peer_mlp_is_seeded_class_balanced_and_returns_attack_margins() -> None:
    generator = np.random.default_rng(67)
    benign = generator.normal(-0.8, 0.5, size=(80, 3))
    attacks = generator.normal(0.8, 0.5, size=(12, 3))
    settings = _training_settings()
    first = train_peer_mlp(benign, attacks, 17, settings)
    repeated = train_peer_mlp(benign, attacks, 17, settings)
    benign_margins = first.margins(benign)
    attack_margins = first.margins(attacks)
    assert benign_margins.shape == (80,)
    assert attack_margins.shape == (12,)
    assert float(np.mean(attack_margins)) > float(np.mean(benign_margins))
    for first_layer, repeated_layer in zip(first.layers, repeated.layers, strict=True):
        np.testing.assert_array_equal(first_layer.weights, repeated_layer.weights)
        np.testing.assert_array_equal(first_layer.bias, repeated_layer.bias)
        assert not first_layer.weights.flags.writeable
        assert not first_layer.bias.flags.writeable


@pytest.mark.parametrize(
    ("benign", "attacks"),
    [
        (np.empty((0, 2)), np.ones((1, 2))),
        (np.ones((2, 1)), np.ones((2, 2))),
        (np.asarray([[np.nan]]), np.ones((1, 1))),
    ],
)
def test_peer_training_rejects_invalid_feature_matrices(
    benign: FloatArray, attacks: FloatArray
) -> None:
    training_settings = _training_settings()
    with pytest.raises(ValueError):
        train_peer_mlp(benign, attacks, 1, training_settings)


def test_peer_inference_rejects_invalid_feature_matrices() -> None:
    model = train_peer_mlp(
        np.asarray([[-1.0], [-0.5]]),
        np.asarray([[0.5], [1.0]]),
        3,
        _training_settings(),
    )
    with pytest.raises(ValueError):
        model.margins(np.empty((0, 1)))
    with pytest.raises(ValueError):
        model.margins(np.ones((2, 2)))
    with pytest.raises(ValueError):
        model.margins(np.asarray([[np.inf]]))
