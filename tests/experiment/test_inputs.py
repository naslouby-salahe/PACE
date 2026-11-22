from pathlib import Path

import numpy as np
import pytest

from pace.config import PeerTrainingSettings, load_configuration
from pace.core.evidence import FloatArray
from pace.experiment.inputs import build_target_frame, train_peer_mlp
from pace.experiment.runner import prepare_clients
from pace.types import DatasetValidationError, EvaluationRole
from tests.support import REPOSITORY_ROOT, campaign_configuration, campaign_devices


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


def test_calibration_split_role_calibrates_on_the_first_half_and_evaluates_on_the_second() -> None:
    configuration = campaign_configuration()
    device = prepare_clients(campaign_devices(), configuration)[0]
    frame = build_target_frame(device, EvaluationRole.DEVELOPMENT_CALIBRATION_SPLIT)
    half = device.calibration.shape[0] // 2
    np.testing.assert_array_equal(frame.calibration, device.calibration[:half])
    assert frame.benign_count == device.calibration.shape[0] - half
    np.testing.assert_array_equal(frame.query[: frame.benign_count], device.calibration[half:])
    assert frame.held_out == device.development
    assert frame.query.shape[0] == frame.benign_count + sum(
        features.shape[0] for _, features in device.development
    )


def test_held_out_role_uses_the_test_partition_and_held_out_attacks() -> None:
    device = prepare_clients(campaign_devices(), campaign_configuration())[0]
    frame = build_target_frame(device)
    np.testing.assert_array_equal(frame.calibration, device.calibration)
    assert frame.benign_count == device.test.shape[0]
    assert frame.held_out == device.held_out


def test_development_role_needs_training_rows_beyond_the_local_budget() -> None:
    device = prepare_clients(campaign_devices(), campaign_configuration())[0]
    with pytest.raises(DatasetValidationError, match="beyond the local budget"):
        build_target_frame(device, EvaluationRole.DEVELOPMENT, device.training.shape[0])
    frame = build_target_frame(device, EvaluationRole.DEVELOPMENT, 5)
    assert frame.benign_count == device.training.shape[0] - 5
