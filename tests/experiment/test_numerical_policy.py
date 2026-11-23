import numpy as np
import pytest
import torch

from pace.experiment.artifacts import require_numerical_policy
from pace.experiment.models import fit_local_ensemble
from pace.types import ExecutionDevice, LocalDetectorSeed, NumericalPolicyError
from tests.support import campaign_configuration

THREAD_VARIABLES = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")


def _environment(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    for variable in THREAD_VARIABLES:
        monkeypatch.setenv(variable, value)


def test_matching_environment_and_torch_threads_satisfy_the_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _environment(monkeypatch, "3")
    monkeypatch.setattr(torch, "get_num_threads", lambda: 3)
    require_numerical_policy(3)


def test_a_mismatched_library_environment_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    _environment(monkeypatch, "2")
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "8")
    monkeypatch.setattr(torch, "get_num_threads", lambda: 2)
    with pytest.raises(NumericalPolicyError, match="OPENBLAS_NUM_THREADS"):
        require_numerical_policy(2)


def test_a_mismatched_torch_thread_count_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    _environment(monkeypatch, "1")
    monkeypatch.setattr(torch, "get_num_threads", lambda: 4)
    with pytest.raises(NumericalPolicyError, match="PyTorch reports 4"):
        require_numerical_policy(1)


def test_autoencoder_training_is_bitwise_reproducible_at_a_fixed_thread_count() -> None:
    settings = campaign_configuration().scientific.local_detectors
    features = np.random.default_rng(7).standard_normal((64, 6))
    seed = LocalDetectorSeed(31)

    def weights() -> list[torch.Tensor]:
        ensemble = fit_local_ensemble(features, settings, seed, ExecutionDevice.CPU)
        return [parameter.detach().clone() for parameter in ensemble.autoencoder.parameters()]

    first, second = weights(), weights()
    assert all(torch.equal(left, right) for left, right in zip(first, second, strict=True))
