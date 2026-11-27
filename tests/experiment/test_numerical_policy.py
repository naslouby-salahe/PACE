import os
import subprocess
import sys
from unittest.mock import Mock

import numpy as np
import pytest
import torch

import pace.experiment.implementation as implementation
from pace.experiment.implementation import require_numerical_policy
from pace.experiment.models import fit_local_ensemble
from pace.types import ExecutionDevice, LocalDetectorSeed, NumericalPolicyError
from tests.support import campaign_configuration

THREAD_VARIABLES = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")


def _environment(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    for variable in THREAD_VARIABLES:
        monkeypatch.setenv(variable, value)


def _loaded_pool(monkeypatch: pytest.MonkeyPatch, threads: int) -> None:
    pool = Mock(internal_api="openblas", filepath="/lib/libopenblas.so", num_threads=threads)
    controller = Mock(lib_controllers=[pool])
    monkeypatch.setattr(implementation, "ThreadpoolController", Mock(return_value=controller))


def test_matching_environment_and_torch_threads_satisfy_the_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _environment(monkeypatch, "3")
    monkeypatch.setattr(torch, "get_num_threads", lambda: 3)
    monkeypatch.setattr(torch, "get_num_interop_threads", lambda: 3)
    _loaded_pool(monkeypatch, 3)
    require_numerical_policy(3)


def test_a_mismatched_library_environment_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    _environment(monkeypatch, "2")
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "8")
    monkeypatch.setattr(torch, "get_num_threads", lambda: 2)
    monkeypatch.setattr(torch, "get_num_interop_threads", lambda: 2)
    with pytest.raises(NumericalPolicyError, match="OPENBLAS_NUM_THREADS"):
        require_numerical_policy(2)


def test_a_mismatched_torch_thread_count_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    _environment(monkeypatch, "1")
    monkeypatch.setattr(torch, "get_num_threads", lambda: 4)
    monkeypatch.setattr(torch, "get_num_interop_threads", lambda: 1)
    with pytest.raises(NumericalPolicyError, match="PyTorch reports 4 intra-operation"):
        require_numerical_policy(1)


def test_a_mismatched_inter_operation_thread_count_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _environment(monkeypatch, "1")
    monkeypatch.setattr(torch, "get_num_threads", lambda: 1)
    monkeypatch.setattr(torch, "get_num_interop_threads", lambda: 7)
    with pytest.raises(NumericalPolicyError, match="7 inter-operation"):
        require_numerical_policy(1)


def test_a_loaded_pool_with_another_thread_count_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _environment(monkeypatch, "1")
    monkeypatch.setattr(torch, "get_num_threads", lambda: 1)
    monkeypatch.setattr(torch, "get_num_interop_threads", lambda: 1)
    _loaded_pool(monkeypatch, 8)
    with pytest.raises(NumericalPolicyError, match=r"openblas in libopenblas.so \(8\)"):
        require_numerical_policy(1)


@pytest.mark.parametrize("threads", ["1", "2"])
def test_a_fresh_process_satisfies_the_policy_after_pace_applies_the_thread_environment(
    threads: str,
) -> None:
    program = (
        "import os, pace, torch; "
        "from pace.experiment.implementation import require_numerical_policy; "
        f"require_numerical_policy({threads}); "
        "print(os.environ['OMP_NUM_THREADS'], os.environ['MKL_NUM_THREADS'], "
        "os.environ['OPENBLAS_NUM_THREADS'], torch.get_num_threads(), "
        "torch.get_num_interop_threads())"
    )
    environment = {key: value for key, value in os.environ.items() if key not in THREAD_VARIABLES}
    environment.update({variable: threads for variable in THREAD_VARIABLES})
    output = subprocess.run(
        [sys.executable, "-c", program],
        capture_output=True,
        text=True,
        check=True,
        env=environment,
    ).stdout.split()
    assert output == [threads] * 5


def test_an_unset_thread_environment_defaults_to_one_thread_for_every_library() -> None:
    program = (
        "import os, pace; "
        "print(os.environ['OMP_NUM_THREADS'], os.environ['MKL_NUM_THREADS'], "
        "os.environ['OPENBLAS_NUM_THREADS'])"
    )
    environment = {key: value for key, value in os.environ.items() if key not in THREAD_VARIABLES}
    output = subprocess.run(
        [sys.executable, "-c", program],
        capture_output=True,
        text=True,
        check=True,
        env=environment,
    ).stdout.split()
    assert output == ["1", "1", "1"]


def test_the_runtime_environment_recorded_in_provenance_includes_the_thread_policy() -> None:
    environment = implementation._runtime_environment()  # pyright: ignore[reportPrivateUsage]
    assert {
        variable.value: setting for variable, setting in environment.thread_environment.items()
    } == {variable: os.environ[variable] for variable in THREAD_VARIABLES}
    assert environment.torch_intraop_threads == torch.get_num_threads()


def test_every_local_detector_score_is_bitwise_reproducible_at_the_policy_thread_count() -> None:
    settings = campaign_configuration().scientific.local_detectors
    generator = np.random.default_rng(11)
    training = generator.standard_normal((300, 6))
    queries = generator.standard_normal((500, 6))
    seed = LocalDetectorSeed(31)

    def scores() -> np.ndarray:
        ensemble = fit_local_ensemble(training, settings, seed, ExecutionDevice.CPU)
        return ensemble.scores(queries).values

    assert np.array_equal(scores(), scores())
