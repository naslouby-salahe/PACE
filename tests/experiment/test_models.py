from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch
from sklearn.ensemble import IsolationForest

from pace.config import LocalBankSettings, LocalDetectorSettings, load_configuration
from pace.core.evidence import target_relative_evidence
from pace.experiment import models
from pace.experiment.models import (
    LocalBankDetectorScores,
    LocalBankFeatures,
    LocalBankResponses,
    build_local_bank,
    fit_local_ensemble,
    margin_response_bank,
    resolve_torch_device,
    select_local_training_rows,
)
from pace.types import (
    DetectorSampleLimit,
    DetectorScoreMatrix,
    ExecutionDevice,
    ExecutionDeviceUnavailableError,
    LocalDetectorError,
    LocalDetectorKind,
    LocalDetectorSeed,
    LocalExpertFamily,
    LocalExpertKind,
)
from tests.support import REPOSITORY_ROOT

SEED = LocalDetectorSeed(41)
SETTINGS = LocalBankSettings(
    response_forest_trees=100,
    pair_count=150,
    pair_bins=8,
    pair_smoothing=0.5,
    anchor_share=0.5,
)
PEERS = 5


def _small_settings(
    sample_limit: DetectorSampleLimit | None = None,
) -> LocalDetectorSettings:
    configuration = load_configuration(REPOSITORY_ROOT / "configs" / "default.yaml", Path.cwd())
    selected_sample_limit = sample_limit
    if selected_sample_limit is None:
        selected_sample_limit = configuration.scientific.local_detectors.detector_sample_limit
    return configuration.scientific.local_detectors.model_copy(
        update={
            "autoencoder_epochs": 2,
            "detector_sample_limit": selected_sample_limit,
        }
    )


def test_local_ensemble_fits_all_seven_benign_only_detectors_deterministically() -> None:
    generator = np.random.default_rng(415)
    benign_training = generator.normal(0.0, 1.0, size=(72, 4))
    benign_training[:, -1] = 0.0
    queries = np.vstack((benign_training[:8], np.full((3, 4), 5.0)))
    settings = _small_settings()
    first = fit_local_ensemble(benign_training, settings, 29, ExecutionDevice.CPU)
    repeated = fit_local_ensemble(benign_training, settings, 29, ExecutionDevice.CPU)
    scores = first.scores(queries)
    repeated_scores = repeated.scores(queries)
    assert scores.values.shape == (7, queries.shape[0])
    assert np.isfinite(scores.values).all()
    assert not scores.values.flags.writeable
    np.testing.assert_array_equal(scores.values, repeated_scores.values)
    assert float(np.mean(scores.values[:, -3:])) > float(np.mean(scores.values[:, :8]))


@pytest.mark.parametrize(
    "features",
    [
        np.empty((0, 2)),
        np.ones((25, 0)),
        np.full((25, 2), np.nan),
    ],
)
def test_local_ensemble_rejects_invalid_training_features(features: np.ndarray) -> None:
    small_settings = _small_settings()
    with pytest.raises(ValueError):
        fit_local_ensemble(features, small_settings, 2, ExecutionDevice.CPU)


def test_local_ensemble_rejects_training_sets_smaller_than_configured_neighbors() -> None:
    matrix = np.ones((20, 3))
    small_settings_input = _small_settings()
    with pytest.raises(ValueError, match="neighborhood sizes"):
        fit_local_ensemble(matrix, small_settings_input, 2, ExecutionDevice.CPU)


def test_local_ensemble_rejects_bounded_sample_smaller_than_neighborhoods() -> None:
    matrix_input = np.ones((72, 3))
    small_settings_input = _small_settings(20)
    with pytest.raises(ValueError, match="training sample is too small"):
        fit_local_ensemble(
            matrix_input,
            small_settings_input,
            2,
            ExecutionDevice.CPU,
        )


def test_local_ensemble_uses_seeded_bounded_samples() -> None:
    generator = np.random.default_rng(21)
    benign_training = generator.normal(size=(72, 3))
    queries = generator.normal(size=(8, 3))
    settings = _small_settings(40)
    first = fit_local_ensemble(benign_training, settings, 7, ExecutionDevice.CPU)
    repeated = fit_local_ensemble(benign_training, settings, 7, ExecutionDevice.CPU)
    np.testing.assert_array_equal(first.scores(queries).values, repeated.scores(queries).values)


def test_library_adapter_rejects_an_unexpected_implementation() -> None:
    with pytest.raises(LocalDetectorError, match="unexpected implementation"):
        models.adapt_library_result(
            object(),
            torch.optim.Optimizer,
            models.TorchOptimizer,
        )


def test_local_ensemble_rejects_invalid_external_detector_outputs() -> None:
    generator = np.random.default_rng(57)
    training = generator.normal(size=(72, 3))
    queries = generator.normal(size=(5, 3))
    ensemble = fit_local_ensemble(training, _small_settings(), 14, ExecutionDevice.CPU)

    with (
        patch.object(
            ensemble.pca,
            "transform",
            return_value=np.full((queries.shape[0], 2), np.nan),
        ),
        patch.object(
            ensemble.pca,
            "inverse_transform",
            return_value=np.zeros_like(queries),
        ),
        pytest.raises(LocalDetectorError, match="PCA returned invalid"),
    ):
        ensemble.scores(queries)

    with (
        patch.object(
            ensemble.isolation_forest,
            "score_samples",
            return_value=np.full(queries.shape[0], np.nan),
        ),
        pytest.raises(LocalDetectorError, match="Novelty detectors returned invalid"),
    ):
        ensemble.scores(queries)

    shape = (queries.shape[0], ensemble.neighbor_count)
    with (
        patch.object(
            ensemble.lof,
            "kneighbors",
            return_value=(np.full(shape, -1.0), np.zeros(shape, dtype=np.int64)),
        ),
        pytest.raises(LocalDetectorError, match="Nearest-neighbor detector returned invalid"),
    ):
        ensemble.scores(queries)


def test_local_training_budget_selects_the_first_chronological_rows() -> None:
    training = np.arange(30, dtype=np.float64).reshape(10, 3)
    selected = select_local_training_rows(training, 4)
    np.testing.assert_array_equal(selected, training[:4])
    assert not selected.flags.writeable


def test_local_training_budget_rejects_incomplete_chronology() -> None:
    with pytest.raises(ValueError, match="fewer rows than the local budget"):
        select_local_training_rows(np.ones((3, 2)), 4)


def test_cuda_device_is_used_when_explicitly_available() -> None:
    with (
        patch(
            "pace.experiment.models.torch.cuda.is_available",
            return_value=True,
        ),
        patch(
            "pace.experiment.models.torch.cuda.current_device",
            return_value=2,
        ),
    ):
        device = resolve_torch_device(ExecutionDevice.CUDA)
        assert device.type == "cuda"
        assert device.index == 2


def test_cuda_request_fails_explicitly_without_falling_back_to_cpu() -> None:
    features = np.ones((32, 3), dtype=np.float64)
    settings = _small_settings()
    with (
        patch(
            "pace.experiment.models.torch.cuda.is_available",
            return_value=False,
        ),
        pytest.raises(ExecutionDeviceUnavailableError),
    ):
        fit_local_ensemble(features, settings, 2, ExecutionDevice.CUDA)


def _scores(generator: np.random.Generator, rows: int) -> DetectorScoreMatrix:
    return DetectorScoreMatrix(np.abs(generator.normal(size=(len(LocalDetectorKind), rows))) + 0.1)


def _inputs(
    seed: int = 3,
) -> tuple[LocalBankDetectorScores, LocalBankFeatures, LocalBankResponses]:
    generator = np.random.default_rng(seed)
    sizes = {"training": 120, "reference": 150, "calibration": 140, "query": 160}
    features = LocalBankFeatures(*(generator.normal(size=(rows, 9)) for rows in sizes.values()))
    responses = LocalBankResponses(
        *(np.abs(generator.normal(size=(rows, PEERS))) for rows in sizes.values())
    )
    detectors = LocalBankDetectorScores(
        _scores(generator, 150), _scores(generator, 140), _scores(generator, 160)
    )
    return detectors, features, responses


def test_bank_contains_one_row_per_expert_kind_and_a_finite_anchor() -> None:
    detectors, features, responses = _inputs()
    bank = build_local_bank(detectors, features, responses, SETTINGS, SEED)
    assert bank.expert_kinds == tuple(LocalExpertKind)
    assert bank.calibration.shape == (len(LocalExpertKind), 140)
    assert bank.query.shape == (len(LocalExpertKind), 160)
    assert bank.anchor_calibration.shape == (140,)
    assert np.isfinite(bank.calibration).all()
    assert (bank.query >= 0.0).all()


def test_anchor_is_the_mean_detector_evidence_without_the_autoencoder() -> None:
    detectors, features, responses = _inputs()
    bank = build_local_bank(detectors, features, responses, SETTINGS, SEED)
    rows = [kind.row for kind in LocalDetectorKind if kind is not LocalDetectorKind.AUTOENCODER]
    expected = np.mean(
        [
            target_relative_evidence(detectors.reference.values[row], detectors.query.values[row])
            for row in rows
        ],
        axis=0,
    )
    np.testing.assert_allclose(bank.anchor_query, expected)


def test_feature_experts_reuse_the_detector_evidence() -> None:
    detectors, features, responses = _inputs()
    bank = build_local_bank(detectors, features, responses, SETTINGS, SEED)
    index = bank.expert_kinds.index(LocalExpertKind.FEATURE_HISTOGRAM)
    row = LocalDetectorKind.HISTOGRAM_BASED_OUTLIER_SCORE.row
    np.testing.assert_allclose(
        bank.calibration[index],
        target_relative_evidence(
            detectors.reference.values[row], detectors.calibration.values[row]
        ),
    )


def test_response_expert_matches_reference_scorer_on_the_response_space() -> None:
    detectors, features, responses = _inputs()
    bank = build_local_bank(detectors, features, responses, SETTINGS, SEED)
    forest = IsolationForest(n_estimators=100, random_state=SEED, n_jobs=1).fit(responses.training)
    expected = target_relative_evidence(
        -forest.score_samples(responses.reference), -forest.score_samples(responses.query)
    )
    np.testing.assert_allclose(
        bank.query[bank.expert_kinds.index(LocalExpertKind.RESPONSE_ISOLATION_FOREST)],
        expected,
        rtol=1e-6,
        atol=1e-9,
    )


def test_bank_is_deterministic_for_a_seed_and_changes_with_it() -> None:
    detectors, features, responses = _inputs()
    first = build_local_bank(detectors, features, responses, SETTINGS, SEED)
    again = build_local_bank(detectors, features, responses, SETTINGS, SEED)
    other = build_local_bank(detectors, features, responses, SETTINGS, LocalDetectorSeed(42))
    np.testing.assert_array_equal(first.query, again.query)
    assert not np.array_equal(first.query, other.query)


def test_fitted_experts_never_see_calibration_or_query_rows() -> None:
    detectors, features, responses = _inputs()
    baseline = build_local_bank(detectors, features, responses, SETTINGS, SEED)
    shifted_features = LocalBankFeatures(
        features.training, features.reference, features.calibration + 5.0, features.query - 5.0
    )
    shifted_responses = LocalBankResponses(
        responses.training, responses.reference, responses.calibration + 3.0, responses.query + 3.0
    )
    shifted = build_local_bank(detectors, shifted_features, shifted_responses, SETTINGS, SEED)
    reference_only = (
        LocalExpertKind.RESPONSE_ISOLATION_FOREST,
        LocalExpertKind.RARITY_ECOD_SUM,
    )
    assert np.isfinite(shifted.query).all()
    assert not np.allclose(baseline.query, shifted.query)
    unaffected = baseline.expert_kinds.index(LocalExpertKind.FEATURE_GAUSSIAN)
    np.testing.assert_array_equal(baseline.query[unaffected], shifted.query[unaffected])
    assert all(kind in baseline.expert_kinds for kind in reference_only)


def test_training_rows_change_every_fitted_expert_but_not_detector_evidence() -> None:
    detectors, features, responses = _inputs()
    baseline = build_local_bank(detectors, features, responses, SETTINGS, SEED)
    altered = build_local_bank(
        detectors,
        LocalBankFeatures(
            features.training * 1.5, features.reference, features.calibration, features.query
        ),
        LocalBankResponses(
            responses.training + 1.0, responses.reference, responses.calibration, responses.query
        ),
        SETTINGS,
        SEED,
    )
    np.testing.assert_array_equal(baseline.anchor_query, altered.anchor_query)
    for kind in LocalExpertKind:
        index = baseline.expert_kinds.index(kind)
        same = np.array_equal(baseline.query[index], altered.query[index])
        assert same is (kind.detector is not None)


def test_removing_a_family_drops_exactly_its_experts() -> None:
    detectors, features, responses = _inputs()
    bank = build_local_bank(detectors, features, responses, SETTINGS, SEED)
    for family in LocalExpertFamily:
        reduced = bank.without_family(family)
        assert all(kind.family is not family for kind in reduced.expert_kinds)
        assert reduced.calibration.shape[0] == sum(
            kind.family is not family for kind in LocalExpertKind
        )


def test_margin_variant_replaces_only_response_rows() -> None:
    detectors, features, responses = _inputs()
    bank = build_local_bank(detectors, features, responses, SETTINGS, SEED)
    margins = LocalBankResponses(
        *(
            np.asarray(np.sqrt(rows) * 3.0)
            for rows in (
                responses.training,
                responses.reference,
                responses.calibration,
                responses.query,
            )
        )
    )
    replaced = margin_response_bank(bank, margins, SETTINGS, SEED)
    for index, kind in enumerate(bank.expert_kinds):
        same = np.array_equal(bank.query[index], replaced.query[index])
        assert same is (kind.family is not LocalExpertFamily.RESPONSE)
