from pathlib import Path

import numpy as np

from pace.config import LocalBankSettings, load_configuration
from pace.experiment.inputs import LocalEvidence, second_order_evidence
from pace.experiment.models import (
    LocalBankDetectorScores,
    LocalBankFeatures,
    LocalBankResponses,
    build_local_bank,
    fit_local_ensemble,
)
from pace.types import (
    AnchorBudgetPolicy,
    BankDesign,
    DetectorScoreMatrix,
    ExecutionDevice,
    LocalDetectorKind,
    LocalDetectorSeed,
    LocalExpertFamily,
    LocalExpertKind,
)
from tests.support import REPOSITORY_ROOT

SETTINGS = LocalBankSettings(
    response_forest_trees=20,
    pair_count=40,
    pair_bins=8,
    pair_smoothing=0.5,
    anchor_share=0.5,
    anchor_budget_policy=AnchorBudgetPolicy.SHARE_CONSERVING,
    design=BankDesign.PER_EXPERT,
)


def _local_evidence() -> LocalEvidence:
    generator = np.random.default_rng(9)
    sizes = {"training": 120, "reference": 150, "calibration": 140, "query": 160}
    features = LocalBankFeatures(*(generator.normal(size=(rows, 9)) for rows in sizes.values()))
    responses = LocalBankResponses(
        *(np.abs(generator.normal(size=(rows, 5))) for rows in sizes.values())
    )

    def scores(rows: int) -> DetectorScoreMatrix:
        return DetectorScoreMatrix(
            np.abs(generator.normal(size=(len(LocalDetectorKind), rows))) + 0.1
        )

    bank = build_local_bank(
        LocalBankDetectorScores(scores(150), scores(140), scores(160)),
        features,
        responses,
        SETTINGS,
        LocalDetectorSeed(5),
    )
    return LocalEvidence(5, bank.anchor_calibration, bank.anchor_query, bank)


def test_dropping_the_gaussian_expert_removes_exactly_that_row() -> None:
    local = _local_evidence()
    tiered = local.tiered(design=BankDesign.NO_GAUSSIAN)
    assert tiered.bank_calibration.shape[0] == len(LocalExpertKind) - 1
    kept = [kind for kind in LocalExpertKind if kind is not LocalExpertKind.FEATURE_GAUSSIAN]
    expected = np.stack(
        [local.bank.calibration[local.bank.expert_kinds.index(kind)] for kind in kept]
    )
    np.testing.assert_allclose(tiered.bank_calibration, expected)


def test_family_budgets_give_each_family_an_equal_share_split_within_the_family() -> None:
    local = _local_evidence()
    shares = local.tiered(design=BankDesign.FAMILY_BUDGETS).bank_shares
    assert shares is not None
    assert np.isclose(sum(shares), 1.0)
    by_family: dict[LocalExpertFamily, float] = {}
    for kind, share in zip(local.bank.expert_kinds, shares, strict=True):
        by_family[kind.family] = by_family.get(kind.family, 0.0) + share
    assert all(np.isclose(total, 1.0 / 3.0) for total in by_family.values())
    feature = shares[local.bank.expert_kinds.index(LocalExpertKind.FEATURE_GAUSSIAN)]
    response = shares[local.bank.expert_kinds.index(LocalExpertKind.RESPONSE_ISOLATION_FOREST)]
    assert np.isclose(feature, 1.0 / 9.0)
    assert np.isclose(response, 1.0 / 3.0)


def test_second_order_evidence_is_the_second_largest_expert_value_per_row() -> None:
    evidence = np.array([[1.0, 9.0, 3.0], [4.0, 2.0, 8.0], [2.0, 5.0, 1.0]])
    np.testing.assert_allclose(second_order_evidence(evidence), [[2.0, 5.0, 3.0]])
    np.testing.assert_allclose(second_order_evidence(evidence[:1]), evidence[:1])
    local = _local_evidence()
    reduced = local.tiered(design=BankDesign.SECOND_ORDER)
    assert reduced.bank_calibration.shape == (1, local.bank.calibration.shape[1])


def test_second_order_aggregation_ignores_a_single_misbehaving_expert() -> None:
    generator = np.random.default_rng(2)
    calibration = generator.exponential(size=(6, 4000))
    query = generator.exponential(size=(6, 4000))
    query[2] += 50.0
    single = np.max(query, axis=0) > np.quantile(np.max(calibration, axis=0), 0.99)
    second = second_order_evidence(query)[0] > np.quantile(
        second_order_evidence(calibration)[0], 0.99
    )
    assert single.mean() > 0.9
    assert second.mean() < 0.25


def test_shrinkage_bounds_the_gaussian_precision_that_a_near_constant_feature_inflates() -> None:
    generator = np.random.default_rng(415)
    training = generator.normal(0.0, 1.0, size=(300, 6))
    training[:, -1] = 1e-3 * generator.normal(size=300)
    configuration = load_configuration(REPOSITORY_ROOT / "configs" / "default.yaml", Path.cwd())
    settings = configuration.scientific.local_detectors.model_copy(
        update={"autoencoder_epochs": 1, "detector_sample_limit": 300}
    )
    shrunk = fit_local_ensemble(training, settings, 3, ExecutionDevice.CPU)
    ridge_covariance = np.cov(training, rowvar=False, ddof=1) + np.eye(6) * (
        settings.covariance_regularization
    )
    ridge_precision = np.linalg.pinv(ridge_covariance, hermitian=True)
    assert (
        np.linalg.eigvalsh(shrunk.gaussian_precision).max()
        < 0.1 * np.linalg.eigvalsh(ridge_precision).max()
    )
    shifted = training[:50].copy()
    shifted[:, -1] += 0.5
    centered = shifted - shrunk.gaussian_mean
    ridge_shift = np.sum((centered @ ridge_precision) * centered, axis=1)
    shrunk_shift = np.sum((centered @ shrunk.gaussian_precision) * centered, axis=1)
    assert shrunk_shift.mean() < 0.1 * ridge_shift.mean()
