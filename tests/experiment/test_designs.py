from pathlib import Path

import numpy as np

from pace.config import LocalBankSettings, load_configuration
from pace.core.decision import TieredLocalEvidence, apply_local_union_peer_rule
from pace.core.joint import apply_joint_local_peer_rule
from pace.experiment.arms import calibration_rule, designed_decisions
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
    CalibrationMode,
    CovarianceEstimator,
    DetectorScoreMatrix,
    ExecutionDevice,
    LocalDetectorKind,
    LocalDetectorSeed,
    LocalExpertFamily,
    LocalExpertKind,
    TieredCalibrationRule,
    UnionCalibration,
)
from tests.support import REPOSITORY_ROOT

SETTINGS = LocalBankSettings(
    response_forest_trees=20,
    pair_count=40,
    pair_bins=8,
    pair_smoothing=0.5,
    anchor_share=0.5,
    anchor_budget_policy=AnchorBudgetPolicy.SHARE_CONSERVING,
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


def _rule() -> TieredCalibrationRule:
    return TieredCalibrationRule(
        anchor_share=0.5,
        anchor_policy=AnchorBudgetPolicy.SHARE_CONSERVING,
        union=UnionCalibration(
            scale_cap=64.0,
            scale_iterations=20,
            mode=CalibrationMode.CROSS_FITTED,
            folds=5,
            resolution_rows=2,
        ),
        design=BankDesign.JOINT_CHANNELS,
    )


def test_joint_calibration_accounts_for_every_channel_and_hits_the_budget() -> None:
    generator = np.random.default_rng(31)
    rates: list[float] = []
    for _ in range(4):
        local = TieredLocalEvidence(
            generator.exponential(size=6000),
            generator.exponential(size=60_000),
            generator.exponential(size=(3, 6000)),
            generator.exponential(size=(3, 60_000)),
        )
        decisions = apply_joint_local_peer_rule(
            local,
            generator.exponential(size=(4, 6000)),
            generator.exponential(size=(4, 60_000)),
            250,
            0.025,
            0.025,
            _rule(),
        )
        assert decisions.anchor_budget is not None
        assert decisions.bank_budget is not None
        assert np.isclose(decisions.anchor_budget + decisions.bank_budget, 0.025)
        assert len(decisions.bank_thresholds) == 3
        assert len(decisions.expert_thresholds) == 4
        assert np.array_equal(
            decisions.combined_alerts, decisions.local_alerts | decisions.peer_alerts
        )
        rates.append(np.mean(decisions.combined_alerts).item())
    assert 0.6 * 0.05 <= float(np.mean(rates)) <= 1.4 * 0.05


def test_joint_calibration_without_a_bank_gives_the_whole_local_budget_to_the_anchor() -> None:
    generator = np.random.default_rng(4)
    local = TieredLocalEvidence(
        generator.exponential(size=3000),
        generator.exponential(size=3000),
        np.empty((0, 3000)),
        np.empty((0, 3000)),
    )
    decisions = apply_joint_local_peer_rule(
        local,
        generator.exponential(size=(3, 3000)),
        generator.exponential(size=(3, 3000)),
        250,
        0.02,
        0.03,
        _rule(),
    )
    assert decisions.anchor_budget == 0.02
    assert decisions.bank_budget is None
    assert decisions.bank_thresholds == ()


def test_shrinkage_bounds_the_gaussian_precision_that_a_near_constant_feature_inflates() -> None:
    generator = np.random.default_rng(415)
    training = generator.normal(0.0, 1.0, size=(300, 6))
    training[:, -1] = 1e-3 * generator.normal(size=300)
    configuration = load_configuration(REPOSITORY_ROOT / "configs" / "default.yaml", Path.cwd())
    base = configuration.scientific.local_detectors.model_copy(
        update={
            "autoencoder_epochs": 1,
            "detector_sample_limit": 300,
            "covariance_estimator": CovarianceEstimator.RIDGE,
        }
    )
    ridge = fit_local_ensemble(training, base, 3, ExecutionDevice.CPU)
    shrunk = fit_local_ensemble(
        training,
        base.model_copy(update={"covariance_estimator": CovarianceEstimator.SHRINKAGE}),
        3,
        ExecutionDevice.CPU,
    )
    assert (
        np.linalg.eigvalsh(shrunk.gaussian_precision).max()
        < 0.1 * np.linalg.eigvalsh(ridge.gaussian_precision).max()
    )
    shifted = training[:50].copy()
    shifted[:, -1] += 0.5
    ridge_shift = np.sum(
        ((shifted - ridge.gaussian_mean) @ ridge.gaussian_precision)
        * (shifted - ridge.gaussian_mean),
        axis=1,
    )
    shrunk_shift = np.sum(
        ((shifted - shrunk.gaussian_mean) @ shrunk.gaussian_precision)
        * (shifted - shrunk.gaussian_mean),
        axis=1,
    )
    assert shrunk_shift.mean() < 0.1 * ridge_shift.mean()


def test_anchor_alternatives_replace_only_the_anchor_and_leave_the_bank_untouched() -> None:
    local = _local_evidence()
    primary = local.tiered()
    for design in (
        BankDesign.MEDIAN_ANCHOR,
        BankDesign.TRIMMED_ANCHOR,
        BankDesign.COMPLEMENTARY_ANCHOR,
    ):
        alternative = local.tiered(design=design)
        np.testing.assert_array_equal(alternative.bank_calibration, primary.bank_calibration)
        np.testing.assert_array_equal(alternative.bank_evaluation, primary.bank_evaluation)
        assert not np.array_equal(alternative.anchor_calibration, primary.anchor_calibration)
        assert alternative.anchor_calibration.shape == primary.anchor_calibration.shape
    for design in (BankDesign.PER_EXPERT, BankDesign.NO_ANCHOR):
        np.testing.assert_array_equal(
            local.tiered(design=design).anchor_calibration, primary.anchor_calibration
        )


def test_the_no_anchor_design_gives_the_whole_local_budget_to_the_bank_union() -> None:
    local = _local_evidence()
    generator = np.random.default_rng(21)
    peer_calibration = np.abs(generator.normal(size=(4, 140)))
    peer_query = np.abs(generator.normal(size=(4, 160)))
    configuration = load_configuration(REPOSITORY_ROOT / "configs" / "default.yaml", Path.cwd())
    scientific = configuration.scientific
    rule = calibration_rule(scientific, design=BankDesign.NO_ANCHOR)
    decisions = designed_decisions(
        local.tiered(design=BankDesign.NO_ANCHOR),
        peer_calibration,
        peer_query,
        scientific,
        0.005,
        0.005,
        rule,
    )
    reference = apply_local_union_peer_rule(
        local.bank.calibration,
        local.bank.query,
        peer_calibration,
        peer_query,
        scientific.calibration_block_size,
        0.005,
        0.005,
        rule.union,
        None,
    )
    np.testing.assert_array_equal(decisions.local_alerts, reference.local_alerts)
    assert decisions.anchor_budget is None
    assert len(decisions.bank_thresholds) == len(LocalExpertKind) - 1
