import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from pace.core.decision import (
    TieredLocalEvidence,
    apply_bonferroni_local_peer_rule,
    apply_local_union_peer_rule,
    apply_max_fusion_rule,
    apply_protected_local_peer_rule,
    apply_tiered_local_peer_rule,
    block_max_threshold,
    calibration_block_indices,
    consecutive_blocks,
    evaluate_bonferroni_peer_union,
    evaluate_peer_union,
    expert_union_alerts,
    fit_bonferroni_peer_union,
    fit_cross_fitted_peer_union,
    fit_peer_union,
    higher_quantile,
    sorted_block_max_threshold,
    sorted_blocks,
)
from pace.types import (
    AllocationAudit,
    AnchorBudgetPolicy,
    CalibrationMode,
    FloatMatrix,
    RowIndexArray,
    TieredCalibrationRule,
    UnionCalibration,
)

UNION = UnionCalibration(scale_cap=64.0, scale_iterations=20)
RULE = TieredCalibrationRule(
    anchor_share=0.5, anchor_policy=AnchorBudgetPolicy.SHARE_CONSERVING, union=UNION
)
CROSS_FITTED_UNION = UnionCalibration(
    scale_cap=64.0, scale_iterations=20, mode=CalibrationMode.CROSS_FITTED, folds=5
)


def test_chronological_blocks_follow_the_frozen_count_rule() -> None:
    blocks = consecutive_blocks(np.arange(600, dtype=np.float64), 250)
    assert tuple(len(block) for block in blocks) == (300, 300)
    assert blocks[0][0] == 0
    assert blocks[0][-1] == 299
    assert blocks[1][0] == 300
    assert blocks[1][-1] == 599


def test_block_indices_preserve_row_order_and_handle_small_calibration_sets() -> None:
    indices = calibration_block_indices(4, 250)
    assert len(indices) == 1
    np.testing.assert_array_equal(indices[0], np.arange(4))


def test_higher_quantile_uses_higher_interpolation() -> None:
    assert higher_quantile(np.asarray([0.0, 10.0, 20.0, 30.0]), 0.5) == 20.0


def test_block_max_threshold_is_maximum_of_higher_block_quantiles() -> None:
    threshold = block_max_threshold(
        (np.asarray([1.0, 2.0, 3.0, 4.0]), np.asarray([0.0, 9.0, 10.0, 11.0])),
        0.25,
    )
    assert threshold == 11.0


def test_strict_threshold_comparison_does_not_alert_on_equality() -> None:
    alerts = expert_union_alerts(
        np.asarray([[2.0, 2.1], [3.0, 2.0]]),
        (2.0, 3.0),
    )
    np.testing.assert_array_equal(alerts, np.asarray([False, True]))


@pytest.mark.parametrize(
    ("values", "quantile"),
    [
        (np.asarray([], dtype=np.float64), 0.5),
        (np.asarray([np.nan]), 0.5),
        (np.asarray([1.0]), -0.1),
        (np.asarray([1.0]), 1.1),
    ],
)
def test_invalid_quantile_inputs_fail(values: np.ndarray, quantile: float) -> None:
    with pytest.raises(ValueError):
        higher_quantile(values, quantile)


def test_invalid_block_inputs_fail() -> None:
    with pytest.raises(ValueError):
        consecutive_blocks(np.asarray([], dtype=np.float64), 250)
    with pytest.raises(ValueError):
        block_max_threshold((), 0.05)
    with pytest.raises(ValueError):
        calibration_block_indices(0, 250)
    with pytest.raises(ValueError):
        expert_union_alerts(np.empty((0, 1)), ())
    with pytest.raises(ValueError):
        expert_union_alerts(np.empty((1, 0)), (0.0,))
    with pytest.raises(ValueError):
        expert_union_alerts(np.empty((1, 1)), ())


def test_calibration_helpers_reject_non_vector_and_non_finite_values() -> None:
    with pytest.raises(ValueError):
        consecutive_blocks(np.ones((1, 1)), 1)
    with pytest.raises(ValueError):
        consecutive_blocks(np.asarray([np.inf]), 1)
    with pytest.raises(ValueError):
        higher_quantile(np.ones((1, 1)), 0.5)


def test_sorted_block_threshold_matches_quantile_rule() -> None:
    generator = np.random.default_rng(7)
    for size in (1, 2, 7, 250, 301):
        blocks = (generator.normal(size=size), generator.normal(size=size + 3))
        for fraction in (0.0005, 0.01, 0.0125, 0.05, 0.3333, 0.5, 0.999):
            budget = fraction
            assert sorted_block_max_threshold(sorted_blocks(blocks), budget) == block_max_threshold(
                blocks, budget
            )


def test_sorted_blocks_reject_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        sorted_blocks(())
    with pytest.raises(ValueError):
        sorted_blocks((np.asarray([np.nan]),))


def test_joint_calibration_bounds_union_alerts_on_each_calibration_block() -> None:
    generator = np.random.default_rng(81)
    calibration = generator.normal(size=(4, 500))
    fitted = fit_peer_union(
        calibration,
        250,
        0.025,
        64.0,
        20,
    )
    assert fitted.joint_multiplier >= 1.0
    assert max(rate for rate in fitted.block_alert_rates) <= 0.025


def test_joint_calibration_produces_independent_expert_thresholds() -> None:
    calibration = np.vstack(
        (
            np.linspace(0.0, 1.0, 500),
            np.linspace(1.0, 5.0, 500),
            np.linspace(-1.0, 0.5, 500),
            np.linspace(2.0, 3.0, 500),
        )
    )
    fitted = fit_peer_union(
        calibration,
        250,
        0.025,
        64.0,
        20,
    )
    assert len(set(fitted.thresholds)) == 4


def test_joint_calibration_is_invariant_to_expert_order() -> None:
    calibration = np.asarray(
        [
            np.linspace(-2.0, 1.0, 600),
            np.linspace(-1.0, 4.0, 600),
            np.sin(np.linspace(0.0, 20.0, 600)),
            np.cos(np.linspace(0.0, 15.0, 600)),
        ]
    )
    permutation = (2, 0, 3, 1)
    settings = (
        250,
        0.025,
        64.0,
        20,
    )

    fitted = fit_peer_union(calibration, *settings)
    permuted = fit_peer_union(calibration[list(permutation)], *settings)

    assert permuted.joint_multiplier == fitted.joint_multiplier
    assert permuted.thresholds == tuple(fitted.thresholds[index] for index in permutation)
    assert permuted.block_alert_rates == fitted.block_alert_rates


def test_joint_calibration_stops_at_configured_cap_when_bound_still_holds() -> None:
    calibration = np.asarray(
        [
            [0.0, 1.0, 2.0, 3.0],
            [0.0, 1.0, 2.0, 4.0],
            [0.0, 1.0, 2.0, 5.0],
            [0.0, 1.0, 2.0, 6.0],
        ]
    )
    fitted = fit_peer_union(
        calibration,
        1,
        0.25,
        2.0,
        4,
    )
    assert fitted.joint_multiplier == 2.0


def test_evaluation_uses_strict_greater_than_threshold() -> None:
    calibration = np.tile(np.asarray([0.0, 1.0, 2.0, 3.0]), (4, 1))
    evaluated = np.full((4, 2), 3.0)
    _, alerts = evaluate_peer_union(
        calibration,
        evaluated,
        250,
        0.25,
        2.0,
        5,
    )
    np.testing.assert_array_equal(alerts, np.asarray([False, False]))


def test_protected_local_and_peer_channels_combine_with_or() -> None:
    local_calibration = np.arange(500, dtype=np.float64)
    local_evaluation = np.asarray([437.0, 438.0, 0.0])
    peer_calibration = np.tile(np.linspace(0.0, 1.0, 500), (4, 1))
    peer_evaluation = np.asarray(
        [[0.0, 0.0, 100.0], [0.0, 0.0, 100.0], [0.0, 0.0, 100.0], [0.0, 0.0, 100.0]]
    )

    decisions = apply_protected_local_peer_rule(
        local_calibration,
        local_evaluation,
        peer_calibration,
        peer_evaluation,
        250,
        0.25,
        0.25,
        2.0,
        20,
    )

    np.testing.assert_array_equal(decisions.local_alerts, [False, True, False])
    assert not decisions.peer_alerts[0]
    assert decisions.peer_alerts[2]
    np.testing.assert_array_equal(
        decisions.combined_alerts,
        np.logical_or(decisions.local_alerts, decisions.peer_alerts),
    )
    assert not decisions.local_alerts.flags.writeable
    assert not decisions.peer_alerts.flags.writeable
    assert not decisions.combined_alerts.flags.writeable


def test_protected_channel_rule_rejects_misaligned_evidence() -> None:
    with pytest.raises(ValueError, match="row-aligned"):
        apply_protected_local_peer_rule(
            np.asarray([0.0, 1.0]),
            np.asarray([0.0, 1.0]),
            np.zeros((4, 3)),
            np.zeros((4, 2)),
            250,
            0.25,
            0.25,
            2.0,
            20,
        )


def test_equal_share_union_keeps_the_independent_bonferroni_thresholds() -> None:
    local_calibration = np.asarray([0.0, 0.2, 0.4, 0.6])
    local_evaluation = np.asarray([0.6, 0.61])
    peer_calibration = np.tile(np.asarray([0.0, 1.0, 2.0, 3.0]), (4, 1))
    peer_evaluation = np.tile(np.asarray([3.0, 3.1]), (4, 1))

    decisions = apply_bonferroni_local_peer_rule(
        local_calibration,
        local_evaluation,
        peer_calibration,
        peer_evaluation,
        4,
        0.25,
        0.5,
    )

    assert decisions.joint_multiplier == 1.0
    assert decisions.local_threshold == 0.6
    assert decisions.expert_thresholds == (3.0,) * 4
    np.testing.assert_array_equal(decisions.local_alerts, [False, True])
    np.testing.assert_array_equal(decisions.peer_alerts, [False, True])
    np.testing.assert_array_equal(decisions.combined_alerts, [False, True])


def test_equal_share_calibration_rejects_missing_or_non_finite_evidence() -> None:
    with pytest.raises(ValueError, match="non-empty expert-by-row"):
        fit_bonferroni_peer_union(
            np.empty((0, 4)),
            4,
            0.25,
        )
    with pytest.raises(ValueError, match="finite"):
        fit_bonferroni_peer_union(
            np.full((4, 4), np.nan),
            4,
            0.25,
        )


def test_equal_share_evaluation_rejects_empty_and_misaligned_evidence() -> None:
    calibration = np.zeros((4, 4))
    with pytest.raises(ValueError, match="evaluation evidence"):
        evaluate_bonferroni_peer_union(
            calibration,
            np.empty((4, 0)),
            4,
            0.25,
        )
    with pytest.raises(ValueError, match="same experts"):
        evaluate_bonferroni_peer_union(
            calibration,
            np.empty((3, 2)),
            4,
            0.25,
        )


def test_collapsed_peer_max_fusion_calibrates_max_probability_evidence_once() -> None:
    local_calibration = np.asarray([0.0, 0.2, 0.4, 0.6])
    local_query = np.asarray([0.4, 0.61, 0.7])
    peer_calibration = np.asarray(
        [
            [0.0, 0.1, 0.2, 0.3],
            [0.0, 0.5, 0.6, 0.7],
        ]
    )
    peer_query = np.asarray(
        [
            [0.8, 0.2, 0.1],
            [0.2, 0.9, 0.6],
        ]
    )

    decisions = apply_max_fusion_rule(
        local_calibration,
        local_query,
        peer_calibration,
        peer_query,
        4,
        0.25,
        0.25,
    )

    assert decisions.peer_threshold == 0.7
    assert decisions.local_threshold == 0.6
    np.testing.assert_array_equal(decisions.peer_alerts, [True, True, False])
    np.testing.assert_array_equal(decisions.local_alerts, [False, True, True])
    np.testing.assert_array_equal(decisions.combined_alerts, [True, True, True])


@given(st.lists(st.floats(min_value=-100.0, max_value=100.0, allow_nan=False), min_size=1))
def test_increasing_query_scores_never_reduce_tail_evidence(scores: list[float]) -> None:
    ordered = np.sort(np.asarray(scores, dtype=np.float64))
    reference = np.linspace(-1.0, 1.0, 9)
    from pace.core.evidence import target_relative_evidence

    evidence = target_relative_evidence(reference, ordered)
    assert np.all(np.diff(evidence) >= -1e-12)


def test_invalid_joint_calibration_inputs_fail_clearly() -> None:
    with pytest.raises(ValueError):
        fit_peer_union(
            np.empty((0, 0)),
            250,
            0.025,
            64.0,
            20,
        )
    with pytest.raises(ValueError):
        fit_peer_union(
            np.full((4, 4), np.nan),
            250,
            0.025,
            64.0,
            20,
        )


def test_unequal_expert_shares_scale_thresholds_per_expert() -> None:
    generator = np.random.default_rng(8)
    evidence = generator.normal(size=(2, 600))
    equal = fit_peer_union(
        evidence,
        250,
        0.1,
        64.0,
        20,
    )
    skewed = fit_peer_union(
        evidence,
        250,
        0.1,
        64.0,
        20,
        (0.09, 0.01),
    )
    assert skewed.thresholds[0] < equal.thresholds[0]
    assert skewed.thresholds[1] > equal.thresholds[1]
    with pytest.raises(ValueError, match="one budget share"):
        fit_peer_union(
            evidence,
            250,
            0.1,
            64.0,
            20,
            (0.1,),
        )


def test_single_expert_budget_is_clamped_instead_of_rejected() -> None:
    generator = np.random.default_rng(21)
    evidence = generator.normal(size=(1, 600))
    calibration = fit_peer_union(
        evidence,
        250,
        0.025,
        64.0,
        20,
    )
    assert calibration.joint_multiplier >= 1.0
    with pytest.raises(ValueError, match="strictly below one"):
        fit_peer_union(
            evidence,
            250,
            0.5,
            64.0,
            20,
            (0.9999,),
        )


def test_peer_union_rejects_non_finite_evaluation_evidence() -> None:
    evidence = np.random.default_rng(2).normal(size=(2, 300))
    broken = evidence.copy()
    broken[0, 0] = np.nan
    with pytest.raises(ValueError, match="evaluation evidence must be finite"):
        evaluate_peer_union(
            evidence,
            broken,
            250,
            0.05,
            64.0,
            20,
        )


def _tiered_inputs() -> tuple[TieredLocalEvidence, np.ndarray, np.ndarray]:
    generator = np.random.default_rng(311)
    anchor_cal, anchor_query = generator.exponential(size=500), generator.exponential(size=700)
    bank_cal, bank_query = (
        generator.exponential(size=(3, 500)),
        generator.exponential(size=(3, 700)),
    )
    peers_cal, peers_query = (
        generator.exponential(size=(4, 500)),
        generator.exponential(size=(4, 700)),
    )
    return (
        TieredLocalEvidence(anchor_cal, anchor_query, bank_cal, bank_query),
        peers_cal,
        peers_query,
    )


def test_tiered_rule_is_the_union_of_anchor_bank_and_peer_unions() -> None:
    local, peers_cal, peers_query = _tiered_inputs()
    decisions = apply_tiered_local_peer_rule(local, peers_cal, peers_query, 250, 0.03, 0.03, RULE)
    _, anchor = evaluate_peer_union(
        local.anchor_calibration[np.newaxis, :],
        local.anchor_evaluation[np.newaxis, :],
        250,
        0.015,
        64.0,
        20,
    )
    _, bank = evaluate_peer_union(
        local.bank_calibration, local.bank_evaluation, 250, 0.015, 64.0, 20
    )
    _, peers = evaluate_peer_union(peers_cal, peers_query, 250, 0.03, 64.0, 20)
    np.testing.assert_array_equal(decisions.local_alerts, anchor | bank)
    np.testing.assert_array_equal(decisions.peer_alerts, peers)
    np.testing.assert_array_equal(decisions.combined_alerts, anchor | bank | peers)
    assert len(decisions.bank_thresholds) == 3


def test_tiered_local_budget_holds_on_every_calibration_block() -> None:
    local, peers_cal, peers_query = _tiered_inputs()
    anchor, _ = evaluate_peer_union(
        local.anchor_calibration[np.newaxis, :],
        local.anchor_evaluation[np.newaxis, :],
        250,
        0.015,
        64.0,
        20,
    )
    bank, _ = evaluate_peer_union(
        local.bank_calibration, local.bank_evaluation, 250, 0.015, 64.0, 20
    )
    assert anchor.worst_block_rate <= 0.015
    assert bank.worst_block_rate <= 0.015
    decisions = apply_tiered_local_peer_rule(local, peers_cal, peers_query, 250, 0.03, 0.03, RULE)
    assert decisions.local_threshold == anchor.thresholds[0]


def test_share_conserving_policy_never_exceeds_the_local_budget() -> None:
    local, peers_cal, peers_query = _tiered_inputs()
    for local_budget in (0.0025, 0.005, 0.01, 0.025, 0.05):
        decisions = apply_tiered_local_peer_rule(
            local, peers_cal, peers_query, 250, local_budget, local_budget, RULE
        )
        assert decisions.anchor_budget is not None
        assert decisions.bank_budget is not None
        assert np.isclose(decisions.anchor_budget + decisions.bank_budget, local_budget)


def test_allocation_audit_reports_effective_budget_and_resolution_floor() -> None:
    audit = AllocationAudit(
        nominal_budget=0.01,
        peer_budget=0.005,
        local_budget=0.005,
        anchor_budget=0.005,
        bank_budget=0.0025,
        calibration_rows=350,
        peer_expert_count=7,
        bank_expert_count=8,
    )
    assert np.isclose(audit.effective_nominal_budget, 0.0125)
    assert np.isclose(audit.resolution_floor, 16 / 351)


def test_cross_fitted_union_is_deterministic_and_valid() -> None:
    generator = np.random.default_rng(5)
    evidence = generator.exponential(size=(6, 800))
    first = fit_cross_fitted_peer_union(evidence, 5, 0.05, 64.0, 20)
    second = fit_cross_fitted_peer_union(evidence, 5, 0.05, 64.0, 20)
    assert first == second
    assert len(first.thresholds) == 6
    assert len(first.block_alert_rates) == 5
    assert first.worst_block_rate >= 0.0


def test_cross_fitted_union_tracks_a_resolvable_budget_on_fresh_data() -> None:
    generator = np.random.default_rng(23)
    rates: list[float] = []
    for _ in range(12):
        calibration = generator.exponential(size=(5, 1500))
        fresh = generator.exponential(size=(5, 20000))
        fitted = fit_cross_fitted_peer_union(calibration, 5, 0.05, 64.0, 24)
        rates.append(np.mean(expert_union_alerts(fresh, fitted.thresholds)))
    assert np.isclose(np.mean(rates), 0.05, atol=0.015)


def test_cross_fitted_union_is_monotone_in_the_budget() -> None:
    generator = np.random.default_rng(9)
    evidence = generator.exponential(size=(4, 1000))
    thresholds = [
        fit_cross_fitted_peer_union(evidence, 5, budget, 64.0, 24).thresholds
        for budget in (0.02, 0.05, 0.1)
    ]
    for tighter, looser in zip(thresholds, thresholds[1:], strict=False):
        assert all(a >= b for a, b in zip(tighter, looser, strict=True))


def test_cross_fitted_union_saturates_at_the_resolution_floor() -> None:
    generator = np.random.default_rng(31)
    evidence = generator.exponential(size=(7, 350))
    fitted = fit_cross_fitted_peer_union(evidence, 5, 0.001, 64.0, 20)
    assert np.isclose(fitted.joint_multiplier, 1.0 / 64.0)
    assert fitted.worst_block_rate > 0.001


def test_cross_fitted_union_rejects_too_few_rows_for_the_folds() -> None:
    with pytest.raises(ValueError, match="at least one row per fold"):
        fit_cross_fitted_peer_union(np.ones((2, 3)), 5, 0.05, 64.0, 20)


def test_cross_fitted_mode_is_selectable_through_the_union_evaluation() -> None:
    generator = np.random.default_rng(13)
    calibration = generator.exponential(size=(4, 800))
    evaluation = generator.exponential(size=(4, 900))
    fitted, alerts = evaluate_peer_union(
        calibration, evaluation, 250, 0.05, 64.0, 20, None, CalibrationMode.CROSS_FITTED, 5
    )
    direct = fit_cross_fitted_peer_union(calibration, 5, 0.05, 64.0, 20)
    assert fitted == direct
    np.testing.assert_array_equal(alerts, expert_union_alerts(evaluation, direct.thresholds))


def test_tiered_rule_runs_with_cross_fitted_calibration() -> None:
    local, peers_cal, peers_query = _tiered_inputs()
    rule = TieredCalibrationRule(
        anchor_share=0.5,
        anchor_policy=AnchorBudgetPolicy.SHARE_CONSERVING,
        union=CROSS_FITTED_UNION,
    )
    decisions = apply_tiered_local_peer_rule(local, peers_cal, peers_query, 250, 0.03, 0.03, rule)
    np.testing.assert_array_equal(
        decisions.combined_alerts, decisions.local_alerts | decisions.peer_alerts
    )
    assert len(decisions.bank_thresholds) == 3


def test_tiered_rule_without_a_bank_gives_the_anchor_the_whole_local_budget() -> None:
    local, peers_cal, peers_query = _tiered_inputs()
    anchor_only = TieredLocalEvidence(
        local.anchor_calibration,
        local.anchor_evaluation,
        np.empty((0, 500)),
        np.empty((0, 700)),
    )
    decisions = apply_tiered_local_peer_rule(
        anchor_only, peers_cal, peers_query, 250, 0.03, 0.03, RULE
    )
    _, anchor = evaluate_peer_union(
        anchor_only.anchor_calibration[np.newaxis, :],
        anchor_only.anchor_evaluation[np.newaxis, :],
        250,
        0.03,
        64.0,
        20,
    )
    np.testing.assert_array_equal(decisions.local_alerts, anchor)
    assert decisions.bank_thresholds == ()


def test_peer_alerts_do_not_depend_on_the_local_branch_design() -> None:
    local, peers_cal, peers_query = _tiered_inputs()
    tiered = apply_tiered_local_peer_rule(local, peers_cal, peers_query, 250, 0.03, 0.03, RULE)
    single_anchor = apply_protected_local_peer_rule(
        local.anchor_calibration,
        local.anchor_evaluation,
        peers_cal,
        peers_query,
        250,
        0.03,
        0.03,
        64.0,
        20,
    )
    np.testing.assert_array_equal(tiered.peer_alerts, single_anchor.peer_alerts)
    assert tiered.expert_thresholds == single_anchor.expert_thresholds


def test_tiered_evidence_rejects_misaligned_rows() -> None:
    with pytest.raises(ValueError, match="align anchor and bank"):
        TieredLocalEvidence(np.ones(10), np.ones(12), np.ones((2, 9)), np.ones((2, 12)))


def test_single_union_with_anchor_weight_reproduces_manual_shares() -> None:
    local, peers_cal, peers_query = _tiered_inputs()
    calibration = np.vstack((local.bank_calibration, local.anchor_calibration[np.newaxis, :]))
    evaluation = np.vstack((local.bank_evaluation, local.anchor_evaluation[np.newaxis, :]))
    shares = (0.25 / 3 * 2, 0.25 / 3 * 2, 0.25 / 3 * 2, 0.5)
    decisions = apply_local_union_peer_rule(
        calibration, evaluation, peers_cal, peers_query, 250, 0.03, 0.03, UNION, shares
    )
    _, expected = evaluate_peer_union(
        calibration, evaluation, 250, 0.03, 64.0, 20, tuple(share * 0.03 for share in shares)
    )
    np.testing.assert_array_equal(decisions.local_alerts, expected)


def _naive_cross_fitted(
    evidence: FloatMatrix, folds: int, budget: float, cap: float, iterations: int
) -> tuple[float, tuple[float, ...]]:
    experts, rows = evidence.shape
    shares = [budget / experts] * experts
    parts: list[RowIndexArray] = list(np.array_split(np.arange(rows, dtype=np.int64), folds))

    def rate(multiplier: float) -> float:
        rates: list[float] = []
        for part in parts:
            train = np.setdiff1d(np.arange(rows, dtype=np.int64), part)
            thresholds = [
                np.quantile(row[train], 1.0 - min(share * multiplier, 0.999), method="higher")
                for row, share in zip(evidence, shares, strict=True)
            ]
            hits = np.any(evidence[:, part] > np.asarray(thresholds)[:, np.newaxis], axis=0)
            rates.append(np.mean(hits).item())
        return float(np.mean(rates))

    lower, upper = 1.0 / cap, cap
    if rate(lower) > budget:
        chosen = lower
    elif rate(upper) <= budget:
        chosen = upper
    else:
        chosen = lower
        for _ in range(iterations):
            midpoint = (chosen * upper) ** 0.5
            if rate(midpoint) <= budget:
                chosen = midpoint
            else:
                upper = midpoint
    return chosen, tuple(
        np.quantile(row, 1.0 - min(share * chosen, 0.999), method="higher").item()
        for row, share in zip(evidence, shares, strict=True)
    )


@pytest.mark.parametrize("budget", [0.002, 0.03, 0.2])
def test_fast_cross_fitting_matches_the_direct_quantile_definition(budget: float) -> None:
    evidence = np.random.default_rng(77).exponential(size=(5, 700))
    chosen, thresholds = _naive_cross_fitted(evidence, 5, budget, 64.0, 20)
    fitted = fit_cross_fitted_peer_union(evidence, 5, budget, 64.0, 20)
    assert np.isclose(fitted.joint_multiplier, chosen)
    np.testing.assert_allclose(fitted.thresholds, thresholds)


COLLAPSE_RULE = TieredCalibrationRule(
    anchor_share=0.5, anchor_policy=AnchorBudgetPolicy.COLLAPSE_TO_ANCHOR, union=UNION
)


def test_collapse_policy_gives_an_unresolvable_anchor_the_whole_local_budget() -> None:
    local, peers_cal, peers_query = _tiered_inputs()
    decisions = apply_tiered_local_peer_rule(
        local, peers_cal, peers_query, 250, 0.005, 0.005, COLLAPSE_RULE
    )
    assert decisions.anchor_budget == 0.005
    assert decisions.bank_budget is None
    assert decisions.bank_thresholds == ()
    anchor, anchor_alerts = evaluate_peer_union(
        local.anchor_calibration[np.newaxis, :],
        local.anchor_evaluation[np.newaxis, :],
        250,
        0.005,
        64.0,
        20,
    )
    assert decisions.local_threshold == anchor.thresholds[0]
    np.testing.assert_array_equal(decisions.local_alerts, anchor_alerts)


def test_collapse_policy_never_exceeds_the_local_budget() -> None:
    local, peers_cal, peers_query = _tiered_inputs()
    for local_budget in (0.0025, 0.005, 0.0125, 0.025, 0.05):
        decisions = apply_tiered_local_peer_rule(
            local, peers_cal, peers_query, 250, local_budget, local_budget, COLLAPSE_RULE
        )
        assert decisions.anchor_budget is not None
        total = decisions.anchor_budget + (decisions.bank_budget or 0.0)
        assert np.isclose(total, local_budget)


def test_collapse_policy_keeps_the_bank_wherever_the_anchor_tier_resolves() -> None:
    local, peers_cal, peers_query = _tiered_inputs()
    collapsed = apply_tiered_local_peer_rule(
        local, peers_cal, peers_query, 250, 0.025, 0.025, COLLAPSE_RULE
    )
    conserving = apply_tiered_local_peer_rule(
        local, peers_cal, peers_query, 250, 0.025, 0.025, RULE
    )
    assert collapsed.bank_budget == conserving.bank_budget
    np.testing.assert_array_equal(collapsed.combined_alerts, conserving.combined_alerts)


def test_collapse_without_a_bank_is_the_plain_anchor_rule() -> None:
    local, peers_cal, peers_query = _tiered_inputs()
    anchor_only = TieredLocalEvidence(
        local.anchor_calibration,
        local.anchor_evaluation,
        np.empty((0, 500)),
        np.empty((0, 700)),
    )
    decisions = apply_tiered_local_peer_rule(
        anchor_only, peers_cal, peers_query, 250, 0.005, 0.005, COLLAPSE_RULE
    )
    assert decisions.anchor_budget == 0.005
    assert decisions.bank_budget is None


def test_calibration_modes_all_run_through_the_tiered_rule() -> None:
    local, peers_cal, peers_query = _tiered_inputs()
    for mode in CalibrationMode:
        rule = TieredCalibrationRule(
            anchor_share=0.5,
            anchor_policy=AnchorBudgetPolicy.SHARE_CONSERVING,
            union=UnionCalibration(scale_cap=64.0, scale_iterations=20, mode=mode),
        )
        decisions = apply_tiered_local_peer_rule(
            local, peers_cal, peers_query, 250, 0.03, 0.03, rule
        )
        np.testing.assert_array_equal(
            decisions.combined_alerts, decisions.local_alerts | decisions.peer_alerts
        )
