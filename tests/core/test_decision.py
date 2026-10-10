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
    fit_peer_union,
    higher_quantile,
    sorted_block_max_threshold,
    sorted_blocks,
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
    decisions = apply_tiered_local_peer_rule(
        local, peers_cal, peers_query, 250, 0.03, 0.03, 0.5, 64.0, 20
    )
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
    decisions = apply_tiered_local_peer_rule(
        local, peers_cal, peers_query, 250, 0.03, 0.03, 0.5, 64.0, 20
    )
    assert decisions.local_threshold == anchor.thresholds[0]


def test_unresolvable_anchor_tier_keeps_the_whole_local_budget() -> None:
    local, peers_cal, peers_query = _tiered_inputs()
    decisions = apply_tiered_local_peer_rule(
        local, peers_cal, peers_query, 250, 0.005, 0.005, 0.5, 64.0, 20
    )
    anchor, anchor_alerts = evaluate_peer_union(
        local.anchor_calibration[np.newaxis, :],
        local.anchor_evaluation[np.newaxis, :],
        250,
        0.005,
        64.0,
        20,
    )
    legacy = apply_protected_local_peer_rule(
        local.anchor_calibration,
        local.anchor_evaluation,
        peers_cal,
        peers_query,
        250,
        0.005,
        0.005,
        64.0,
        20,
    )
    assert decisions.local_threshold == anchor.thresholds[0] == legacy.local_threshold
    assert np.all(decisions.local_alerts[anchor_alerts])
    assert np.all(decisions.combined_alerts[legacy.local_alerts])


def test_tiered_rule_without_a_bank_gives_the_anchor_the_whole_local_budget() -> None:
    local, peers_cal, peers_query = _tiered_inputs()
    anchor_only = TieredLocalEvidence(
        local.anchor_calibration,
        local.anchor_evaluation,
        np.empty((0, 500)),
        np.empty((0, 700)),
    )
    decisions = apply_tiered_local_peer_rule(
        anchor_only, peers_cal, peers_query, 250, 0.03, 0.03, 0.5, 64.0, 20
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
    tiered = apply_tiered_local_peer_rule(
        local, peers_cal, peers_query, 250, 0.03, 0.03, 0.5, 64.0, 20
    )
    legacy = apply_protected_local_peer_rule(
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
    np.testing.assert_array_equal(tiered.peer_alerts, legacy.peer_alerts)
    assert tiered.expert_thresholds == legacy.expert_thresholds


def test_tiered_evidence_rejects_misaligned_rows() -> None:
    with pytest.raises(ValueError, match="align anchor and bank"):
        TieredLocalEvidence(np.ones(10), np.ones(12), np.ones((2, 9)), np.ones((2, 12)))


def test_single_union_with_anchor_weight_reproduces_manual_shares() -> None:
    local, peers_cal, peers_query = _tiered_inputs()
    calibration = np.vstack((local.bank_calibration, local.anchor_calibration[np.newaxis, :]))
    evaluation = np.vstack((local.bank_evaluation, local.anchor_evaluation[np.newaxis, :]))
    shares = (0.25 / 3 * 2, 0.25 / 3 * 2, 0.25 / 3 * 2, 0.5)
    decisions = apply_local_union_peer_rule(
        calibration, evaluation, peers_cal, peers_query, 250, 0.03, 0.03, 64.0, 20, shares
    )
    _, expected = evaluate_peer_union(
        calibration, evaluation, 250, 0.03, 64.0, 20, tuple(share * 0.03 for share in shares)
    )
    np.testing.assert_array_equal(decisions.local_alerts, expected)
