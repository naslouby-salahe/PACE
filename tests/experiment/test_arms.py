from dataclasses import replace

import numpy as np
import pytest

from pace.config import ResolvedConfiguration
from pace.core.decision import apply_protected_local_peer_rule
from pace.experiment import runner
from pace.experiment.arms import protected_channel_decisions
from pace.experiment.inputs import (
    PeerEvidence,
    PreparedDevice,
    TargetFrame,
    build_target_frame,
    compute_local_evidence,
    compute_peer_evidence,
)
from pace.reporting.metrics import summarize_campaign
from pace.types import AlphaLevel, LocalExpertKind, StudyArm
from tests.support import campaign_configuration, campaign_devices, run_single_seed_campaign

SEED = 3


def _arm_configuration(arms: tuple[StudyArm, ...]) -> ResolvedConfiguration:
    configuration = campaign_configuration()
    return configuration.model_copy(
        update={"scientific": configuration.scientific.model_copy(update={"study_arms": arms})}
    )


def test_every_study_arm_runs_and_reports_aligned_detection_rates() -> None:
    configuration = _arm_configuration(tuple(StudyArm))
    results = run_single_seed_campaign(campaign_devices(), configuration, 3)
    for target in results:
        assert {performance.arm for performance in target.arms} == set(StudyArm)
        for performance in target.arms:
            assert len(performance.detection_rates) == len(target.attack_types)
            assert 0.0 <= performance.false_alert_rate <= 1.0


def test_arm_results_are_deterministic_and_do_not_change_main_method() -> None:
    plain = run_single_seed_campaign(campaign_devices(), campaign_configuration(), 5)
    with_arms = run_single_seed_campaign(campaign_devices(), _arm_configuration(tuple(StudyArm)), 5)
    repeated = run_single_seed_campaign(campaign_devices(), _arm_configuration(tuple(StudyArm)), 5)
    assert with_arms == repeated
    for left, right in zip(plain, with_arms, strict=True):
        assert left.attack_types == right.attack_types
        assert left.realized_false_alert_rate == right.realized_false_alert_rate
        assert left.arms == ()


def test_campaign_summary_reports_each_requested_arm() -> None:
    arms = (
        StudyArm.NO_LOCAL_CHANNEL,
        StudyArm.RESERVE_QUARTER,
        StudyArm.MEAN_ENSEMBLE_LOCAL_BRANCH,
    )
    configuration = _arm_configuration(arms)
    results = tuple(
        result
        for seed in (1, 2)
        for result in run_single_seed_campaign(campaign_devices(), configuration, seed)
    )
    summary = summarize_campaign(results, AlphaLevel.ALPHA_005, configuration.reporting)
    assert {item.arm for item in summary.arms} == set(arms)
    for item in summary.arms:
        assert item.target_count == len(item.targets) == 5
        for target in item.targets:
            assert [seed.seed for seed in target.seed_differences] == [1, 2]
        assert 0.0 <= item.macro_detection_rate <= 1.0
        assert item.method_arm_confidence_interval.lower_bound <= (
            item.method_arm_confidence_interval.upper_bound
        )


def test_duplicate_or_unknown_study_arms_are_rejected() -> None:
    from pydantic import ValidationError

    configuration = campaign_configuration()
    payload = configuration.scientific.model_dump(mode="json")
    model_type = type(configuration.scientific)
    for invalid in (
        ["NO_LOCAL_CHANNEL", "NO_LOCAL_CHANNEL"],
        ["NOT_AN_ARM"],
        "NO_LOCAL_CHANNEL",
    ):
        payload["study_arms"] = invalid
        with pytest.raises(ValidationError):
            model_type.model_validate(payload)


def _configuration(arms: tuple[StudyArm, ...] = ()) -> ResolvedConfiguration:
    configuration = campaign_configuration()
    scientific = configuration.scientific.model_copy(update={"study_arms": arms})
    return configuration.model_copy(update={"scientific": scientific})


def _local_inputs(
    configuration: ResolvedConfiguration,
) -> tuple[PreparedDevice, TargetFrame, PeerEvidence]:
    prepared = runner.prepare_clients(campaign_devices(), configuration)
    experts = runner.fit_peer_experts(prepared, configuration, SEED)
    target = prepared[0]
    frame = build_target_frame(target)
    peers = compute_peer_evidence(tuple(e for e in experts if e.client != target.client), frame)
    return target, frame, peers


def test_legacy_arm_reproduces_the_frozen_protected_local_rule() -> None:
    configuration = _configuration((StudyArm.MEAN_ENSEMBLE_LOCAL_BRANCH,))
    target, frame, peers = _local_inputs(configuration)
    local = compute_local_evidence(target, frame, configuration, SEED, peers)
    scientific = configuration.scientific
    expected = apply_protected_local_peer_rule(
        local.calibration,
        local.query,
        peers.calibration_evidence,
        peers.query_evidence,
        scientific.calibration_block_size,
        scientific.alpha.fraction * scientific.reserve_fraction,
        scientific.alpha.fraction * (1.0 - scientific.reserve_fraction),
        scientific.joint_scale_cap,
        scientific.joint_scale_iterations,
    ).combined_alerts
    (result,) = run_single_seed_campaign(campaign_devices(), configuration, SEED)[:1]
    (arm,) = result.arms
    assert arm.arm is StudyArm.MEAN_ENSEMBLE_LOCAL_BRANCH
    assert arm.false_alert_rate == frame.false_alert_rate(expected)
    assert arm.detection_rates == frame.detection_rates(expected)


def test_bank_ablation_arms_run_in_the_requested_order() -> None:
    ablations = (
        StudyArm.BANK_WITHOUT_FEATURE_EXPERTS,
        StudyArm.BANK_WITHOUT_RESPONSE_EXPERTS,
        StudyArm.BANK_WITHOUT_RARITY_EXPERTS,
        StudyArm.BANK_WITHOUT_ANCHOR,
        StudyArm.BANK_SINGLE_UNION,
        StudyArm.RESPONSE_ON_MARGINS,
    )
    for target in run_single_seed_campaign(campaign_devices(), _configuration(ablations), SEED):
        assert tuple(item.arm for item in target.arms) == ablations
        assert all(0.0 <= item.false_alert_rate <= 1.0 for item in target.arms)


def test_bank_evidence_uses_only_the_first_local_training_rows() -> None:
    configuration = _configuration()
    target, frame, peers = _local_inputs(configuration)
    rows = configuration.scientific.local_detectors.local_training_rows
    assert target.training.shape[0] > rows
    baseline = compute_local_evidence(target, frame, configuration, SEED, peers)
    scrambled = target.training.copy()
    scrambled[rows:] = 99.0
    altered = compute_local_evidence(
        replace(target, training=scrambled), frame, configuration, SEED, peers
    )
    np.testing.assert_array_equal(baseline.bank.query, altered.bank.query)
    np.testing.assert_array_equal(baseline.bank.anchor_query, altered.bank.anchor_query)
    np.testing.assert_array_equal(baseline.query, altered.query)


def test_bank_evidence_is_chronological_calibration_rows_cannot_change_query_scores() -> None:
    configuration = _configuration()
    target, frame, peers = _local_inputs(configuration)
    baseline = compute_local_evidence(target, frame, configuration, SEED, peers)
    shifted = replace(frame, calibration=frame.calibration + 10.0)
    altered = compute_local_evidence(
        target, shifted, configuration, SEED, compute_peer_evidence(peers.experts, shifted)
    )
    np.testing.assert_array_equal(baseline.bank.query, altered.bank.query)
    assert not np.array_equal(baseline.bank.calibration, altered.bank.calibration)


def test_bank_reports_every_expert_kind() -> None:
    configuration = _configuration()
    target, frame, peers = _local_inputs(configuration)
    local = compute_local_evidence(target, frame, configuration, SEED, peers)
    assert local.bank.expert_kinds == tuple(LocalExpertKind)


def test_protected_decisions_use_the_anchor_and_every_bank_expert() -> None:
    configuration = _configuration()
    target, frame, peers = _local_inputs(configuration)
    local = compute_local_evidence(target, frame, configuration, SEED, peers)
    decisions = protected_channel_decisions(
        local,
        peers.calibration_evidence,
        peers.query_evidence,
        configuration.scientific,
        0.025,
        0.025,
    )
    assert len(decisions.bank_thresholds) == len(LocalExpertKind)
