import numpy as np

from pace.config import CalibrationStudySettings, ReportingSettings
from pace.reporting.inference import (
    bootstrap_p_value,
    cluster_bootstrap_estimates,
    dataset_groups,
    estimates_interval,
    holm_adjusted,
    within_dataset_estimates,
)
from pace.types import (
    ClientName,
    DatasetClientId,
    DatasetName,
    InferenceMethod,
    TargetPairedDifference,
)

STUDY = CalibrationStudySettings(
    rows=(100, 300),
    budgets=(0.01, 0.05),
    expert_counts=(1, 4),
    replicates=2,
    test_rows=2000,
    attack_rows=500,
    seed=1,
)


def _settings(replicates: int = 4000) -> ReportingSettings:
    return ReportingSettings(
        bootstrap_replicates=replicates,
        bootstrap_seed=41,
        confidence_level=0.95,
        high_false_alert_targets=10,
        benchmark_targets=2,
        calibration_study=STUDY,
    )


def _difference(dataset: DatasetName, client: str, value: float) -> TargetPairedDifference:
    return TargetPairedDifference(DatasetClientId(dataset, ClientName(client)), value)


def test_dataset_groups_are_sorted_and_independent_of_input_order() -> None:
    differences = (
        _difference(DatasetName.TON_IOT, "b", 0.2),
        _difference(DatasetName.IOT_23, "z", 0.1),
        _difference(DatasetName.TON_IOT, "a", 0.4),
    )
    forward = dataset_groups(differences)
    backward = dataset_groups(tuple(reversed(differences)))
    assert list(forward) == list(backward)
    for dataset in forward:
        np.testing.assert_array_equal(forward[dataset], backward[dataset])
    np.testing.assert_array_equal(forward[DatasetName.TON_IOT], np.array([0.4, 0.2]))


def test_cluster_bootstrap_is_deterministic_for_a_seed() -> None:
    groups = (np.array([0.1, 0.2, 0.3]), np.array([-0.1, 0.0]))
    first = cluster_bootstrap_estimates(groups, _settings(), InferenceMethod.DATASET_CLUSTERS)
    repeated = cluster_bootstrap_estimates(groups, _settings(), InferenceMethod.DATASET_CLUSTERS)
    np.testing.assert_array_equal(first, repeated)
    assert first.shape == (4000,)


def test_cluster_bootstrap_captures_between_dataset_variance_that_stratification_ignores() -> None:
    groups = (np.full(10, 0.3), np.full(10, -0.3))
    cluster = estimates_interval(
        cluster_bootstrap_estimates(groups, _settings(), InferenceMethod.DATASET_CLUSTERS),
        _settings(),
    )
    assert cluster.lower_bound < -0.25
    assert cluster.upper_bound > 0.25


def test_cluster_bootstrap_of_identical_datasets_collapses_to_the_common_value() -> None:
    groups = (np.full(6, 0.07), np.full(4, 0.07), np.full(9, 0.07))
    for method in (InferenceMethod.DATASET_CLUSTERS, InferenceMethod.DATASET_MEANS):
        estimates = cluster_bootstrap_estimates(groups, _settings(1000), method)
        np.testing.assert_allclose(estimates, 0.07)


def test_equal_weight_estimator_weights_datasets_equally_and_the_target_mean_by_size() -> None:
    groups = (np.full(90, 0.0), np.full(10, 1.0))
    weighted = cluster_bootstrap_estimates(groups, _settings(), InferenceMethod.DATASET_CLUSTERS)
    equal = cluster_bootstrap_estimates(groups, _settings(), InferenceMethod.DATASET_MEANS)
    assert abs(float(np.mean(equal)) - 0.5) < 0.02
    assert float(np.min(weighted)) == 0.0
    assert float(np.max(weighted)) == 1.0
    assert float(np.mean(weighted)) < float(np.mean(equal))


def test_within_dataset_bootstrap_is_centred_on_the_sample_mean() -> None:
    values = np.array([0.01, 0.02, 0.05, 0.08, 0.04])
    estimates = within_dataset_estimates(values, _settings(8000))
    assert abs(float(np.mean(estimates)) - float(np.mean(values))) < 0.003
    interval = estimates_interval(estimates, _settings(8000))
    assert interval.lower_bound < float(np.mean(values)) < interval.upper_bound


def test_p_value_is_small_for_a_clear_effect_and_one_for_a_null_effect() -> None:
    clear = within_dataset_estimates(np.array([0.1, 0.12, 0.09, 0.11, 0.13]), _settings())
    null = within_dataset_estimates(np.zeros(5), _settings())
    assert bootstrap_p_value(clear) < 0.01
    assert bootstrap_p_value(null) == 1.0


def test_holm_adjustment_matches_a_hand_computed_example() -> None:
    adjusted = holm_adjusted((0.01, 0.04, 0.03, 0.5))
    np.testing.assert_allclose(adjusted, (0.04, 0.09, 0.09, 0.5))
    assert holm_adjusted((0.02,)) == (0.02,)
    assert all(a >= p for a, p in zip(adjusted, (0.01, 0.04, 0.03, 0.5), strict=True))
    np.testing.assert_allclose(holm_adjusted((0.6, 0.7)), (1.0, 1.0))


def test_a_dataset_drawn_twice_receives_independent_within_dataset_resamples() -> None:
    groups = (np.array([0.0, 1.0]), np.array([0.0, 1.0]))
    estimates = cluster_bootstrap_estimates(groups, _settings(8000), InferenceMethod.DATASET_MEANS)
    values, counts = np.unique(np.round(estimates, 6), return_counts=True)
    shares = dict(zip(values.tolist(), (counts / counts.sum()).tolist(), strict=True))
    assert set(shares) <= {0.0, 0.25, 0.5, 0.75, 1.0}
    assert abs(shares[0.25] - 0.25) < 0.03
    assert abs(shares[0.5] - 0.375) < 0.03
