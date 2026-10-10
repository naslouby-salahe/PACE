import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import BaseModel, ValidationError

from pace.config import (
    BudgetSweepSettings,
    PeerTrainingSettings,
    ProtocolMatrixSettings,
    ReportingSettings,
    ResolvedConfiguration,
    RuntimeSettings,
    ScientificSettings,
    SplitSettings,
    configuration_for_protocol_variant,
    load_configuration,
)
from pace.types import (
    AlphaLevel,
    BudgetAnalysisKind,
    ConfigurationError,
    LogLevel,
    StudyArm,
    StudyStratum,
)
from tests.support import REPOSITORY_ROOT, protocol_matrix


def test_configuration_loads_frozen_semantic_types_and_resolves_paths(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    configuration = load_configuration(REPOSITORY_ROOT / "configs" / "default.yaml", project_root)
    assert configuration.scientific.alpha is AlphaLevel.ALPHA_005
    assert configuration.scientific.reserve_fraction == 0.5
    assert configuration.scientific.calibration_block_size == 250
    assert configuration.scientific.minimum_peer_count == 4
    assert configuration.scientific.seed == 17
    assert configuration.scientific.learning_seeds == (6, 7, 8)
    assert configuration.scientific.joint_scale_cap == 64.0
    assert configuration.scientific.joint_scale_iterations == 20
    assert configuration.scientific.peer_training.hidden_layer_widths == (
        64,
        16,
    )
    assert configuration.scientific.peer_training.epochs == 8
    assert configuration.scientific.peer_training.batch_size == 256
    assert configuration.scientific.peer_training.learning_rate == 0.001
    assert configuration.scientific.local_detectors.local_training_rows == 400
    assert configuration.splitting.training_fraction == 0.4
    assert configuration.splitting.minimum_calibration_rows == 350
    assert configuration.splitting.minimum_attack_rows == 600
    assert configuration.splitting.minimum_attack_rows_for(StudyStratum.IOT_23) == 400
    assert (
        configuration.splitting.minimum_attack_rows_for(StudyStratum.UNSW_IOT_ATTACK_FLOWS_G1) == 30
    )
    assert configuration.splitting.minimum_attack_rows_for(StudyStratum.GOTHAM_2025) == 600
    assert configuration.splitting.attack_rows_per_half_cap == 10000
    assert configuration.reporting.bootstrap_replicates == 10000
    assert configuration.reporting.bootstrap_seed == 314159
    assert configuration.reporting.confidence_level == 0.95
    assert configuration.protocol_matrix.alpha_levels == (
        AlphaLevel.ALPHA_001,
        AlphaLevel.ALPHA_005,
    )
    assert configuration.protocol_matrix.local_training_row_counts == (
        100,
        400,
        750,
        2000,
        20000,
    )
    assert tuple((spec.kind, spec.budgets) for spec in configuration.protocol_matrix.analyses) == (
        (BudgetAnalysisKind.UNIVERSAL_PRIMARY, (100, 400)),
        (BudgetAnalysisKind.UNIVERSAL_ROBUSTNESS, (100, 400, 750)),
        (BudgetAnalysisKind.BROAD_SCALING, (100, 400, 2000)),
        (BudgetAnalysisKind.HIGH_RESOURCE, (100, 400, 2000, 20000)),
    )
    assert configuration.protocol_matrix.power.minimum_meaningful_effect == 0.02
    assert configuration.runtime.provenance_active is True
    assert configuration.runtime.raw_data_root == (project_root / "data" / "raw").resolve()
    assert configuration.runtime.output_root == (project_root / "outputs").resolve()
    seed_attribute = "seed"
    with pytest.raises(ValidationError):
        setattr(configuration.scientific, seed_attribute, 12)


def test_protocol_matrix_resolves_distinct_scientific_campaigns(tmp_path: Path) -> None:
    configuration = load_configuration(REPOSITORY_ROOT / "configs" / "default.yaml", tmp_path)

    variants = tuple(
        configuration_for_protocol_variant(configuration, alpha, local_training_rows)
        for alpha in configuration.protocol_matrix.alpha_levels
        for local_training_rows in configuration.protocol_matrix.local_training_row_counts
    )

    assert len(variants) == 10
    assert len({variant.scientific_digest for variant in variants}) == 10
    assert tuple(variant.scientific.alpha for variant in variants) == (
        *(AlphaLevel.ALPHA_001,) * 5,
        *(AlphaLevel.ALPHA_005,) * 5,
    )
    assert (
        tuple(variant.scientific.local_detectors.local_training_rows for variant in variants)
        == (
            100,
            400,
            750,
            2000,
            20000,
        )
        * 2
    )
    assert configuration.scientific.alpha is AlphaLevel.ALPHA_005
    assert configuration.scientific.local_detectors.local_training_rows == 400
    alternate_contents = (
        (REPOSITORY_ROOT / "configs" / "default.yaml")
        .read_text(encoding="utf-8")
        .replace("  alpha: ALPHA_005", "  alpha: ALPHA_001")
        .replace("    local_training_rows: 400", "    local_training_rows: 100")
    )
    alternate_path = tmp_path / "alternate.yaml"
    alternate_path.write_text(alternate_contents, encoding="utf-8")
    independently_loaded = load_configuration(alternate_path, tmp_path)
    assert variants[0].scientific_digest == independently_loaded.scientific_digest


def test_configuration_variant_rejects_values_outside_matrix_and_detector_limits(
    tmp_path: Path,
) -> None:
    configuration = load_configuration(REPOSITORY_ROOT / "configs" / "default.yaml", tmp_path)
    alpha_limited = configuration.model_copy(
        update={
            "protocol_matrix": protocol_matrix(
                configuration.protocol_matrix.local_training_row_counts, (AlphaLevel.ALPHA_005,)
            )
        }
    )
    with pytest.raises(ConfigurationError, match="outside the configured protocol matrix"):
        configuration_for_protocol_variant(
            alpha_limited,
            AlphaLevel.ALPHA_001,
            400,
        )

    rows_limited = configuration.model_copy(
        update={
            "protocol_matrix": protocol_matrix((10,), configuration.protocol_matrix.alpha_levels)
        }
    )
    with pytest.raises(ConfigurationError, match="too small for configured detectors"):
        configuration_for_protocol_variant(
            rows_limited,
            AlphaLevel.ALPHA_005,
            10,
        )


def test_configuration_models_round_trip_serialized_scalar_wrappers(tmp_path: Path) -> None:
    configuration = load_configuration(REPOSITORY_ROOT / "configs" / "default.yaml", tmp_path)
    serialized = configuration.model_dump(mode="python")
    serialized["scientific_digest"] = configuration.scientific_digest
    serialized["reporting_digest"] = configuration.reporting_digest

    restored = ResolvedConfiguration.model_validate(serialized, strict=True)

    assert restored == configuration


def test_scientific_digest_ignores_runtime_path_and_logging(tmp_path: Path) -> None:
    source = REPOSITORY_ROOT / "configs" / "default.yaml"
    first = load_configuration(source, tmp_path / "first")
    second = load_configuration(source, tmp_path / "second")
    assert first.scientific_digest == second.scientific_digest


def test_scientific_digest_includes_chronological_split_protocol(tmp_path: Path) -> None:
    source = (REPOSITORY_ROOT / "configs" / "default.yaml").read_text(encoding="utf-8")
    changed = tmp_path / "changed-split.yaml"
    changed.write_text(
        source.replace("training_fraction: 0.4", "training_fraction: 0.3").replace(
            "test_fraction: 0.2", "test_fraction: 0.3"
        ),
        encoding="utf-8",
    )
    default = load_configuration(REPOSITORY_ROOT / "configs" / "default.yaml", tmp_path)
    altered = load_configuration(changed, tmp_path)
    assert default.scientific_digest != altered.scientific_digest


def test_reporting_configuration_has_a_separate_deterministic_digest(tmp_path: Path) -> None:
    source = (REPOSITORY_ROOT / "configs" / "default.yaml").read_text(encoding="utf-8")
    changed = tmp_path / "changed-reporting.yaml"
    changed.write_text(
        source.replace("bootstrap_seed: 314159", "bootstrap_seed: 271828"),
        encoding="utf-8",
    )
    default = load_configuration(REPOSITORY_ROOT / "configs" / "default.yaml", tmp_path)
    altered = load_configuration(changed, tmp_path)
    assert default.scientific_digest == altered.scientific_digest
    assert default.reporting_digest != altered.reporting_digest


def test_configuration_rejects_an_empty_study_arm_list(tmp_path: Path) -> None:
    source = (REPOSITORY_ROOT / "configs" / "default.yaml").read_text(encoding="utf-8")
    changed = tmp_path / "no-arms.yaml"
    arm_list = source.split("  study_arms:\n", 1)[1].split("runtime:\n", 1)[0]
    changed.write_text(
        source.replace(f"  study_arms:\n{arm_list}", "  study_arms: []\n"),
        encoding="utf-8",
    )

    with pytest.raises(ConfigurationError, match="at least one study arm"):
        load_configuration(changed, tmp_path)


def test_configuration_rejects_duplicate_learning_seeds(tmp_path: Path) -> None:
    source = (REPOSITORY_ROOT / "configs" / "default.yaml").read_text(encoding="utf-8")
    changed = tmp_path / "duplicate-seeds.yaml"
    changed.write_text(
        source.replace("learning_seeds: [6, 7, 8]", "learning_seeds: [0, 1, 1]"), encoding="utf-8"
    )

    with pytest.raises(ConfigurationError):
        load_configuration(changed, tmp_path)


def test_unknown_configuration_keys_are_rejected(tmp_path: Path) -> None:
    configuration_path = tmp_path / "invalid.yaml"
    configuration_path.write_text(
        "scientific:\n  alpha: ALPHA_005\n  reserve_fraction: 0.5\n"
        "  calibration_block_size: 250\n  minimum_peer_count: 4\n"
        "  seed: 17\n  joint_scale_cap: 64.0\n"
        "  joint_scale_iterations: 20\n"
        "  peer_training:\n    hidden_layer_widths: [64, 16]\n"
        "    epochs: 8\n    batch_size: 256\n    learning_rate: 0.001\n"
        "  extra_setting: 2\n"
        "runtime:\n  raw_data_root: data/raw\n  output_root: outputs\n  log_level: INFO\n"
        "splitting:\n  training_fraction: 0.4\n  reference_fraction: 0.2\n"
        "  calibration_fraction: 0.2\n  test_fraction: 0.2\n"
        "  minimum_calibration_rows: 350\n  minimum_attack_rows: 600\n"
        "  attack_rows_per_half_cap: 10000\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigurationError):
        load_configuration(configuration_path, tmp_path)


@pytest.mark.parametrize(
    ("existing", "replacement"),
    [
        ("alpha: ALPHA_005", "alpha: unsupported"),
        ("reserve_fraction: 0.5", "reserve_fraction: 1.0"),
        ("calibration_block_size: 250", "calibration_block_size: 0"),
        ("minimum_peer_count: 4", "minimum_peer_count: 3"),
        ("seed: 17", "seed: -1"),
        ("joint_scale_cap: 64.0", "joint_scale_cap: 0.5"),
        ("joint_scale_iterations: 20", "joint_scale_iterations: 0"),
        ("hidden_layer_widths: [64, 16]", "hidden_layer_widths: [64]"),
        ("epochs: 8", "epochs: 0"),
        ("batch_size: 256", "batch_size: 0"),
        ("learning_rate: 0.001", "learning_rate: 0.0"),
        ("hidden_layer_widths: [64, 16]", "hidden_layer_widths: [64, true]"),
        ("autoencoder_hidden_widths: [20, 10]", "autoencoder_hidden_widths: [20]"),
        ("autoencoder_epochs: 30", "autoencoder_epochs: false"),
        ("autoencoder_learning_rate: 0.001", "autoencoder_learning_rate: 0.0"),
        ("autoencoder_batch_size: 256", "autoencoder_batch_size: 256.0"),
        ("detector_sample_limit: 8000", "detector_sample_limit: true"),
        ("isolation_forest_trees: 100", "isolation_forest_trees: 100.0"),
        ("pca_explained_variance: 0.95", "pca_explained_variance: '0.95'"),
        ("covariance_regularization: 0.001", "covariance_regularization: '0.001'"),
        ("lof_neighbors: 20", "lof_neighbors: 8000"),
        ("nearest_neighbors: 10", "nearest_neighbors: 8000"),
        ("pca_explained_variance: 0.95", "pca_explained_variance: 1.0"),
        ("covariance_regularization: 0.001", "covariance_regularization: -0.001"),
        ("detector_sample_limit: 8000", "detector_sample_limit: 20"),
        ("local_training_rows: 400", "local_training_rows: 20"),
        ("output_root: outputs", "output_root: ''"),
        ("raw_data_root: data/raw", "raw_data_root: ''"),
        ("log_level: INFO", "log_level: unsupported"),
        ("execution_device: CPU", "execution_device: unsupported"),
        ("calibration_fraction: 0.2", "calibration_fraction: 0.3"),
        ("minimum_attack_rows: 600", "minimum_attack_rows: 1"),
        ("training_fraction: 0.4", "training_fraction: 1"),
        ("minimum_calibration_rows: 350", "minimum_calibration_rows: 0"),
        ("attack_rows_per_half_cap: 10000", "attack_rows_per_half_cap: 0"),
        ("bootstrap_replicates: 10000", "bootstrap_replicates: 999"),
        ("bootstrap_seed: 314159", "bootstrap_seed: -1"),
        ("confidence_level: 0.95", "confidence_level: 1.0"),
        ("learning_seeds: [6, 7, 8]", "learning_seeds: []"),
        ("test_fraction: 0.2", "test_fraction: 0.1"),
        ("alpha_001: [0.55, 0.75, 1.0, 1.4]", "alpha_001: [0.55, 0.75, 1.0]"),
        ("alpha_levels: [ALPHA_001, ALPHA_005]", "alpha_levels: []"),
        ("training_fraction: 0.4", "training_fraction: '0.4'"),
        ("minimum_calibration_rows: 350", "minimum_calibration_rows: 350.0"),
        ("minimum_attack_rows: 600", "minimum_attack_rows: true"),
        ("attack_rows_per_half_cap: 10000", "attack_rows_per_half_cap: 10000.0"),
        ("bootstrap_replicates: 10000", "bootstrap_replicates: true"),
        ("bootstrap_seed: 314159", "bootstrap_seed: 314159.0"),
        ("confidence_level: 0.95", "confidence_level: '0.95'"),
        ("alpha_levels: [ALPHA_001, ALPHA_005]", "alpha_levels: [ALPHA_001, ALPHA_001]"),
        ("provenance_active: true", "provenance_active: 'true'"),
        ("provenance_active: true", "provenance_active: 1"),
        (
            "local_training_row_counts: [100, 400, 750, 2000, 20000]",
            "local_training_row_counts: [400, 100, 750, 2000, 20000]",
        ),
        ("alpha_001: [0.55, 0.75, 1.0, 1.4]", "alpha_001: [0.55, '0.75', 1.0, 1.4]"),
    ],
)
def test_invalid_semantic_settings_are_rejected(
    tmp_path: Path, existing: str, replacement: str
) -> None:
    source = (REPOSITORY_ROOT / "configs" / "default.yaml").read_text(encoding="utf-8")
    configuration_path = tmp_path / "invalid.yaml"
    configuration_path.write_text(source.replace(existing, replacement), encoding="utf-8")
    with pytest.raises(ConfigurationError):
        load_configuration(configuration_path, tmp_path)


def test_configuration_accepts_already_validated_domain_values() -> None:
    configuration = load_configuration(REPOSITORY_ROOT / "configs" / "default.yaml", Path.cwd())
    scientific = ScientificSettings(
        alpha=AlphaLevel.ALPHA_005,
        reserve_fraction=0.5,
        calibration_block_size=250,
        minimum_peer_count=4,
        seed=17,
        learning_seeds=(0, 1, 2),
        joint_scale_cap=64.0,
        joint_scale_iterations=20,
        budget_sweep_multipliers=configuration.scientific.budget_sweep_multipliers,
        peer_training=PeerTrainingSettings(
            hidden_layer_widths=(64, 16),
            epochs=8,
            batch_size=256,
            learning_rate=0.001,
        ),
        local_detectors=configuration.scientific.local_detectors,
        local_bank=configuration.scientific.local_bank,
    )
    assert tuple(
        multiplier
        for multiplier in scientific.budget_sweep_multipliers.for_alpha(AlphaLevel.ALPHA_001)
    ) == (0.55, 0.75, 1.0, 1.4)
    assert tuple(
        multiplier
        for multiplier in scientific.budget_sweep_multipliers.for_alpha(AlphaLevel.ALPHA_005)
    ) == (0.4, 0.75, 1.0, 1.4)
    with pytest.raises(ValidationError, match="do not match the evaluation protocol"):
        BudgetSweepSettings(
            alpha_001=(
                0.4,
                0.75,
                1.0,
                1.4,
            ),
            alpha_005=configuration.scientific.budget_sweep_multipliers.alpha_005,
        )
    runtime = RuntimeSettings(
        raw_data_root=Path("data/raw"),
        output_root=Path("outputs"),
        log_level=LogLevel.INFO,
        execution_device=configuration.runtime.execution_device,
        lab_network=configuration.runtime.lab_network,
    )
    splitting = SplitSettings(
        training_fraction=0.4,
        reference_fraction=0.2,
        calibration_fraction=0.2,
        test_fraction=0.2,
        minimum_calibration_rows=350,
        minimum_attack_rows=600,
        attack_rows_per_half_cap=10000,
    )
    assert scientific.alpha is AlphaLevel.ALPHA_005
    assert runtime.log_level is LogLevel.INFO
    assert splitting.attack_rows_per_half_cap == 10000


def _section(name: str) -> dict[str, Any]:
    text = (REPOSITORY_ROOT / "configs" / "default.yaml").read_text(encoding="utf-8")
    return yaml.safe_load(text)[name]


@pytest.mark.parametrize(
    ("section", "model", "changes"),
    [
        ("scientific", ScientificSettings, {"reserve_fraction": 1}),
        ("scientific", ScientificSettings, {"reserve_fraction": True}),
        ("scientific", ScientificSettings, {"calibration_block_size": 250.0}),
        ("scientific", ScientificSettings, {"calibration_block_size": "250"}),
        ("scientific", ScientificSettings, {"calibration_block_size": 0}),
        ("scientific", ScientificSettings, {"alpha": "ALPHA_999"}),
        ("scientific", ScientificSettings, {"alpha": "alpha_005"}),
        ("scientific", ScientificSettings, {"learning_seeds": [0, 0]}),
        ("scientific", ScientificSettings, {"learning_seeds": []}),
        ("scientific", ScientificSettings, {"learning_seeds": "012"}),
        ("scientific", ScientificSettings, {"minimum_peer_count": 3}),
        ("scientific", ScientificSettings, {"study_arms": ["RESERVE_QUARTER", "RESERVE_QUARTER"]}),
        ("scientific", ScientificSettings, {"study_arms": ["NOT_AN_ARM"]}),
        ("scientific", ScientificSettings, {"unexpected": 1}),
        ("runtime", RuntimeSettings, {"raw_data_root": ""}),
        ("runtime", RuntimeSettings, {"execution_device": "TPU"}),
        ("runtime", RuntimeSettings, {"log_level": "LOUD"}),
        ("splitting", SplitSettings, {"training_fraction": 0.5}),
        ("protocol_matrix", ProtocolMatrixSettings, {"alpha_levels": ["ALPHA_001", "ALPHA_001"]}),
        ("protocol_matrix", ProtocolMatrixSettings, {"local_training_row_counts": [400, 100]}),
        ("protocol_matrix", ProtocolMatrixSettings, {"analyses": []}),
        (
            "protocol_matrix",
            ProtocolMatrixSettings,
            {"analyses": [{"kind": "BROAD_SCALING", "budgets": [100, 400, 2000, 20000]}]},
        ),
        (
            "protocol_matrix",
            ProtocolMatrixSettings,
            {
                "analyses": [
                    {"kind": "UNIVERSAL_PRIMARY", "budgets": [100, 400]},
                    {"kind": "UNIVERSAL_PRIMARY", "budgets": [100, 400, 750, 2000, 20000]},
                ]
            },
        ),
        (
            "protocol_matrix",
            ProtocolMatrixSettings,
            {
                "analyses": [
                    {"kind": "UNIVERSAL_PRIMARY", "budgets": [400, 100]},
                    {"kind": "HIGH_RESOURCE", "budgets": [750, 2000, 20000]},
                ]
            },
        ),
        (
            "protocol_matrix",
            ProtocolMatrixSettings,
            {
                "power": {
                    "minimum_meaningful_effect": 0.02,
                    "difference_sds": [0.2, 0.1],
                    "significance_level": 0.05,
                    "power": 0.8,
                }
            },
        ),
        (
            "protocol_matrix",
            ProtocolMatrixSettings,
            {
                "power": {
                    "minimum_meaningful_effect": 0.0,
                    "difference_sds": [0.1],
                    "significance_level": 0.05,
                    "power": 0.8,
                }
            },
        ),
        ("reporting", ReportingSettings, {"bootstrap_replicates": 999}),
        ("reporting", ReportingSettings, {"confidence_level": 1}),
    ],
)
def test_yaml_values_are_validated_strictly(
    section: str, model: type[BaseModel], changes: dict[str, Any]
) -> None:
    values = _section(section)
    values.update(changes)
    payload = json.dumps(values)
    with pytest.raises(ValidationError):
        model.model_validate_json(payload, strict=True)


def test_nested_yaml_values_are_validated_strictly() -> None:
    peer = _section("scientific")
    peer["peer_training"]["hidden_layer_widths"] = [64.0, 16]
    payload = json.dumps(peer)
    with pytest.raises(ValidationError):
        ScientificSettings.model_validate_json(payload, strict=True)
    peer = _section("scientific")
    peer["peer_training"]["hidden_layer_widths"] = [64, 16, 2]
    payload = json.dumps(peer)
    with pytest.raises(ValidationError):
        ScientificSettings.model_validate_json(payload, strict=True)
    sweep = _section("scientific")
    sweep["budget_sweep_multipliers"]["alpha_005"] = [0.75, 0.4, 1.0, 1.4]
    payload = json.dumps(sweep)
    with pytest.raises(ValidationError):
        ScientificSettings.model_validate_json(payload, strict=True)
    split = _section("splitting")
    split["attack_minimum_overrides"][0]["study_stratum"] = "NOT_A_STRATUM"
    payload = json.dumps(split)
    with pytest.raises(ValidationError):
        SplitSettings.model_validate_json(payload, strict=True)
    split = _section("splitting")
    split["attack_minimum_overrides"].append(dict(split["attack_minimum_overrides"][0]))
    payload = json.dumps(split)
    with pytest.raises(ValidationError):
        SplitSettings.model_validate_json(payload, strict=True)


def test_valid_yaml_scalars_enums_and_sequences_are_lifted() -> None:
    scientific = _section("scientific")
    scientific.update({"reserve_fraction": 0.25, "study_arms": ["RESERVE_QUARTER"]})
    parsed = ScientificSettings.model_validate_json(json.dumps(scientific), strict=True)
    assert parsed.reserve_fraction == 0.25
    assert parsed.study_arms == (StudyArm.RESERVE_QUARTER,)
    assert ScientificSettings.model_validate_json(parsed.model_dump_json(), strict=True) == parsed


def test_provenance_can_be_switched_off_explicitly_and_survives_path_resolution(
    tmp_path: Path,
) -> None:
    source = (REPOSITORY_ROOT / "configs" / "default.yaml").read_text(encoding="utf-8")
    relaxed = tmp_path / "relaxed.yaml"
    relaxed.write_text(
        source.replace("provenance_active: true", "provenance_active: false"), encoding="utf-8"
    )
    omitted = tmp_path / "omitted.yaml"
    omitted.write_text(source.replace("  provenance_active: true\n", ""), encoding="utf-8")

    assert load_configuration(relaxed, tmp_path).runtime.provenance_active is False
    assert load_configuration(omitted, tmp_path).runtime.provenance_active is True
    assert (
        load_configuration(relaxed, tmp_path).scientific_digest
        == load_configuration(
            REPOSITORY_ROOT / "configs" / "default.yaml", tmp_path
        ).scientific_digest
    )
