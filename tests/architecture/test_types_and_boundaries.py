import ast
import re
from collections import Counter
from pathlib import Path

from tests.architecture.source_index import (
    REPOSITORY_ROOT,
    SOURCE_ROOT,
    TYPE_MODULE,
    annotation_names,
    parse_source,
    production_files,
    source_location,
)

RAW_TYPES = {"Any", "bool", "bytes", "complex", "float", "int", "object", "str"}
SEMANTIC_FIELDS = {
    "ScientificSettings": {
        "alpha": "AlphaLevel",
        "reserve_fraction": "ReserveFraction",
        "calibration_block_size": "CalibrationBlockSize",
        "minimum_peer_count": "MinimumPeerCount",
        "seed": "LearningSeed",
        "learning_seeds": "tuple[LearningSeed, ...]",
        "joint_scale_cap": "JointScaleCap",
        "joint_scale_iterations": "ScaleSearchIterations",
        "budget_sweep_multipliers": "BudgetSweepSettings",
        "peer_training": "PeerTrainingSettings",
        "local_detectors": "LocalDetectorSettings",
        "local_bank": "LocalBankSettings",
    },
    "LocalBankSettings": {
        "response_forest_trees": "DetectorCount",
        "pair_count": "PairCount",
        "pair_bins": "QuantileBinCount",
        "pair_smoothing": "CountSmoothing",
        "anchor_share": "AnchorShare",
    },
    "BudgetSweepSettings": {
        "alpha_001": "tuple[BudgetMultiplier, ...]",
        "alpha_005": "tuple[BudgetMultiplier, ...]",
    },
    "PeerTrainingSettings": {
        "hidden_layer_widths": "tuple[LayerWidth, LayerWidth]",
        "epochs": "TrainingEpochCount",
        "batch_size": "TrainingBatchSize",
        "learning_rate": "LearningRate",
    },
    "LocalDetectorSettings": {
        "autoencoder_hidden_widths": "tuple[LayerWidth, LayerWidth]",
        "autoencoder_epochs": "TrainingEpochCount",
        "autoencoder_batch_size": "TrainingBatchSize",
        "autoencoder_learning_rate": "LearningRate",
        "local_training_rows": "LocalTrainingRowCount",
        "detector_sample_limit": "DetectorSampleLimit",
        "pca_explained_variance": "PCAExplainedVariance",
        "isolation_forest_trees": "DetectorCount",
        "lof_neighbors": "DetectorCount",
        "nearest_neighbors": "DetectorCount",
        "histogram_bins": "DetectorCount",
        "covariance_regularization": "CovarianceRegularization",
    },
    "RuntimeSettings": {
        "raw_data_root": "Location",
        "output_root": "Location",
        "log_level": "LogLevel",
        "execution_device": "ExecutionDevice",
    },
    "ResolvedConfiguration": {
        "scientific": "ScientificSettings",
        "runtime": "RuntimeSettings",
        "splitting": "SplitSettings",
        "reporting": "ReportingSettings",
        "scientific_digest": "ScientificDigest",
        "reporting_digest": "ScientificDigest",
    },
    "DatasetSourceInventory": {
        "dataset": "DatasetName",
        "location": "Path",
        "availability": "DatasetAvailability",
        "file_count": "SourceFileCount",
        "byte_count": "SourceByteCount",
    },
    "DatasetInventoryReport": {
        "data_root": "Path",
        "sources": "tuple[DatasetSourceInventory, ...]",
    },
    "DatasetSplitPlan": {
        "client": "DatasetClientId",
        "benign_roles": "ChronologicalIndices",
        "attack_splits": "tuple[AttackCaptureSplit, ...]",
        "excluded_attack_types": "tuple[AttackType, ...]",
    },
    "AttackType": {"identifier": "AttackTypeName"},
    "AttackCapture": {
        "attack_type": "AttackType",
        "source_file": "Path",
        "features": "FloatMatrix",
        "source_order": "RowIndexArray",
    },
    "AttackCaptureSplit": {
        "attack_type": "AttackType",
        "indices": "AttackHalfIndices",
    },
    "AttackHalfIndices": {
        "peer_development": "RowIndexArray",
        "held_out": "RowIndexArray",
    },
    "ChronologicalIndices": {
        "training": "RowIndexArray",
        "reference": "RowIndexArray",
        "calibration": "RowIndexArray",
        "test": "RowIndexArray",
    },
    "DatasetClientId": {"dataset": "DatasetName", "name": "ClientName"},
    "DatasetClientData": {
        "client": "DatasetClientId",
        "target_cohort_role": "TargetCohortRole",
        "feature_names": "tuple[FeatureName, ...]",
        "benign_features": "FloatMatrix",
        "benign_source_order": "RowIndexArray",
        "attack_captures": "tuple[AttackCapture, ...]",
    },
    "SplitSettings": {
        "training_fraction": "SplitFraction",
        "reference_fraction": "SplitFraction",
        "calibration_fraction": "SplitFraction",
        "test_fraction": "SplitFraction",
        "minimum_calibration_rows": "MinimumCalibrationRows",
        "minimum_attack_rows": "MinimumAttackRows",
        "attack_minimum_overrides": "tuple[AttackMinimumOverride, ...]",
        "attack_rows_per_half_cap": "AttackRowsPerHalfCap",
    },
    "AttackMinimumOverride": {
        "study_stratum": "StudyStratum",
        "minimum_attack_rows": "MinimumAttackRows",
    },
    "ReportingSettings": {
        "bootstrap_replicates": "BootstrapReplicateCount",
        "bootstrap_seed": "BootstrapSeed",
        "confidence_level": "ConfidenceLevel",
    },
    "PeerMLP": {
        "layers": "tuple[DenseLayer, DenseLayer, DenseLayer]",
        "feature_count": "FeatureCount",
    },
    "DetectionOutcome": {
        "expert_ids": "tuple[ExpertId, ...]",
        "local_threshold": "Threshold",
        "expert_thresholds": "tuple[Threshold, ...]",
        "joint_multiplier": "JointMultiplier",
        "peer_block_alert_rates": "tuple[FalseAlertRate, ...]",
        "evaluation_count": "ObservationCount",
        "alert_count": "ObservationCount",
        "benign_alert_rate": "FalseAlertRate",
        "attack_detection_rate": "DetectionRate",
        "local_alerts": "tuple[AlertDecision, ...]",
        "peer_alerts": "tuple[AlertDecision, ...]",
        "alerts": "tuple[AlertDecision, ...]",
    },
    "AttackTypePerformance": {
        "attack_type": "AttackType",
        "test_rows": "ObservationCount",
        "detection_rate": "DetectionRate",
        "local_detection_rate": "DetectionRate",
        "max_fusion_detection_rate": "DetectionRate",
        "bonferroni_detection_rate": "DetectionRate",
    },
    "TargetPerformance": {
        "target": "DatasetClientId",
        "seed": "LearningSeed",
        "split_digest": "ScientificDigest",
        "peer_model_seeds": "tuple[ModelSeedProvenance, ...]",
        "local_model_seed": "ModelSeedProvenance",
        "benign_test_rows": "ObservationCount",
        "realized_false_alert_rate": "FalseAlertRate",
        "attack_types": "tuple[AttackTypePerformance, ...]",
        "local_false_alert_rate": "FalseAlertRate",
        "max_fusion_false_alert_rate": "FalseAlertRate",
        "bonferroni_false_alert_rate": "FalseAlertRate",
        "peer_count": "MinimumPeerCount",
        "budget_sweep": "tuple[BudgetSweepPerformance, ...]",
    },
    "BudgetSweepPerformance": {
        "multiplier": "BudgetMultiplier",
        "detection_rate": "DetectionRate",
        "false_alert_rate": "FalseAlertRate",
    },
    "ModelSeedProvenance": {
        "owner": "DatasetClientId",
        "seed": "LearningSeed",
        "domain": "SeedDomain",
    },
    "TargetMetricSummary": {
        "target": "DatasetClientId",
        "method_detection_rate": "DetectionRate",
        "local_detection_rate": "DetectionRate",
        "max_fusion_detection_rate": "DetectionRate",
        "bonferroni_detection_rate": "DetectionRate",
        "detection_difference": "RateDifference",
        "max_fusion_detection_difference": "RateDifference",
        "method_max_fusion_difference": "RateDifference",
        "realized_false_alert_rate": "FalseAlertRate",
        "local_false_alert_rate": "FalseAlertRate",
        "max_fusion_false_alert_rate": "FalseAlertRate",
        "bonferroni_false_alert_rate": "FalseAlertRate",
        "matched_fpr_difference": "RateDifference | None",
    },
    "CampaignMetricSummary": {
        "target_count": "ObservationCount",
        "targets": "tuple[TargetMetricSummary, ...]",
        "macro_detection_rate": "DetectionRate",
        "local_macro_detection_rate": "DetectionRate",
        "max_fusion_macro_detection_rate": "DetectionRate",
        "bonferroni_macro_detection_rate": "DetectionRate",
        "macro_detection_difference": "RateDifference",
        "method_local_confidence_interval": "RateConfidenceInterval",
        "max_fusion_macro_detection_difference": "RateDifference",
        "method_max_fusion_confidence_interval": "RateConfidenceInterval",
        "matched_fpr_target_count": "ObservationCount",
        "matched_fpr_macro_difference": "RateDifference | None",
        "matched_fpr_confidence_interval": "RateConfidenceInterval | None",
        "mean_method_max_fusion_difference": "RateDifference",
        "worst_method_max_fusion_difference": "RateDifference",
        "improved_target_share": "FalseAlertRate",
        "worst_detection_difference": "RateDifference",
        "tenth_percentile_detection_difference": "RateDifference",
        "catastrophic_cell_count": "ObservationCount",
        "harmed_target_count": "ObservationCount",
        "mean_false_alert_rate": "FalseAlertRate",
        "median_false_alert_rate": "FalseAlertRate",
        "ninetieth_percentile_false_alert_rate": "FalseAlertRate",
        "worst_false_alert_rate": "FalseAlertRate",
        "exceedance_share": "FalseAlertRate",
        "local_mean_false_alert_rate": "FalseAlertRate",
        "max_fusion_mean_false_alert_rate": "FalseAlertRate",
        "max_fusion_worst_false_alert_rate": "FalseAlertRate",
        "bonferroni_mean_false_alert_rate": "FalseAlertRate",
        "bonferroni_worst_false_alert_rate": "FalseAlertRate",
    },
    "CampaignReportResult": {
        "artifact_path": "ArtifactPath",
        "metrics": "CampaignMetricSummary",
        "reuse_state": "ArtifactReuseState",
    },
    "CampaignWorkflowResult": {
        "artifact_path": "ArtifactPath",
        "verification": "ArtifactVerification",
        "reuse_state": "ArtifactReuseState",
        "target_count": "ObservationCount",
    },
    "CampaignStatus": {
        "dataset": "DatasetName",
        "artifact_path": "ArtifactPath",
        "state": "CampaignCompletionState",
        "target_count": "ObservationCount | None",
    },
    "ChannelDecisions": {
        "local_threshold": "Threshold",
        "expert_thresholds": "tuple[Threshold, ...]",
        "joint_multiplier": "JointMultiplier",
        "peer_block_alert_rates": "tuple[FalseAlertRate, ...]",
        "local_alerts": "BooleanArray",
        "peer_alerts": "BooleanArray",
        "combined_alerts": "BooleanArray",
    },
    "SyntheticOutcome": {
        "fixture_kind": "FixtureKind",
        "row_count": "ObservationCount",
        "detected_count": "ObservationCount",
        "local_alert_count": "ObservationCount",
        "peer_alert_count": "ObservationCount",
        "benign_alert_rate": "FalseAlertRate",
        "attack_detection_rate": "DetectionRate",
        "local_threshold": "Threshold",
        "expert_thresholds": "tuple[Threshold, ...]",
        "joint_multiplier": "JointMultiplier",
        "peer_block_alert_rates": "tuple[FalseAlertRate, ...]",
        "alerts": "tuple[AlertDecision, ...]",
    },
    "CampaignOutcome": {"targets": "tuple[TargetPerformance, ...]"},
    "EvidenceArtifact": {
        "schema_version": "ArtifactSchemaVersion",
        "artifact_kind": "ArtifactKind",
        "scientific_configuration": "ScientificSettings",
        "splitting_configuration": "SplitSettings",
        "configuration_digest": "ScientificDigest",
        "input_digest": "ScientificDigest",
        "implementation_digest": "ScientificDigest",
        "verification_state": "VerificationState",
        "outcome": "SyntheticOutcome | CampaignOutcome",
        "content_digest": "ScientificDigest",
    },
}


def _class_nodes(tree: ast.Module) -> dict[str, ast.ClassDef]:
    return {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}


def _class_fields(node: ast.ClassDef) -> dict[str, ast.expr]:
    fields: dict[str, ast.expr] = {}
    for assignment in node.body:
        if isinstance(assignment, ast.AnnAssign) and isinstance(assignment.target, ast.Name):
            fields[assignment.target.id] = assignment.annotation
    return fields


def _type_alias_statement(node: ast.stmt) -> bool:
    if isinstance(node, ast.Assign) and len(node.targets) == 1:
        target = node.targets[0]
        return (
            isinstance(target, ast.Name)
            and target.id[:1].isupper()
            and not target.id.isupper()
            and isinstance(
                node.value,
                ast.Name | ast.Attribute | ast.BinOp | ast.Subscript | ast.Call,
            )
        )
    if isinstance(node, ast.AnnAssign):
        return (
            isinstance(node.target, ast.Name)
            and node.target.id[:1].isupper()
            and not node.target.id.isupper()
            and ast.unparse(node.annotation).endswith("TypeAlias")
        )
    return False


def test_production_source_scan_is_complete() -> None:
    expected = set(SOURCE_ROOT.rglob("*.py"))
    scanned = set(production_files())
    assert expected == scanned


def _without_yaml_markers(annotation: str) -> str:
    annotation = re.sub(r"Annotated\[(.*), YamlSequence\]", r"\1", annotation)
    return re.sub(r"Yaml\[(\w+)\]", r"\1", annotation)


def test_semantic_configuration_and_result_fields_use_domain_types() -> None:
    source_by_path = {
        path.relative_to(SOURCE_ROOT).as_posix(): parse_source(path) for path in production_files()
    }
    for class_name, expected_fields in SEMANTIC_FIELDS.items():
        matching = [
            node
            for tree in source_by_path.values()
            for node in _class_nodes(tree).values()
            if node.name == class_name
        ]
        assert len(matching) == 1, class_name
        actual_fields = _class_fields(matching[0])
        for field_name, expected_type in expected_fields.items():
            annotation = actual_fields[field_name]
            assert _without_yaml_markers(ast.unparse(annotation)) == expected_type
            assert not annotation_names(annotation) & RAW_TYPES


def test_semantic_type_aliases_are_owned_by_types_module() -> None:
    offenders = [
        source_location(path, node.lineno)
        for path in production_files()
        if path != TYPE_MODULE
        for node in ast.walk(parse_source(path))
        if isinstance(node, ast.Assign | ast.AnnAssign) and _type_alias_statement(node)
    ]
    assert not offenders, offenders


def test_type_alias_ownership_rule_detects_primitive_laundering() -> None:
    mutations = (
        "PolicyName = str",
        "ClientCount = int",
        "FeatureValues = NDArray[np.float64]",
        "Payload: TypeAlias = dict[str, object]",
    )
    for mutation in mutations:
        tree = ast.parse(mutation)
        assert any(_type_alias_statement(node) for node in tree.body), mutation


def test_all_project_enums_and_semantic_value_objects_live_in_types_module() -> None:
    outside = [
        source_location(path, node.lineno)
        for path in production_files()
        if path != TYPE_MODULE
        for node in ast.walk(parse_source(path))
        if isinstance(node, ast.ClassDef)
        and any(
            isinstance(base, ast.Name) and base.id in {"Enum", "IntEnum", "StrEnum"}
            for base in node.bases
        )
    ]
    assert not outside, outside
    names = [
        node.name for node in ast.walk(parse_source(TYPE_MODULE)) if isinstance(node, ast.ClassDef)
    ]
    duplicates = [name for name, count in Counter(names).items() if count > 1]
    assert not duplicates, duplicates


def test_domain_types_file_does_not_depend_on_application_layers() -> None:
    forbidden_packages = {
        "pace.config",
        "pace.core",
        "pace.datasets",
        "pace.experiment",
        "pace.reporting",
        "pace.cli",
    }
    imports = [
        node.module or ""
        for node in ast.walk(parse_source(TYPE_MODULE))
        if isinstance(node, ast.ImportFrom)
    ]
    assert not [
        name for name in imports if any(name.startswith(prefix) for prefix in forbidden_packages)
    ]


def test_numeric_array_aliases_are_authoritative_in_types_module() -> None:
    expected_aliases = {
        "FloatArray": "NDArray[np.float64]",
        "BooleanArray": "NDArray[np.bool_]",
        "ClassLabelArray": "NDArray[np.int64]",
        "RowIndexArray": "NDArray[np.int64]",
        "TimestampArray": "NDArray[np.datetime64]",
    }
    type_tree = parse_source(TYPE_MODULE)
    actual_aliases = {
        node.targets[0].id: ast.unparse(node.value)
        for node in type_tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id in expected_aliases
    }
    assert actual_aliases == expected_aliases
    duplicate_definitions = [
        source_location(path, node.lineno)
        for path in production_files()
        if path != TYPE_MODULE
        for node in ast.walk(parse_source(path))
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id in expected_aliases
            for target in node.targets
        )
    ]
    assert not duplicate_definitions, duplicate_definitions


def test_numeric_array_types_are_not_redeclared_by_application_layers() -> None:
    offenders = [
        source_location(path, node.lineno)
        for path in production_files()
        if path != TYPE_MODULE
        for node in ast.walk(parse_source(path))
        if isinstance(node, ast.ImportFrom) and node.module == "numpy.typing"
    ]
    assert not offenders, offenders


def test_forbidden_organizational_names_do_not_enter_implementation() -> None:
    tokens = _forbidden_tokens()
    roots = (
        SOURCE_ROOT,
        REPOSITORY_ROOT / "tests",
        REPOSITORY_ROOT / "configs",
        REPOSITORY_ROOT / "rules",
    )
    inspected = tuple(
        path
        for root in roots
        for path in root.rglob("*")
        if path.is_file() and path.suffix in {".py", ".yaml", ".yml", ".toml", ".ini"}
    )
    offenders = [
        f"{path.relative_to(REPOSITORY_ROOT)} contains {token}"
        for path in inspected
        for token in tokens
        if token.casefold() in path.read_text(encoding="utf-8").casefold()
    ]
    assert not offenders, offenders


def _forbidden_tokens() -> tuple[str, ...]:
    return (
        "road" + "map",
        "mile" + "stone",
        "phase" + "-" + "1",
        "step" + "-" + "m1",
        "docs/" + "PACE" + ".md",
        "TECHNICAL_" + "ARCHITECTURE" + ".md",
        "U" + "-E",
        "U" + "-A",
        "J" + "2",
        "J" + "9",
        "R" + "0",
    )


def test_forbidden_name_scanner_detects_mutation() -> None:
    mutation = "run_" + "mile" + "stone" + "_1"
    offenders = [token for token in _forbidden_tokens() if token.casefold() in mutation.casefold()]
    assert offenders


def _layer(module_name: str) -> int | None:
    prefixes = (
        ("pace.types", 0),
        ("pace.config", 1),
        ("pace.core", 2),
        ("pace.datasets", 3),
        ("pace.experiment", 4),
        ("pace.reporting", 5),
        ("pace.cli", 6),
    )
    return next((level for prefix, level in prefixes if module_name.startswith(prefix)), None)


def _imports(tree: ast.Module) -> tuple[str, ...]:
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.append(node.module)
    return tuple(imported)


def _dependency_violations(files: tuple[Path, ...]) -> list[str]:
    violations: list[str] = []
    for path in files:
        module_path = path.relative_to(SOURCE_ROOT).with_suffix("").as_posix()
        importer = "pace." + module_path.replace("/", ".")
        importer_layer = _layer(importer)
        if importer_layer is None:
            continue
        for imported in _imports(parse_source(path)):
            imported_layer = _layer(imported)
            if imported_layer is not None and imported_layer > importer_layer:
                violations.append(f"{importer} imports higher layer {imported}")
    return violations


def test_dependencies_only_point_toward_domain_types() -> None:
    violations = _dependency_violations(production_files())
    assert not violations, violations


def test_dependency_scanner_detects_forbidden_direction() -> None:
    fixture = ast.parse("from pace.reporting.analysis import summarize")
    assert _imports(fixture) == ("pace.reporting.analysis",)
    imported_layer = _layer("pace.reporting.analysis")
    importer_layer = _layer("pace.core.evidence")
    assert imported_layer is not None
    assert importer_layer is not None
    assert imported_layer > importer_layer


def test_production_python_has_no_comments_or_docstrings() -> None:
    offenders: list[str] = []
    for path in production_files():
        tree = parse_source(path)
        for node in ast.walk(tree):
            if (
                isinstance(
                    node,
                    ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef,
                )
                and ast.get_docstring(node, clean=False) is not None
            ):
                line_number = 1 if isinstance(node, ast.Module) else node.lineno
                offenders.append(f"{source_location(path, line_number)} has a docstring")
    assert not offenders, offenders
