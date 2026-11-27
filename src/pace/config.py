import json
import math
from pathlib import Path

import yaml
from pydantic import ValidationError, field_validator, model_validator

from pace.types import (
    AlphaLevel,
    AnchorBudgetPolicy,
    AnchorMembers,
    AnchorShare,
    AttackRowsPerHalfCap,
    BankDesign,
    BenignModeCount,
    BoostingIterationCount,
    BootstrapReplicateCount,
    BootstrapSeed,
    BudgetAnalysisKind,
    BudgetMultiplier,
    CalibrationBlockSize,
    CalibrationMode,
    CalibrationVariant,
    ClientName,
    ConditionLimitRule,
    ConfidenceLevel,
    ConfigurationError,
    CountSmoothing,
    CovarianceConditionLimit,
    CovarianceEstimator,
    CovarianceRegularization,
    CrossFitFoldCount,
    DetectorCount,
    DetectorSampleLimit,
    DisplayText,
    EvaluationRole,
    ExecutionDevice,
    ExternalBaseline,
    FalseAlarmBudget,
    FederatedRoundCount,
    FrozenStrictModel,
    HiddenRatio,
    JointScaleCap,
    KitNetClusterSize,
    LabNetwork,
    LayerWidth,
    LearningRate,
    LearningSeed,
    LocalTrainingRowCount,
    Location,
    LogLevel,
    MinimumAttackRows,
    MinimumCalibrationRows,
    MinimumMeaningfulEffect,
    MinimumPeerCount,
    ModeRowCount,
    NegligibleSpread,
    NumericalThreadCount,
    ObservationCount,
    PairCount,
    PairedDifferenceSd,
    PCAExplainedVariance,
    PeerRowLimit,
    ProvenanceActive,
    QuantileBinCount,
    QuantileLevel,
    RecurrentDepth,
    ReferenceRowCount,
    ReplicateCount,
    ReportedTargetCount,
    ReserveFraction,
    ResolutionRows,
    ScaleFloor,
    ScaleFloorShare,
    ScaleSearchIterations,
    ScientificDigest,
    SequenceLookback,
    SignificanceLevel,
    SourceVerification,
    SplitFraction,
    StatisticalPower,
    StudyArm,
    StudyStratum,
    SupervisedBudgetShare,
    SymbolCount,
    SymbolFitRowLimit,
    TextEncoding,
    TrainingBatchSize,
    TrainingEpochCount,
    canonical_json,
    payload_digest,
)


class PeerTrainingSettings(FrozenStrictModel, frozen=True):
    hidden_layer_widths: tuple[LayerWidth, LayerWidth]
    epochs: TrainingEpochCount
    batch_size: TrainingBatchSize
    learning_rate: LearningRate


class LocalDetectorSettings(FrozenStrictModel, frozen=True):
    autoencoder_hidden_widths: tuple[LayerWidth, LayerWidth]
    autoencoder_epochs: TrainingEpochCount
    autoencoder_batch_size: TrainingBatchSize
    autoencoder_learning_rate: LearningRate
    local_training_rows: LocalTrainingRowCount
    detector_sample_limit: DetectorSampleLimit
    pca_explained_variance: PCAExplainedVariance
    isolation_forest_trees: DetectorCount
    lof_neighbors: DetectorCount
    nearest_neighbors: DetectorCount
    histogram_bins: DetectorCount
    covariance_regularization: CovarianceRegularization
    covariance_estimator: CovarianceEstimator
    covariance_condition_limit: CovarianceConditionLimit | None = None
    covariance_condition_rule: ConditionLimitRule = ConditionLimitRule.FIXED

    def with_training_rows(self, rows: LocalTrainingRowCount) -> "LocalDetectorSettings":
        return self.model_copy(update=dict(local_training_rows=rows))

    @model_validator(mode="after")
    def detector_counts_fit_training_limit(self) -> "LocalDetectorSettings":
        neighborhood_size = max(self.lof_neighbors, self.nearest_neighbors)
        if self.local_training_rows <= neighborhood_size:
            raise ValueError("Local training row count must exceed both neighborhood sizes")
        if self.detector_sample_limit <= neighborhood_size:
            raise ValueError("Detector sample limit must exceed both neighborhood sizes")
        return self


class LocalBankSettings(FrozenStrictModel, frozen=True):
    response_forest_trees: DetectorCount
    pair_count: PairCount
    pair_bins: QuantileBinCount
    pair_smoothing: CountSmoothing
    anchor_share: AnchorShare
    anchor_budget_policy: AnchorBudgetPolicy
    design: BankDesign
    anchor_members: AnchorMembers


class BudgetSweepSettings(FrozenStrictModel, frozen=True):
    alpha_001: tuple[BudgetMultiplier, ...]
    alpha_005: tuple[BudgetMultiplier, ...]

    @field_validator("alpha_001", "alpha_005", mode="after")
    @classmethod
    def multipliers_increase(
        cls, value: tuple[BudgetMultiplier, ...]
    ) -> tuple[BudgetMultiplier, ...]:
        if len(value) < 2 or tuple(sorted(set(value))) != value:
            raise ValueError("Budget multipliers must be unique and increasing")
        return value

    @model_validator(mode="after")
    def matches_frozen_sweep(self) -> "BudgetSweepSettings":
        for alpha in AlphaLevel:
            if self.for_alpha(alpha) != alpha.frozen_budget_multipliers:
                raise ValueError(
                    f"{alpha.name} budget multipliers do not match the evaluation protocol"
                )
        return self

    def for_alpha(self, alpha: AlphaLevel) -> tuple[BudgetMultiplier, ...]:
        if alpha is AlphaLevel.ALPHA_001:
            return self.alpha_001
        return self.alpha_005


class KitNetSettings(FrozenStrictModel, frozen=True):
    max_autoencoder_size: KitNetClusterSize
    hidden_ratio: HiddenRatio
    learning_rate: LearningRate
    epochs: TrainingEpochCount


class McdeSettings(FrozenStrictModel, frozen=True):
    supervised_budget_share: SupervisedBudgetShare
    boosting_iterations: BoostingIterationCount
    boosting_learning_rate: LearningRate
    benign_modes: BenignModeCount
    peer_row_limit: PeerRowLimit
    scale_floor: ScaleFloor
    negligible_spread: NegligibleSpread
    scale_floor_share: ScaleFloorShare
    minimum_mode_rows: ModeRowCount
    sparse_reference_rows: ReferenceRowCount


class DiotSettings(FrozenStrictModel, frozen=True):
    symbol_count: SymbolCount
    symbol_fit_row_limit: SymbolFitRowLimit
    unseen_quantile: QuantileLevel
    lookback: SequenceLookback
    hidden_width: LayerWidth
    hidden_layers: RecurrentDepth
    federated_rounds: FederatedRoundCount
    local_epochs: TrainingEpochCount
    fine_tune_epochs: TrainingEpochCount
    batch_size: TrainingBatchSize
    learning_rate: LearningRate
    peer_row_limit: PeerRowLimit


class BaselineSettings(FrozenStrictModel, frozen=True):
    methods: tuple[ExternalBaseline, ...]
    maximum_local_training_rows: LocalTrainingRowCount
    kitnet: KitNetSettings
    mcde: McdeSettings
    diot: DiotSettings

    @field_validator("methods", mode="after")
    @classmethod
    def methods_are_unique(
        cls, value: tuple[ExternalBaseline, ...]
    ) -> tuple[ExternalBaseline, ...]:
        if len(set(value)) != len(value):
            raise ValueError("External baselines must be unique")
        return value


class ScientificSettings(FrozenStrictModel, frozen=True):
    alpha: AlphaLevel
    reserve_fraction: ReserveFraction
    calibration_block_size: CalibrationBlockSize
    minimum_peer_count: MinimumPeerCount
    seed: LearningSeed
    learning_seeds: tuple[LearningSeed, ...]
    joint_scale_cap: JointScaleCap
    joint_scale_iterations: ScaleSearchIterations
    budget_sweep_multipliers: BudgetSweepSettings
    peer_training: PeerTrainingSettings
    local_detectors: LocalDetectorSettings
    local_bank: LocalBankSettings
    calibration_mode: CalibrationMode
    cross_fit_folds: CrossFitFoldCount
    resolution_rows: ResolutionRows
    curve_variants: tuple[CalibrationVariant, ...]
    operating_grid: tuple[FalseAlarmBudget, ...]
    evaluation_role: EvaluationRole
    baselines: BaselineSettings
    study_arms: tuple[StudyArm, ...] = ()
    target_clients: tuple[ClientName, ...] = ()
    numerical_threads: NumericalThreadCount = 1

    def for_variant(
        self,
        alpha: AlphaLevel,
        local_detectors: LocalDetectorSettings,
        target_clients: tuple[ClientName, ...] = (),
    ) -> "ScientificSettings":
        return self.model_copy(
            update=dict(alpha=alpha, local_detectors=local_detectors, target_clients=target_clients)
        )

    @field_validator("learning_seeds", mode="after")
    @classmethod
    def seeds_are_unique(cls, value: tuple[LearningSeed, ...]) -> tuple[LearningSeed, ...]:
        if not value or len(set(value)) != len(value):
            raise ValueError("Learning seeds must be a unique, non-empty sequence")
        return value

    @field_validator("curve_variants", mode="after")
    @classmethod
    def variants_are_unique(
        cls, value: tuple[CalibrationVariant, ...]
    ) -> tuple[CalibrationVariant, ...]:
        if len(set(value)) != len(value):
            raise ValueError("Calibration curve variants must be unique")
        return value

    @field_validator("operating_grid", mode="after")
    @classmethod
    def grid_is_increasing(
        cls, value: tuple[FalseAlarmBudget, ...]
    ) -> tuple[FalseAlarmBudget, ...]:
        if len(value) < 4 or tuple(sorted(set(value))) != value:
            raise ValueError("Operating grid must hold at least four unique, increasing budgets")
        return value

    @field_validator("study_arms", mode="after")
    @classmethod
    def arms_are_unique(cls, value: tuple[StudyArm, ...]) -> tuple[StudyArm, ...]:
        if len(set(value)) != len(value):
            raise ValueError("Study arms must be unique")
        return value

    @field_validator("target_clients", mode="after")
    @classmethod
    def target_clients_are_unique(cls, value: tuple[ClientName, ...]) -> tuple[ClientName, ...]:
        if len(set(value)) != len(value):
            raise ValueError("Target clients must be unique")
        return value


class RuntimeSettings(FrozenStrictModel, frozen=True):
    raw_data_root: Location
    output_root: Location
    log_level: LogLevel
    execution_device: ExecutionDevice
    lab_network: LabNetwork
    provenance_active: ProvenanceActive = True

    @property
    def verification(self) -> SourceVerification:
        return SourceVerification.for_provenance(self.provenance_active)

    def rooted_at(self, repository_root: Path) -> "RuntimeSettings":
        return RuntimeSettings(
            raw_data_root=(repository_root / self.raw_data_root).resolve(),
            output_root=(repository_root / self.output_root).resolve(),
            log_level=self.log_level,
            execution_device=self.execution_device,
            lab_network=self.lab_network,
            provenance_active=self.provenance_active,
        )


class AttackMinimumOverride(FrozenStrictModel, frozen=True):
    study_stratum: StudyStratum
    minimum_attack_rows: MinimumAttackRows


class SplitSettings(FrozenStrictModel, frozen=True):
    training_fraction: SplitFraction
    reference_fraction: SplitFraction
    calibration_fraction: SplitFraction
    test_fraction: SplitFraction
    minimum_calibration_rows: MinimumCalibrationRows
    minimum_attack_rows: MinimumAttackRows
    attack_rows_per_half_cap: AttackRowsPerHalfCap
    attack_minimum_overrides: tuple[AttackMinimumOverride, ...] = ()

    def minimum_attack_rows_for(self, study_stratum: StudyStratum) -> MinimumAttackRows:
        for override in self.attack_minimum_overrides:
            if override.study_stratum is study_stratum:
                return override.minimum_attack_rows
        return self.minimum_attack_rows

    @model_validator(mode="after")
    def fractions_cover_all_rows(self) -> "SplitSettings":
        total = sum(
            (
                self.training_fraction,
                self.reference_fraction,
                self.calibration_fraction,
                self.test_fraction,
            )
        )
        if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("Chronological split fractions must sum to one")
        strata = tuple(override.study_stratum for override in self.attack_minimum_overrides)
        if len(set(strata)) != len(strata):
            raise ValueError("Attack minimum overrides must have unique study strata")
        return self


class CalibrationStudySettings(FrozenStrictModel, frozen=True):
    rows: tuple[ObservationCount, ...]
    budgets: tuple[FalseAlarmBudget, ...]
    expert_counts: tuple[DetectorCount, ...]
    replicates: ReplicateCount
    test_rows: ObservationCount
    attack_rows: ObservationCount
    seed: LearningSeed


class ReportingSettings(FrozenStrictModel, frozen=True):
    bootstrap_replicates: BootstrapReplicateCount
    bootstrap_seed: BootstrapSeed
    confidence_level: ConfidenceLevel
    high_false_alert_targets: ReportedTargetCount
    benchmark_targets: ReportedTargetCount
    calibration_study: CalibrationStudySettings


class AnalysisSpec(FrozenStrictModel, frozen=True):
    kind: BudgetAnalysisKind
    budgets: tuple[LocalTrainingRowCount, ...]

    @field_validator("budgets", mode="after")
    @classmethod
    def budgets_increase(
        cls, value: tuple[LocalTrainingRowCount, ...]
    ) -> tuple[LocalTrainingRowCount, ...]:
        if not value or tuple(sorted(set(value))) != value:
            raise ValueError("Analysis budgets must be unique, increasing and non-empty")
        return value


class PowerSettings(FrozenStrictModel, frozen=True):
    minimum_meaningful_effect: MinimumMeaningfulEffect
    difference_sds: tuple[PairedDifferenceSd, ...]
    significance_level: SignificanceLevel
    power: StatisticalPower

    @field_validator("difference_sds", mode="after")
    @classmethod
    def sds_increase(cls, value: tuple[PairedDifferenceSd, ...]) -> tuple[PairedDifferenceSd, ...]:
        if not value or tuple(sorted(set(value))) != value:
            raise ValueError("Assumed difference standard deviations must be unique and increasing")
        return value


class ProtocolMatrixSettings(FrozenStrictModel, frozen=True):
    alpha_levels: tuple[AlphaLevel, ...]
    local_training_row_counts: tuple[LocalTrainingRowCount, ...]
    analyses: tuple[AnalysisSpec, ...]
    power: PowerSettings

    @model_validator(mode="after")
    def analyses_cover_the_requested_budgets(self) -> "ProtocolMatrixSettings":
        kinds = tuple(spec.kind for spec in self.analyses)
        if not kinds or len(set(kinds)) != len(kinds):
            raise ValueError("Protocol analyses must be unique and non-empty")
        used = {budget for spec in self.analyses for budget in spec.budgets}
        if used != set(self.local_training_row_counts):
            raise ValueError("Protocol analyses must use exactly the requested training budgets")
        return self

    @field_validator("alpha_levels", mode="after")
    @classmethod
    def alpha_levels_are_unique(cls, value: tuple[AlphaLevel, ...]) -> tuple[AlphaLevel, ...]:
        if not value or len(set(value)) != len(value):
            raise ValueError("Protocol alpha levels must be unique and non-empty")
        return value

    @field_validator("local_training_row_counts", mode="after")
    @classmethod
    def row_counts_increase(
        cls, value: tuple[LocalTrainingRowCount, ...]
    ) -> tuple[LocalTrainingRowCount, ...]:
        if not value or tuple(sorted(set(value))) != value:
            raise ValueError("Protocol training row counts must be unique and increasing")
        return value


class ResolvedConfiguration(FrozenStrictModel, frozen=True):
    scientific: ScientificSettings
    runtime: RuntimeSettings
    splitting: SplitSettings
    reporting: ReportingSettings
    protocol_matrix: ProtocolMatrixSettings
    scientific_digest: ScientificDigest
    reporting_digest: ScientificDigest


class _ConfigurationInput(FrozenStrictModel, frozen=True):
    scientific: ScientificSettings
    runtime: RuntimeSettings
    splitting: SplitSettings
    reporting: ReportingSettings
    protocol_matrix: ProtocolMatrixSettings

    @model_validator(mode="after")
    def development_role_has_one_training_size(self) -> "_ConfigurationInput":
        if (
            self.scientific.evaluation_role is EvaluationRole.DEVELOPMENT
            and len(self.protocol_matrix.local_training_row_counts) > 1
        ):
            raise ValueError(
                "The DEVELOPMENT evaluation role depends on the local training size, so the "
                "protocol matrix may list only one local training row count"
            )
        return self


class _ScientificDigestMaterial(FrozenStrictModel, frozen=True):
    scientific: ScientificSettings
    splitting: SplitSettings


def _scientific_digest(
    scientific: ScientificSettings, splitting: SplitSettings
) -> ScientificDigest:
    return payload_digest(
        canonical_json(_ScientificDigestMaterial(scientific=scientific, splitting=splitting))
    )


def _first_cause(error: Exception) -> DisplayText:
    if isinstance(error, ValidationError):
        location, message = f"{error}".splitlines()[1:3]
        return DisplayText(f"{location}: {message.strip()}")
    return DisplayText(" ".join(f"{error}".split()))


def load_configuration(configuration_path: Path, repository_root: Path) -> ResolvedConfiguration:
    try:
        decoded = yaml.safe_load(configuration_path.read_text(encoding=TextEncoding.UTF8))
        supplied = _ConfigurationInput.model_validate_json(json.dumps(decoded), strict=True)
    except (OSError, yaml.YAMLError, ValidationError, TypeError) as error:
        raise ConfigurationError(
            f"Configuration is invalid: {configuration_path}: {_first_cause(error)}"
        ) from error
    if not supplied.scientific.study_arms:
        raise ConfigurationError(
            f"Configuration must name at least one study arm: {configuration_path}"
        )
    resolved_runtime = supplied.runtime.rooted_at(repository_root)
    digest = _scientific_digest(supplied.scientific, supplied.splitting)
    reporting_digest = payload_digest(canonical_json(supplied.reporting))
    return ResolvedConfiguration(
        scientific=supplied.scientific,
        runtime=resolved_runtime,
        splitting=supplied.splitting,
        reporting=supplied.reporting,
        protocol_matrix=supplied.protocol_matrix,
        scientific_digest=digest,
        reporting_digest=reporting_digest,
    )


def configuration_for_protocol_variant(
    configuration: ResolvedConfiguration,
    alpha: AlphaLevel,
    local_training_rows: LocalTrainingRowCount,
    target_clients: tuple[ClientName, ...] = (),
) -> ResolvedConfiguration:
    if alpha not in configuration.protocol_matrix.alpha_levels:
        raise ConfigurationError("Alpha level is outside the configured protocol matrix")
    if local_training_rows not in configuration.protocol_matrix.local_training_row_counts:
        raise ConfigurationError("Local training size is outside the configured protocol matrix")
    local_settings = configuration.scientific.local_detectors
    if local_training_rows <= max(
        local_settings.lof_neighbors,
        local_settings.nearest_neighbors,
    ):
        raise ConfigurationError("Local training size is too small for configured detectors")
    scientific = configuration.scientific.for_variant(
        alpha, local_settings.with_training_rows(local_training_rows), target_clients
    )
    return ResolvedConfiguration(
        scientific=scientific,
        runtime=configuration.runtime,
        splitting=configuration.splitting,
        reporting=configuration.reporting,
        protocol_matrix=configuration.protocol_matrix,
        scientific_digest=_scientific_digest(scientific, configuration.splitting),
        reporting_digest=configuration.reporting_digest,
    )
