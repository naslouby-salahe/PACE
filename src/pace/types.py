import hashlib
import ipaddress
import json
from array import array
from collections.abc import Callable
from dataclasses import dataclass
from datetime import timedelta
from enum import Enum, IntEnum, StrEnum
from pathlib import Path, PurePath, PurePosixPath
from typing import IO, Annotated, NewType

import numpy as np
import typer
from numpy.typing import NDArray
from pydantic import BaseModel, BeforeValidator, Field

FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]
PositiveFloat = Annotated[float, Field(gt=0.0, allow_inf_nan=False)]
AtLeastOneFloat = Annotated[float, Field(ge=1.0, allow_inf_nan=False)]
UnitFloat = Annotated[float, Field(ge=0.0, le=1.0, allow_inf_nan=False)]
OpenUnitFloat = Annotated[float, Field(gt=0.0, lt=1.0, allow_inf_nan=False)]
SignedUnitFloat = Annotated[float, Field(ge=-1.0, le=1.0, allow_inf_nan=False)]
NonNegativeInt = Annotated[int, Field(ge=0)]
PositiveInt = Annotated[int, Field(ge=1)]
NonBlankText = Annotated[str, Field(pattern=r"\S")]
Sha256Hex = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]

ReserveFraction = OpenUnitFloat
FalseAlarmBudget = OpenUnitFloat
SplitFraction = OpenUnitFloat
RetentionFraction = OpenUnitFloat
PairedDifferenceSd = PositiveFloat
SignificanceLevel = OpenUnitFloat
StatisticalPower = OpenUnitFloat
MinimumMeaningfulEffect = PositiveFloat
ProvenanceActive = Annotated[bool, Field(strict=True)]
CriterionSatisfied = Annotated[bool, Field(strict=True)]
PCAExplainedVariance = OpenUnitFloat
AnchorShare = OpenUnitFloat
CountSmoothing = PositiveFloat
ConfidenceLevel = OpenUnitFloat
QuantileLevel = UnitFloat
FixtureShift = PositiveFloat
FalseAlertRate = UnitFloat
DetectionRate = UnitFloat
RateDifference = SignedUnitFloat

JointScaleCap = AtLeastOneFloat
JointMultiplier = AtLeastOneFloat
BudgetMultiplier = PositiveFloat
Threshold = FiniteFloat
LearningRate = PositiveFloat
CovarianceRegularization = PositiveFloat

ObservationCount = NonNegativeInt
SourceFileCount = NonNegativeInt
ChunkSize = PositiveInt
QueueCapacity = PositiveInt
SourceByteCount = NonNegativeInt
FeatureCount = PositiveInt
CalibrationBlockSize = PositiveInt
LayerWidth = PositiveInt
TrainingEpochCount = PositiveInt
TrainingBatchSize = PositiveInt
DetectorCount = PositiveInt
DetectorRow = NonNegativeInt
PairCount = PositiveInt
QuantileBinCount = Annotated[int, Field(ge=2)]
DetectorSampleLimit = PositiveInt
LocalTrainingRowCount = PositiveInt
MinimumCalibrationRows = PositiveInt
MinimumAttackRows = Annotated[int, Field(ge=2)]
AttackRowsPerHalfCap = PositiveInt
MinimumPeerCount = Annotated[int, Field(ge=4)]
BootstrapReplicateCount = Annotated[int, Field(ge=1000)]
ScaleSearchIterations = PositiveInt

LearningSeed = NonNegativeInt
BootstrapSeed = NonNegativeInt
ArtifactSchemaVersion = PositiveInt
ExpertId = NonBlankText
ScientificDigest = Sha256Hex


def _reject_blank_location(value: object) -> object:
    if isinstance(value, str) and not value.strip():
        raise ValueError("Locations must not be blank")
    return value


Location = Annotated[Path, BeforeValidator(_reject_blank_location)]

FloatArray = NDArray[np.float64]
FloatMatrix = NDArray[np.float64]
Float32Matrix = NDArray[np.float32]
IntMatrix = NDArray[np.int64]
BooleanArray = NDArray[np.bool_]
ClassLabelArray = NDArray[np.int64]
RowIndexArray = NDArray[np.int64]
TimestampArray = NDArray[np.datetime64]
TimestampValues = NDArray[np.float64] | NDArray[np.int64]
FeatureName = NewType("FeatureName", str)
ClientName = NewType("ClientName", str)
AttackTypeName = NewType("AttackTypeName", str)
DigestComponentName = NewType("DigestComponentName", str)
RuntimePackageVersion = NewType("RuntimePackageVersion", str)
PythonVersionText = NewType("PythonVersionText", str)
PlatformName = NewType("PlatformName", str)
MachineArchitecture = NewType("MachineArchitecture", str)
ArtifactPayload = NewType("ArtifactPayload", bytes)
CacheSlot = NewType("CacheSlot", int)
LocalDetectorSeed = NewType("LocalDetectorSeed", int)
CudaDeviceIndex = NewType("CudaDeviceIndex", int)
SourceBinaryStream = IO[bytes]
UnixTimestampNanoseconds = NewType("UnixTimestampNanoseconds", int)

DisplayText = NewType("DisplayText", str)
HexColor = NewType("HexColor", str)
PValue = Annotated[float, Field(ge=0.0, le=1.0, allow_inf_nan=False)]
FigureExtent = Annotated[float, Field(gt=0.0, allow_inf_nan=False)]
PlotPosition = FiniteFloat
FeatureValue = NewType("FeatureValue", float)
EpochSeconds = NewType("EpochSeconds", float)
SourceRowNumber = NewType("SourceRowNumber", int)
SourceInteger = NewType("SourceInteger", int)
ScenarioNumber = NewType("ScenarioNumber", int)
MonthNumber = NewType("MonthNumber", int)
SourceLine = NewType("SourceLine", bytes)
SourceCell = NewType("SourceCell", str)
SourceColumn = NewType("SourceColumn", str)
ColumnIndex = NewType("ColumnIndex", int)
IPv4AddressText = NewType("IPv4AddressText", str)
LabHostNumber = NewType("LabHostNumber", int)


LabNetwork = ipaddress.IPv4Network


def lab_host_ip(network: LabNetwork, host_number: LabHostNumber) -> IPv4AddressText:
    return IPv4AddressText(network[host_number].compressed)


class FrozenStrictModel(BaseModel, frozen=True, extra="forbid", strict=True):
    pass


class ArtifactField(StrEnum):
    CONTENT_DIGEST = "content_digest"


class ArtifactSerializationMode(Enum):
    INCLUDE_DIGEST = "INCLUDE_DIGEST"
    OMIT_DIGEST = "OMIT_DIGEST"


class TextEncoding(StrEnum):
    ASCII = "ascii"
    UTF8 = "utf-8"
    UTF8_SIG = "utf-8-sig"


class SplitRole(StrEnum):
    TRAINING = "training"
    REFERENCE = "reference"
    CALIBRATION = "calibration"
    TEST = "test"
    PEER_DEVELOPMENT = "peer_development"
    HELD_OUT = "held_out"
    EXCLUDED = "excluded"


class CacheFileName(StrEnum):
    MANIFEST = "manifest.json"
    BENIGN_FEATURES = "benign"
    SOURCE_ORDER = "order"
    CAPTURE_FEATURES = "features"
    STAGING = ".staging-"


class ByteBlockSize(IntEnum):
    DIGEST = 1_048_576


def canonical_json(
    model: BaseModel,
    mode: ArtifactSerializationMode = ArtifactSerializationMode.INCLUDE_DIGEST,
) -> ArtifactPayload:
    omitted = mode is ArtifactSerializationMode.OMIT_DIGEST
    return ArtifactPayload(
        json.dumps(
            model.model_dump(
                mode="json", exclude={ArtifactField.CONTENT_DIGEST} if omitted else None
            ),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode(TextEncoding.UTF8)
    )


def big_endian_bytes(value: ObservationCount, width: ChunkSize) -> ArtifactPayload:
    return ArtifactPayload(value.to_bytes(width, byteorder="big"))


def big_endian_value(payload: ArtifactPayload) -> ObservationCount:
    return int.from_bytes(payload, byteorder="big")


def payload_digest(payload: ArtifactPayload) -> ScientificDigest:
    return hashlib.sha256(payload).hexdigest()


def placeholder_digest() -> ScientificDigest:
    return LengthPrefixedHasher().hexdigest()


HashedLabel = ClientName | AttackTypeName | DigestComponentName | ScientificDigest | SplitRole


class LengthPrefixedHasher:
    def __init__(self) -> None:
        self._hasher = hashlib.sha256()

    def count(self, value: ObservationCount) -> None:
        self._hasher.update(big_endian_bytes(value, 8))

    def raw(self, content: ArtifactPayload) -> None:
        self._hasher.update(content)

    def framed(self, content: ArtifactPayload) -> None:
        self.count(len(content))
        self._hasher.update(content)

    def framed_label(self, label: HashedLabel) -> None:
        self.framed(ArtifactPayload(label.encode(TextEncoding.UTF8)))

    def framed_member(self, member: Enum) -> None:
        self.framed(ArtifactPayload(member.name.encode(TextEncoding.UTF8)))

    def framed_path(self, path: PurePath) -> None:
        self.framed(ArtifactPayload(path.as_posix().encode(TextEncoding.UTF8)))

    def hexdigest(self) -> ScientificDigest:
        return self._hasher.hexdigest()


class RowReservoir:
    def __init__(self, capacity: ObservationCount, feature_width: FeatureCount) -> None:
        self.source_rows = array("q")
        self.features = array("f")
        self.capacity = capacity
        self.feature_width = feature_width
        self.source_population_rows = 0
        self.sampling_population_rows = 0


class FloatRowReservoir(RowReservoir):
    def __init__(self, capacity: ObservationCount, feature_width: FeatureCount) -> None:
        super().__init__(capacity, feature_width)
        self.timestamps = array("d")


class IntRowReservoir(RowReservoir):
    def __init__(self, capacity: ObservationCount, feature_width: FeatureCount) -> None:
        super().__init__(capacity, feature_width)
        self.timestamps = array("q")


class AlphaLevel(Enum):
    ALPHA_001 = "ALPHA_001"
    ALPHA_005 = "ALPHA_005"

    @property
    def fraction(self) -> FalseAlarmBudget:
        match self:
            case AlphaLevel.ALPHA_001:
                return 0.01
            case AlphaLevel.ALPHA_005:
                return 0.05

    @property
    def frozen_budget_multipliers(self) -> tuple[BudgetMultiplier, ...]:
        match self:
            case AlphaLevel.ALPHA_001:
                return (0.55, 0.75, 1.0, 1.4)
            case AlphaLevel.ALPHA_005:
                return (0.4, 0.75, 1.0, 1.4)

    @property
    def acceptance_false_alert_increase(self) -> FalseAlertRate:
        match self:
            case AlphaLevel.ALPHA_001:
                return 0.004
            case AlphaLevel.ALPHA_005:
                return 0.002

    @property
    def acceptance_gain_without_excluded_dataset(self) -> RateDifference:
        match self:
            case AlphaLevel.ALPHA_001:
                return 0.005
            case AlphaLevel.ALPHA_005:
                return 0.0


class VerificationState(Enum):
    VERIFIED = "VERIFIED"


class LogLevel(Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class ExecutionDevice(Enum):
    CPU = "CPU"
    CUDA = "CUDA"


class LocalDetectorKind(Enum):
    AUTOENCODER = "AUTOENCODER"
    PCA = "PCA"
    ISOLATION_FOREST = "ISOLATION_FOREST"
    LOCAL_OUTLIER_FACTOR = "LOCAL_OUTLIER_FACTOR"
    K_NEAREST_NEIGHBORS = "K_NEAREST_NEIGHBORS"
    HISTOGRAM_BASED_OUTLIER_SCORE = "HISTOGRAM_BASED_OUTLIER_SCORE"
    GAUSSIAN = "GAUSSIAN"

    @property
    def row(self) -> DetectorRow:
        return tuple(LocalDetectorKind).index(self)

    @property
    def is_anchor_member(self) -> bool:
        return self is not LocalDetectorKind.AUTOENCODER


class PcaSolver(StrEnum):
    FULL = "full"


class LocalExpertFamily(Enum):
    FEATURE = "FEATURE"
    RESPONSE = "RESPONSE"
    RARITY = "RARITY"


class LocalExpertKind(Enum):
    FEATURE_ISOLATION_FOREST = "FEATURE_ISOLATION_FOREST"
    FEATURE_HISTOGRAM = "FEATURE_HISTOGRAM"
    FEATURE_GAUSSIAN = "FEATURE_GAUSSIAN"
    RESPONSE_ISOLATION_FOREST = "RESPONSE_ISOLATION_FOREST"
    RARITY_PAIR_COOCCURRENCE = "RARITY_PAIR_COOCCURRENCE"
    RARITY_ECOD_SUM = "RARITY_ECOD_SUM"

    @property
    def detector(self) -> LocalDetectorKind | None:
        match self:
            case LocalExpertKind.FEATURE_ISOLATION_FOREST:
                return LocalDetectorKind.ISOLATION_FOREST
            case LocalExpertKind.FEATURE_HISTOGRAM:
                return LocalDetectorKind.HISTOGRAM_BASED_OUTLIER_SCORE
            case LocalExpertKind.FEATURE_GAUSSIAN:
                return LocalDetectorKind.GAUSSIAN
            case _:
                return None

    @property
    def family(self) -> LocalExpertFamily:
        match self:
            case (
                LocalExpertKind.FEATURE_ISOLATION_FOREST
                | LocalExpertKind.FEATURE_HISTOGRAM
                | LocalExpertKind.FEATURE_GAUSSIAN
            ):
                return LocalExpertFamily.FEATURE
            case LocalExpertKind.RESPONSE_ISOLATION_FOREST:
                return LocalExpertFamily.RESPONSE
            case LocalExpertKind.RARITY_PAIR_COOCCURRENCE | LocalExpertKind.RARITY_ECOD_SUM:
                return LocalExpertFamily.RARITY


class ExitStatus(IntEnum):
    SUCCESS = 0
    FAILURE = 1
    USAGE_ERROR = 2


class ProgramName(StrEnum):
    PACE = "pace"


class ConfigurationFile(StrEnum):
    DEFAULT = "configs/default.yaml"


class EnvironmentVariable(StrEnum):
    CONFIGURATION = "PACE_CONFIG"


class CommandTarget(StrEnum):
    IOT_23 = "iot_23"
    TON_IOT = "ton_iot"
    GOTHAM_2025 = "gotham_2025"
    CTU_13 = "ctu_13"
    N_BAIOT = "n_baiot"
    UNSW_G1 = "unsw_g1"
    UNSW_G2 = "unsw_g2"
    ALL = "all"

    @property
    def strata(self) -> "tuple[StudyStratum, ...]":
        match self:
            case CommandTarget.IOT_23:
                return (StudyStratum.IOT_23,)
            case CommandTarget.TON_IOT:
                return (StudyStratum.TON_IOT,)
            case CommandTarget.GOTHAM_2025:
                return (StudyStratum.GOTHAM_2025,)
            case CommandTarget.CTU_13:
                return (StudyStratum.CTU_13,)
            case CommandTarget.N_BAIOT:
                return (StudyStratum.N_BAIOT,)
            case CommandTarget.UNSW_G1:
                return (StudyStratum.UNSW_IOT_ATTACK_FLOWS_G1,)
            case CommandTarget.UNSW_G2:
                return (StudyStratum.UNSW_IOT_ATTACK_FLOWS_G2,)
            case CommandTarget.ALL:
                return tuple(StudyStratum)


class StudyArm(Enum):
    NO_LOCAL_CHANNEL = "NO_LOCAL_CHANNEL"
    LOCAL_ONLY = "LOCAL_ONLY"
    RESERVE_QUARTER = "RESERVE_QUARTER"
    PEER_MEAN_FUSION = "PEER_MEAN_FUSION"
    PEER_MAX_FUSION = "PEER_MAX_FUSION"
    MEAN_ENSEMBLE_LOCAL_BRANCH = "MEAN_ENSEMBLE_LOCAL_BRANCH"
    BANK_WITHOUT_FEATURE_EXPERTS = "BANK_WITHOUT_FEATURE_EXPERTS"
    BANK_WITHOUT_RESPONSE_EXPERTS = "BANK_WITHOUT_RESPONSE_EXPERTS"
    BANK_WITHOUT_RARITY_EXPERTS = "BANK_WITHOUT_RARITY_EXPERTS"
    BANK_WITHOUT_ANCHOR = "BANK_WITHOUT_ANCHOR"
    BANK_SINGLE_UNION = "BANK_SINGLE_UNION"
    RESPONSE_ON_MARGINS = "RESPONSE_ON_MARGINS"

    @property
    def removed_expert_family(self) -> LocalExpertFamily:
        match self:
            case StudyArm.BANK_WITHOUT_FEATURE_EXPERTS:
                return LocalExpertFamily.FEATURE
            case StudyArm.BANK_WITHOUT_RESPONSE_EXPERTS:
                return LocalExpertFamily.RESPONSE
            case StudyArm.BANK_WITHOUT_RARITY_EXPERTS:
                return LocalExpertFamily.RARITY
            case _:
                raise ValueError(f"{self.name} does not remove an expert family")

    @property
    def display_label(self) -> "DisplayText":
        if self is StudyArm.MEAN_ENSEMBLE_LOCAL_BRANCH:
            return ReportedMethod.MEAN_ENSEMBLE.display_label
        return DisplayText(self.name.replace("_", " ").capitalize())

    @property
    def reserve_variant(self) -> ReserveFraction:
        if self is StudyArm.RESERVE_QUARTER:
            return 0.25
        raise ValueError(f"{self.name} does not vary the local reserve")


class ImplementationScope(Enum):
    CAMPAIGN = "CAMPAIGN"
    SYNTHETIC = "SYNTHETIC"
    REPORT = "REPORT"

    @property
    def excluded_sources(self) -> tuple[PurePosixPath, ...]:
        presentation = (
            PurePosixPath("cli.py"),
            PurePosixPath("__main__.py"),
            PurePosixPath("experiment/status.py"),
            PurePosixPath("experiment/logging_setup.py"),
        )
        match self:
            case ImplementationScope.CAMPAIGN:
                return (
                    *presentation,
                    PurePosixPath("reporting"),
                    PurePosixPath("experiment/smoke.py"),
                )
            case ImplementationScope.SYNTHETIC:
                return (*presentation, PurePosixPath("reporting"))
            case ImplementationScope.REPORT:
                return (*presentation, PurePosixPath("experiment/smoke.py"))


class RuntimePackage(StrEnum):
    NUMPY = "numpy"
    SCIPY = "scipy"
    SCIKIT_LEARN = "scikit-learn"
    TORCH = "torch"
    PYDANTIC = "pydantic"
    PYYAML = "PyYAML"
    STRUCTLOG = "structlog"


class MissingRuntimePackage(StrEnum):
    UNAVAILABLE = "unavailable"


class DigestNamespace(StrEnum):
    SOURCE = "source"
    ENVIRONMENT = "environment"


class EnvironmentFile(StrEnum):
    PYPROJECT = "pyproject.toml"
    LOCK = "uv.lock"
    RUNTIME = "runtime.json"


class OutputDirectory(StrEnum):
    SMOKE = "smoke"
    CAMPAIGNS = "campaigns"
    LOGS = "logs"
    CACHE = "cache"
    RESULTS = "results"
    CLIENTS = "clients"
    PREFLIGHT = "preflight"
    REPORTS = "reports"
    FIGURES = "figures"
    TABLES = "tables"


class ArtifactFileName(StrEnum):
    DECISION = "decision.json"
    CAMPAIGN_RESULTS = "results.json"
    REPORT = "report.json"
    BUDGET_PLAN = "budget-plan.json"


class ArtifactKind(Enum):
    SYNTHETIC_DECISION = "SYNTHETIC_DECISION"
    DATASET_CAMPAIGN = "DATASET_CAMPAIGN"


class SeedDomain(Enum):
    PEER_MODEL = "PEER_MODEL"
    LOCAL_MODEL = "LOCAL_MODEL"


class ArtifactReuseState(Enum):
    CREATED = "CREATED"
    REUSED = "REUSED"


class CampaignCompletionState(Enum):
    COMPLETE = "COMPLETE"
    MISSING = "MISSING"
    INVALID = "INVALID"
    STALE = "STALE"


class ReplacementPolicy(Enum):
    PRESERVE_VALID = "PRESERVE_VALID"
    REPLACE_REQUESTED = "REPLACE_REQUESTED"


class LogEvent(StrEnum):
    FIGURE_DRAWN = "figure.drawn"
    TABLE_WRITTEN = "table.written"
    FIGURE_WRITTEN = "figure.written"
    RESULTS_WRITTEN = "results.written"
    TABLES_WRITTEN = "tables.written"
    FIGURES_WRITTEN = "figures.written"
    COMMAND_STARTED = "command.started"
    COMMAND_COMPLETED = "command.completed"
    COMMAND_FAILED = "command.failed"
    CONFIGURATION_LOADED = "configuration.loaded"
    RUN_STARTED = "run.started"
    RUN_REUSED = "run.reused"
    LOGGING_CONFIGURED = "logging.configured"
    ARTIFACT_VERIFIED = "artifact.verified"
    ARTIFACT_PERSISTED = "artifact.persisted"
    RUN_FAILED = "run.failed"
    CALIBRATION_COMPLETED = "calibration.completed"
    ARTIFACT_WRITTEN = "artifact.write.completed"
    CAMPAIGN_STARTED = "campaign.started"
    CAMPAIGN_REUSED = "campaign.reused"
    CAMPAIGN_COMPLETED = "campaign.completed"
    CAMPAIGN_FAILED = "campaign.failed"
    CAMPAIGN_STATUS = "campaign.status"
    CAMPAIGN_PREPARED = "campaign.prepared"
    CAMPAIGN_SEED_COMPLETED = "campaign.seed.completed"
    DOCTOR_STARTED = "doctor.started"
    DOCTOR_COMPLETED = "doctor.completed"
    DATASET_INSPECTED = "dataset.inspected"
    DATASET_LOADED = "dataset.loaded"
    REPORT_STARTED = "report.started"
    REPORT_COMPLETED = "report.completed"
    REPORT_REUSED = "report.reused"
    MATRIX_VARIANT_STARTED = "matrix.variant.started"
    MATRIX_VARIANT_COMPLETED = "matrix.variant.completed"
    PREFLIGHT_COMPLETED = "preflight.completed"
    BUDGET_PLAN_PERSISTED = "budget.plan.persisted"
    CLIENT_CACHE_HIT = "client.cache.hit"
    CLIENT_CACHE_MISS = "client.cache.miss"
    CLIENT_CACHE_STORED = "client.cache.stored"
    CLIENT_CACHE_REJECTED = "client.cache.rejected"
    CLIENT_CACHE_PRUNED = "client.cache.pruned"
    ADAPTER_SOURCE_READ = "adapter.source.read"
    ADAPTER_CLIENT_BUILT = "adapter.client.built"
    PEER_EXPERTS_TRAINED = "peer.experts.trained"
    LOCAL_ENSEMBLE_FITTED = "local.ensemble.fitted"


class ArtifactCommitState(StrEnum):
    UNKNOWN = "unknown"


class CommandName(StrEnum):
    DOCTOR = "doctor"
    SMOKE = "smoke"
    STATUS = "status"
    RUN = "run"
    REPORT = "report"
    PREFLIGHT = "preflight"
    ACCEPT = "accept"

    @property
    def log_file_name(self) -> PurePosixPath:
        return PurePosixPath(f"{self}.jsonl")


class EvaluationRule(Enum):
    CATASTROPHIC_DROP = "CATASTROPHIC_DROP"
    HARMED_DROP = "HARMED_DROP"
    FALSE_ALERT_EXCEEDANCE = "FALSE_ALERT_EXCEEDANCE"

    @property
    def threshold(self) -> Threshold:
        match self:
            case EvaluationRule.CATASTROPHIC_DROP:
                return -0.05
            case EvaluationRule.HARMED_DROP:
                return -0.02
            case EvaluationRule.FALSE_ALERT_EXCEEDANCE:
                return 1.5


class ColumnHeading(StrEnum):
    FROM_ROWS = "from_rows"
    TO_ROWS = "to_rows"
    OUTCOME = "outcome"
    ANALYSIS = "analysis"
    ALPHA = "alpha"
    LOCAL_ROWS = "local_rows"
    TARGETS = "targets"
    NOMINAL_GAIN = "nominal_gain"
    NOMINAL_LOWER = "nominal_lower"
    NOMINAL_UPPER = "nominal_upper"
    MATCHED_GAIN = "matched_gain"
    MATCHED_LOWER = "matched_lower"
    MATCHED_UPPER = "matched_upper"
    GAIN_WITHOUT_EXCLUDED = "gain_without_ton_iot"
    WITHOUT_LOWER = "without_lower"
    WITHOUT_UPPER = "without_upper"
    HARMED_TARGETS = "harmed_targets"
    SEVERE_TARGETS = "severe_targets"
    WORST_TARGET = "worst_target"
    FALSE_ALERT_INCREASE = "false_alert_increase"
    ACCEPTED = "accepted"
    CRITERION = "criterion"
    MEASURE = "measure"
    VALUE = "value"
    THRESHOLD = "threshold"
    SATISFIED = "satisfied"
    STRATUM = "stratum"
    TARGET = "target"
    PACE_DETECTION = "pace_detection"
    MEAN_ENSEMBLE_DETECTION = "mean_ensemble_detection"
    GAIN = "gain"
    LOWER = "lower"
    UPPER = "upper"
    MEDIAN_GAIN = "median_gain"
    WINS = "wins"
    TIES = "ties"
    LOSSES = "losses"
    SIGN_TEST_P = "sign_test_p"
    PACE_FALSE_ALERT = "pace_false_alert"
    MEAN_ENSEMBLE_FALSE_ALERT = "mean_ensemble_false_alert"
    LOCAL_DETECTION = "local_detection"
    PEER_UNION_DETECTION = "peer_union_detection"
    MAX_FUSION_DETECTION = "max_fusion_detection"
    BONFERRONI_DETECTION = "bonferroni_detection"
    BENIGN_TEST_ROWS = "benign_test_rows"
    ARM = "arm"
    ARM_DETECTION = "arm_detection"
    ARM_FALSE_ALERT = "arm_false_alert"
    METHOD = "method"
    MEAN = "mean"
    MEDIAN = "median"
    P90 = "p90"
    WORST = "worst"
    SHARE_ABOVE_ALPHA = "share_above_alpha"
    SHARE_ABOVE_TWICE_ALPHA = "share_above_twice_alpha"
    SEED = "seed"
    SCOPE = "scope"
    LEFT_OUT = "left_out"
    TRAINING_ROWS = "training_rows"
    ATTACK_FAMILIES = "attack_families"
    PEERS = "peers"
    SETTING = "setting"
    ASSUMED_SD = "assumed_sd"
    MINIMUM_DETECTABLE_EFFECT = "minimum_detectable_effect"
    POWER_AT_MEANINGFUL_EFFECT = "power_at_meaningful_effect"
    PEERS_ONLY = "peers_only"
    LOCAL_ONLY = "local_only"
    BOTH = "both"
    NEITHER = "neither"
    PEER_BRANCH = "peer_branch"
    LOCAL_BRANCH = "local_branch"
    COMBINED = "combined"


class TableName(StrEnum):
    HEADLINE = "headline"
    ACCEPTANCE = "acceptance"
    STRATA = "strata"
    TARGETS = "targets"
    ABLATIONS = "ablations"
    SCALING = "scaling"
    FALSE_ALERT_RATES = "false-alert-rates"
    SEEDS = "seeds"
    SENSITIVITY = "sensitivity"
    DECOMPOSITION = "detection-decomposition"
    FALSE_ALERT_DECOMPOSITION = "false-alert-decomposition"
    CENSUS = "census"
    CONFIGURATION = "configuration"
    POWER = "power"
    BUDGET_EFFECTS = "budget-effects"


class FigureName(StrEnum):
    HEADLINE_GAIN = "headline-gain"
    TARGET_GAINS = "target-gains"
    STRATUM_GAINS = "stratum-gains"
    OPERATING_CURVES = "operating-curves"
    FALSE_ALERT_DISTRIBUTION = "false-alert-distribution"
    ABLATIONS = "ablations"
    DETECTION_DECOMPOSITION = "detection-decomposition"
    FAMILY_GAINS = "family-gains"
    BUDGET_SCALING = "budget-scaling"
    PEER_DROWNING = "peer-drowning"
    SEED_STABILITY = "seed-stability"
    SENSITIVITY = "sensitivity"
    POWER = "power"
    CENSUS = "census"
    RESOLUTION_MECHANISM = "resolution-mechanism"
    GAIN_AGAINST_PEER_COVERAGE = "gain-against-peer-coverage"
    FALSE_ALERT_DECOMPOSITION = "false-alert-decomposition"

    @property
    def is_stratum_level(self) -> bool:
        return self in (
            FigureName.TARGET_GAINS,
            FigureName.OPERATING_CURVES,
            FigureName.FALSE_ALERT_DISTRIBUTION,
            FigureName.ABLATIONS,
            FigureName.DETECTION_DECOMPOSITION,
            FigureName.FAMILY_GAINS,
            FigureName.PEER_DROWNING,
            FigureName.GAIN_AGAINST_PEER_COVERAGE,
            FigureName.FALSE_ALERT_DECOMPOSITION,
        )


class ImageFormat(StrEnum):
    PDF = "pdf"
    PNG = "png"


class TableFormat(StrEnum):
    CSV = "csv"
    LATEX = "tex"


class ReportedMethod(Enum):
    PACE = "PACE"
    MEAN_ENSEMBLE = "MEAN_ENSEMBLE"
    LOCAL_BRANCH = "LOCAL_BRANCH"
    PEER_UNION = "PEER_UNION"
    MAX_FUSION = "MAX_FUSION"
    BONFERRONI = "BONFERRONI"

    @property
    def display_label(self) -> "DisplayText":
        match self:
            case ReportedMethod.PACE:
                return DisplayText("PACE")
            case ReportedMethod.MEAN_ENSEMBLE:
                return DisplayText("PACE with a mean-ensemble local branch")
            case ReportedMethod.LOCAL_BRANCH:
                return DisplayText("PACE local branch")
            case ReportedMethod.PEER_UNION:
                return DisplayText("PACE peer union")
            case ReportedMethod.MAX_FUSION:
                return DisplayText("Max fusion")
            case ReportedMethod.BONFERRONI:
                return DisplayText("Bonferroni union")

    @property
    def color(self) -> "HexColor":
        match self:
            case ReportedMethod.PACE:
                return HexColor("#0b6e4f")
            case ReportedMethod.MEAN_ENSEMBLE:
                return HexColor("#c0392b")
            case ReportedMethod.LOCAL_BRANCH:
                return HexColor("#8e44ad")
            case ReportedMethod.PEER_UNION:
                return HexColor("#2471a3")
            case ReportedMethod.MAX_FUSION:
                return HexColor("#7f8c8d")
            case ReportedMethod.BONFERRONI:
                return HexColor("#d68910")


class AcceptanceRule(Enum):
    SEVERE_TARGET_LOSS = "SEVERE_TARGET_LOSS"
    HARMED_TARGET_LOSS = "HARMED_TARGET_LOSS"
    HARMED_TARGET_SHARE = "HARMED_TARGET_SHARE"
    EXCEEDANCE_INCREASE = "EXCEEDANCE_INCREASE"
    GAIN_LOWER_BOUND_FLOOR = "GAIN_LOWER_BOUND_FLOOR"
    EXCEEDANCE_MULTIPLE = "EXCEEDANCE_MULTIPLE"
    TIE_BAND = "TIE_BAND"

    @property
    def threshold(self) -> Threshold:
        match self:
            case AcceptanceRule.SEVERE_TARGET_LOSS:
                return 0.05
            case AcceptanceRule.HARMED_TARGET_LOSS:
                return 0.01
            case AcceptanceRule.HARMED_TARGET_SHARE:
                return 0.03
            case AcceptanceRule.EXCEEDANCE_INCREASE:
                return 0.03
            case AcceptanceRule.GAIN_LOWER_BOUND_FLOOR:
                return -0.005
            case AcceptanceRule.EXCEEDANCE_MULTIPLE:
                return 2.0
            case AcceptanceRule.TIE_BAND:
                return 0.005


class AcceptanceDesign(Enum):
    PROTOCOL = "PROTOCOL"

    @property
    def comparator(self) -> StudyArm:
        return StudyArm.MEAN_ENSEMBLE_LOCAL_BRANCH

    @property
    def excluded_dataset(self) -> "DatasetName":
        return DatasetName.TON_IOT


class AcceptanceCriterion(Enum):
    NON_INFERIORITY = "NON_INFERIORITY"
    NOMINAL_GAIN = "NOMINAL_GAIN"
    MATCHED_GAIN = "MATCHED_GAIN"
    FALSE_ALERT_RATE = "FALSE_ALERT_RATE"
    GAIN_WITHOUT_EXCLUDED_DATASET = "GAIN_WITHOUT_EXCLUDED_DATASET"
    SEED_SIGNS = "SEED_SIGNS"
    STRATUM_ROBUSTNESS = "STRATUM_ROBUSTNESS"


class AcceptanceMeasure(Enum):
    SEVERE_TARGETS = DisplayText("severe targets")
    HARMED_TARGET_SHARE = DisplayText("harmed target share")
    NOMINAL_GAIN_LOWER_BOUND = DisplayText("nominal gain lower bound")
    MATCHED_GAIN_LOWER_BOUND = DisplayText("matched gain lower bound")
    GAIN_WITHOUT_EXCLUDED = DisplayText("gain without excluded dataset")
    LOWER_BOUND_WITHOUT_EXCLUDED = DisplayText("lower bound without excluded dataset")
    FALSE_ALERT_RATE_INCREASE = DisplayText("false-alert rate increase")
    EXCEEDANCE_SHARE_INCREASE = DisplayText("exceedance share increase")
    SMALLEST_SEED_GAIN = DisplayText("smallest seed gain")
    SMALLEST_LEAVE_ONE_STRATUM_GAIN = DisplayText("smallest leave-one-stratum-out gain")

    def __init__(self, label: DisplayText) -> None:
        self.label = label


class SatisfiedAnswer(StrEnum):
    YES = "yes"
    NO = "no"


class DetectionContribution(Enum):
    PEERS_ONLY = DisplayText("peers only")
    LOCAL_ONLY = DisplayText("local only")
    BOTH = DisplayText("both")
    NEITHER = DisplayText("neither")

    def __init__(self, label: DisplayText) -> None:
        self.label = label

    @classmethod
    def stack_order(cls) -> tuple["DetectionContribution", ...]:
        return (cls.PEERS_ONLY, cls.BOTH, cls.LOCAL_ONLY, cls.NEITHER)


class LeaveOneOutScope(Enum):
    TARGET = (DisplayText("target"), DisplayText("leave one target out"), 0.0)
    STRATUM = (DisplayText("stratum"), DisplayText("leave one stratum out"), 1.0)

    def __init__(self, column: DisplayText, axis: DisplayText, position: PlotPosition) -> None:
        self.column_label = column
        self.axis_label = axis
        self.position = position

    @property
    def ink(self) -> "PlotInk":
        match self:
            case LeaveOneOutScope.TARGET:
                return PlotInk.CYCLE_0
            case LeaveOneOutScope.STRATUM:
                return PlotInk.CYCLE_1

    @property
    def marker_area(self) -> FigureExtent:
        match self:
            case LeaveOneOutScope.TARGET:
                return 10.0
            case LeaveOneOutScope.STRATUM:
                return 30.0

    @property
    def opacity(self) -> UnitFloat:
        match self:
            case LeaveOneOutScope.TARGET:
                return 0.6
            case LeaveOneOutScope.STRATUM:
                return 1.0


class ConfigurationSection(StrEnum):
    SCIENTIFIC = "scientific"
    SPLITTING = "splitting"
    REPORTING = "reporting"
    PROTOCOL_MATRIX = "protocol_matrix"


class SummaryQuantile(Enum):
    LOWER_DECILE = 0.1
    UPPER_DECILE = 0.9

    def __init__(self, level: QuantileLevel) -> None:
        self.level = level


class FalseAlertGuide(Enum):
    NOMINAL = "nominal"
    TWICE = "twice"

    @property
    def factor(self) -> Threshold:
        match self:
            case FalseAlertGuide.NOMINAL:
                return 1.0
            case FalseAlertGuide.TWICE:
                return AcceptanceRule.EXCEEDANCE_MULTIPLE.threshold


class SeedPanel(Enum):
    ALL_TARGETS = DisplayText("all targets")
    WITHOUT_EXCLUDED_DATASET = DisplayText("without TON_IoT")

    def __init__(self, label: DisplayText) -> None:
        self.label = label

    @property
    def ink(self) -> "PlotInk":
        match self:
            case SeedPanel.ALL_TARGETS:
                return PlotInk.CYCLE_0
            case SeedPanel.WITHOUT_EXCLUDED_DATASET:
                return PlotInk.CYCLE_1

    @property
    def excluded_dataset(self) -> "DatasetName | None":
        if self is SeedPanel.ALL_TARGETS:
            return None
        return AcceptanceDesign.PROTOCOL.excluded_dataset


class PlotInk(StrEnum):
    BLACK = "black"
    CYCLE_0 = "C0"
    CYCLE_1 = "C1"
    CYCLE_2 = "C2"
    CYCLE_3 = "C3"


class PlotLineStyle(StrEnum):
    DASHED = "--"
    DOTTED = ":"


class PlotMarker(StrEnum):
    CIRCLE = "o"
    CROSS = "x"


class PlotStepWhere(StrEnum):
    POST = "post"


class PlotLegendAnchor(StrEnum):
    OUTSIDE_RIGHT = "outside right center"


class FigureLayout(StrEnum):
    CONSTRAINED = "constrained"


class FigureResolution(IntEnum):
    PNG = 200


class PdfMetadataField(StrEnum):
    CREATOR = "Creator"
    PRODUCER = "Producer"
    CREATION_DATE = "CreationDate"
    SOFTWARE = "Software"


class LatexAlignment(StrEnum):
    LEFT = "l"


class CsvNewline(StrEnum):
    UNIVERSAL = ""


class BudgetGuide(Enum):
    HALF_LOCAL = (DisplayText("half local budget"), 2.0)
    LOCAL = (DisplayText("local budget"), 1.0)

    def __init__(self, label: DisplayText, divisor: Threshold) -> None:
        self.label = label
        self.divisor = divisor

    def budget(self, alpha: AlphaLevel, reserve: ReserveFraction) -> FalseAlarmBudget:
        return alpha.fraction * reserve / self.divisor

    def text(self, alpha: AlphaLevel) -> DisplayText:
        return DisplayText(f"{self.label}, \u03b1 = {alpha.fraction}")


class ReferenceAnnotation(Enum):
    PACE_SWEEP = DisplayText("PACE sweep")
    ONE_ALERT_PER_BLOCK = DisplayText("one alert per block")

    def __init__(self, label: DisplayText) -> None:
        self.label = label


class GainSeries(Enum):
    NOMINAL = DisplayText("nominal gain")
    MATCHED_FALSE_ALERT = DisplayText("gain at the mean-ensemble variant's false-alert rate")
    WITHOUT_EXCLUDED = DisplayText("nominal gain without TON_IoT")

    def __init__(self, label: DisplayText) -> None:
        self.label = label

    @property
    def ink(self) -> PlotInk:
        match self:
            case GainSeries.NOMINAL:
                return PlotInk.CYCLE_0
            case GainSeries.MATCHED_FALSE_ALERT:
                return PlotInk.CYCLE_1
            case GainSeries.WITHOUT_EXCLUDED:
                return PlotInk.CYCLE_2

    def point(
        self, cell: "AcceptanceCell"
    ) -> tuple[RateDifference | None, "RateConfidenceInterval | None"]:
        match self:
            case GainSeries.NOMINAL:
                return cell.nominal_gain, cell.nominal_interval
            case GainSeries.MATCHED_FALSE_ALERT:
                return cell.matched_gain, cell.matched_interval
            case GainSeries.WITHOUT_EXCLUDED:
                return cell.gain_without_excluded, cell.gain_without_excluded_interval


class UnavailableReading(StrEnum):
    NOT_AVAILABLE = "n/a"


class SourceInventoryStatus(StrEnum):
    READY = "Dataset sources ready"
    INCOMPLETE = "Dataset sources incomplete"


class AcceptanceOutcomeWord(StrEnum):
    GATING = "gating"
    ACCEPTED = "accepted"
    FAILED = "failed"


class DatasetName(Enum):
    IOT_23 = "IOT_23"
    TON_IOT = "TON_IOT"
    GOTHAM_2025 = "GOTHAM_2025"
    CTU_13 = "CTU_13"
    N_BAIOT = "N_BAIOT"
    UNSW_IOT_ATTACK_FLOWS = "UNSW_IOT_ATTACK_FLOWS"

    @property
    def display_label(self) -> "DisplayText":
        return DisplayText("N-BaIoT" if self is DatasetName.N_BAIOT else self.name)

    @property
    def source_location(self) -> PurePosixPath:
        match self:
            case DatasetName.IOT_23:
                return PurePosixPath(CorpusPath.IOT_23_ARCHIVE)
            case DatasetName.TON_IOT:
                return PurePosixPath(CorpusPath.TON_IOT_PROCESSED)
            case DatasetName.GOTHAM_2025:
                return PurePosixPath(CorpusPath.GOTHAM_PROCESSED)
            case DatasetName.CTU_13:
                return PurePosixPath(CorpusPath.CTU_13_ARCHIVE)
            case DatasetName.N_BAIOT:
                return PurePosixPath(CorpusPath.N_BAIOT)
            case DatasetName.UNSW_IOT_ATTACK_FLOWS:
                return PurePosixPath(CorpusPath.UNSW)

    @property
    def location_kind(self) -> "DatasetLocationKind":
        match self:
            case DatasetName.IOT_23 | DatasetName.CTU_13:
                return DatasetLocationKind.FILE
            case _:
                return DatasetLocationKind.DIRECTORY


class StudyStratum(Enum):
    IOT_23 = "IOT_23"
    TON_IOT = "TON_IOT"
    GOTHAM_2025 = "GOTHAM_2025"
    CTU_13 = "CTU_13"
    N_BAIOT = "N_BAIOT"
    UNSW_IOT_ATTACK_FLOWS_G1 = "UNSW_IOT_ATTACK_FLOWS_G1"
    UNSW_IOT_ATTACK_FLOWS_G2 = "UNSW_IOT_ATTACK_FLOWS_G2"

    @property
    def color(self) -> HexColor:
        match self:
            case StudyStratum.IOT_23:
                return HexColor("#1b9e77")
            case StudyStratum.TON_IOT:
                return HexColor("#d95f02")
            case StudyStratum.GOTHAM_2025:
                return HexColor("#7570b3")
            case StudyStratum.CTU_13:
                return HexColor("#e7298a")
            case StudyStratum.N_BAIOT:
                return HexColor("#66a61e")
            case StudyStratum.UNSW_IOT_ATTACK_FLOWS_G1:
                return HexColor("#e6ab02")
            case StudyStratum.UNSW_IOT_ATTACK_FLOWS_G2:
                return HexColor("#a6761d")

    @property
    def display_label(self) -> DisplayText:
        if self is StudyStratum.N_BAIOT:
            return self.dataset.display_label
        return DisplayText(self.name)

    @property
    def dataset(self) -> DatasetName:
        match self:
            case StudyStratum.IOT_23:
                return DatasetName.IOT_23
            case StudyStratum.TON_IOT:
                return DatasetName.TON_IOT
            case StudyStratum.GOTHAM_2025:
                return DatasetName.GOTHAM_2025
            case StudyStratum.CTU_13:
                return DatasetName.CTU_13
            case StudyStratum.N_BAIOT:
                return DatasetName.N_BAIOT
            case StudyStratum.UNSW_IOT_ATTACK_FLOWS_G1 | StudyStratum.UNSW_IOT_ATTACK_FLOWS_G2:
                return DatasetName.UNSW_IOT_ATTACK_FLOWS


def default_study_stratum(dataset: DatasetName) -> StudyStratum:
    match dataset:
        case DatasetName.IOT_23:
            return StudyStratum.IOT_23
        case DatasetName.TON_IOT:
            return StudyStratum.TON_IOT
        case DatasetName.GOTHAM_2025:
            return StudyStratum.GOTHAM_2025
        case DatasetName.CTU_13:
            return StudyStratum.CTU_13
        case DatasetName.N_BAIOT:
            return StudyStratum.N_BAIOT
        case DatasetName.UNSW_IOT_ATTACK_FLOWS:
            raise DatasetValidationError(
                "UNSW IoT Attack Flows has two study strata; select G1 or G2"
            )


class FeatureSchemaId(Enum):
    IOT_23_ZEEK_ENGINEERED_V1 = "IOT_23_ZEEK_ENGINEERED_V1"
    TON_IOT_ZEEK_ENGINEERED_V1 = "TON_IOT_ZEEK_ENGINEERED_V1"
    GOTHAM_TSHARK_ENGINEERED_V1 = "GOTHAM_TSHARK_ENGINEERED_V1"
    CTU_13_ARGUS_ENGINEERED_V1 = "CTU_13_ARGUS_ENGINEERED_V1"
    N_BAIOT_KITSUNE_SIGNED_LOG_V1 = "N_BAIOT_KITSUNE_SIGNED_LOG_V1"
    UNSW_MUD_G1_MINUTE_COUNTERS = "UNSW_MUD_G1_MINUTE_COUNTERS"
    UNSW_MUD_G2_MINUTE_COUNTERS = "UNSW_MUD_G2_MINUTE_COUNTERS"

    @property
    def dataset(self) -> DatasetName:
        match self:
            case FeatureSchemaId.IOT_23_ZEEK_ENGINEERED_V1:
                return DatasetName.IOT_23
            case FeatureSchemaId.TON_IOT_ZEEK_ENGINEERED_V1:
                return DatasetName.TON_IOT
            case FeatureSchemaId.GOTHAM_TSHARK_ENGINEERED_V1:
                return DatasetName.GOTHAM_2025
            case FeatureSchemaId.CTU_13_ARGUS_ENGINEERED_V1:
                return DatasetName.CTU_13
            case FeatureSchemaId.N_BAIOT_KITSUNE_SIGNED_LOG_V1:
                return DatasetName.N_BAIOT
            case FeatureSchemaId.UNSW_MUD_G1_MINUTE_COUNTERS:
                return DatasetName.UNSW_IOT_ATTACK_FLOWS
            case FeatureSchemaId.UNSW_MUD_G2_MINUTE_COUNTERS:
                return DatasetName.UNSW_IOT_ATTACK_FLOWS


class DatasetAvailability(Enum):
    AVAILABLE = "AVAILABLE"
    MISSING = "MISSING"
    EMPTY = "EMPTY"
    INVALID_LAYOUT = "INVALID_LAYOUT"
    UNREADABLE = "UNREADABLE"


class DatasetLocationKind(Enum):
    FILE = "FILE"
    DIRECTORY = "DIRECTORY"


class AlertDecision(Enum):
    ALERT = "ALERT"
    PASS = "PASS"


class FixtureKind(Enum):
    DETERMINISTIC_SYNTHETIC = "DETERMINISTIC_SYNTHETIC"


class TargetEligibilityIssue(Enum):
    NO_ATTACK_TYPES = "NO_ATTACK_TYPES"
    INSUFFICIENT_CALIBRATION_ROWS = "INSUFFICIENT_CALIBRATION_ROWS"
    INSUFFICIENT_PEERS = "INSUFFICIENT_PEERS"


class TargetCohortRole(Enum):
    REPORTED_TARGET = "REPORTED_TARGET"
    PEER_ONLY = "PEER_ONLY"
    OUTSIDE_PEER_POOL = "OUTSIDE_PEER_POOL"
    ELIGIBILITY_DRIVEN = "ELIGIBILITY_DRIVEN"


class PaceError(Exception):
    pass


class ConfigurationError(PaceError):
    pass


class ArtifactValidationError(PaceError):
    pass


class ArtifactConflictError(PaceError):
    pass


class LocalDetectorError(PaceError):
    pass


class ExecutionDeviceUnavailableError(LocalDetectorError):
    pass


class DetectorScoreMatrix:
    __slots__ = ("_values",)

    def __init__(self, values: FloatMatrix) -> None:
        if (
            values.ndim != 2
            or values.shape[0] != len(LocalDetectorKind)
            or values.shape[1] == 0
            or not np.isfinite(values).all()
        ):
            raise ValueError("Detector scores must be a finite matrix with one row per detector")
        scores = np.array(values, dtype=np.float64, copy=True)
        scores.setflags(write=False)
        self._values = scores

    @property
    def values(self) -> FloatMatrix:
        return self._values


@dataclass(frozen=True, slots=True, kw_only=True)
class BudgetSweepPerformance:
    multiplier: BudgetMultiplier
    detection_rate: DetectionRate
    false_alert_rate: FalseAlertRate


@dataclass(frozen=True, slots=True, kw_only=True)
class RateConfidenceInterval:
    lower_bound: RateDifference
    upper_bound: RateDifference
    confidence_level: ConfidenceLevel

    def __post_init__(self) -> None:
        if self.lower_bound > self.upper_bound:
            raise ValueError("Confidence interval lower bound exceeds its upper bound")


@dataclass(frozen=True, slots=True, kw_only=True)
class DatasetSourceInventory:
    dataset: DatasetName
    location: Path
    availability: DatasetAvailability
    file_count: SourceFileCount
    byte_count: SourceByteCount

    def __post_init__(self) -> None:
        if self.availability is DatasetAvailability.AVAILABLE and self.file_count == 0:
            raise ValueError("Available dataset sources must contain at least one file")
        if self.availability is not DatasetAvailability.AVAILABLE and (
            self.file_count != 0 or self.byte_count != 0
        ):
            raise ValueError("Unavailable dataset sources cannot report readable file contents")

    @property
    def is_available(self) -> bool:
        return self.availability is DatasetAvailability.AVAILABLE


class DatasetValidationError(PaceError):
    pass


@dataclass(frozen=True, slots=True)
class DatasetClientId:
    dataset: DatasetName
    name: ClientName

    def __post_init__(self) -> None:
        if not self.name.strip() or Path(self.name).name != self.name:
            raise ValueError("Dataset client identifiers must be non-empty path names")


def _validate_feature_matrix(features: FloatMatrix) -> None:
    if (
        features.ndim != 2
        or features.shape[0] == 0
        or features.shape[1] == 0
        or not np.isfinite(features).all()
    ):
        raise ValueError("Feature matrices must be non-empty and finite")


def _validate_source_order(source_order: RowIndexArray, expected_rows: ObservationCount) -> None:
    if (
        source_order.ndim != 1
        or source_order.size != expected_rows
        or not np.issubdtype(source_order.dtype, np.integer)
        or np.any(source_order < 0)
        or (source_order.size > 1 and np.any(np.diff(source_order) <= 0))
    ):
        raise ValueError("Source row order must be non-negative and strictly increasing")


@dataclass(frozen=True, slots=True)
class AttackType:
    identifier: AttackTypeName

    def __post_init__(self) -> None:
        if not self.identifier or self.identifier != self.identifier.strip():
            raise ValueError("Attack type identifiers must be non-empty and trimmed")


class NBaIoTAttackFamily(StrEnum):
    GAFGYT = "gafgyt_attacks"
    MIRAI = "mirai_attacks"


class NBaIoTSourceFile(StrEnum):
    BENIGN = "benign_traffic.csv"
    COMBO = "combo.csv"
    JUNK = "junk.csv"
    SCAN = "scan.csv"
    TCP = "tcp.csv"
    UDP = "udp.csv"
    ACK = "ack.csv"
    SYN = "syn.csv"
    UDPPLAIN = "udpplain.csv"


class NBaIoTLayout(IntEnum):
    FEATURE_COUNT = 115


class NBaIoTAttackVariation(Enum):
    GAFGYT_COMBO = (
        NBaIoTAttackFamily.GAFGYT,
        NBaIoTSourceFile.COMBO,
        AttackTypeName("GAFGYT/COMBO"),
    )
    GAFGYT_JUNK = (NBaIoTAttackFamily.GAFGYT, NBaIoTSourceFile.JUNK, AttackTypeName("GAFGYT/JUNK"))
    GAFGYT_SCAN = (NBaIoTAttackFamily.GAFGYT, NBaIoTSourceFile.SCAN, AttackTypeName("GAFGYT/SCAN"))
    GAFGYT_TCP = (NBaIoTAttackFamily.GAFGYT, NBaIoTSourceFile.TCP, AttackTypeName("GAFGYT/TCP"))
    GAFGYT_UDP = (NBaIoTAttackFamily.GAFGYT, NBaIoTSourceFile.UDP, AttackTypeName("GAFGYT/UDP"))
    MIRAI_ACK = (NBaIoTAttackFamily.MIRAI, NBaIoTSourceFile.ACK, AttackTypeName("MIRAI/ACK"))
    MIRAI_SCAN = (NBaIoTAttackFamily.MIRAI, NBaIoTSourceFile.SCAN, AttackTypeName("MIRAI/SCAN"))
    MIRAI_SYN = (NBaIoTAttackFamily.MIRAI, NBaIoTSourceFile.SYN, AttackTypeName("MIRAI/SYN"))
    MIRAI_UDP = (NBaIoTAttackFamily.MIRAI, NBaIoTSourceFile.UDP, AttackTypeName("MIRAI/UDP"))
    MIRAI_UDPPLAIN = (
        NBaIoTAttackFamily.MIRAI,
        NBaIoTSourceFile.UDPPLAIN,
        AttackTypeName("MIRAI/UDPPLAIN"),
    )

    def __init__(
        self,
        family: NBaIoTAttackFamily,
        filename: NBaIoTSourceFile,
        identifier: AttackTypeName,
    ) -> None:
        self.family = family
        self.filename = filename
        self.identifier = identifier

    @property
    def attack_type(self) -> AttackType:
        return AttackType(self.identifier)


class UnswAttackFamily(StrEnum):
    ARP_SPOOF = "ArpSpoof"
    PING_OF_DEATH = "PingOfDeath"
    SMURF = "Smurf"
    SNMP_REFLECTION = "Snmp"
    SSDP_REFLECTION = "Ssdp"
    TCP_SYN_DEVICE = "TcpSynDevice"
    TCP_SYN_REFLECTION = "TcpSynReflection"
    UDP_DEVICE = "UdpDevice"

    @classmethod
    def from_annotation(cls, name: "SourceCell") -> "UnswAttackFamily":
        for family in cls:
            if name.startswith(family):
                return family
        raise DatasetValidationError(f"Unknown UNSW attack annotation: {name}")

    @property
    def attack_type(self) -> AttackType:
        match self:
            case UnswAttackFamily.ARP_SPOOF:
                return AttackType(AttackTypeName("ARP_SPOOF"))
            case UnswAttackFamily.PING_OF_DEATH:
                return AttackType(AttackTypeName("PING_OF_DEATH"))
            case UnswAttackFamily.SMURF:
                return AttackType(AttackTypeName("SMURF"))
            case UnswAttackFamily.SNMP_REFLECTION:
                return AttackType(AttackTypeName("SNMP_REFLECTION"))
            case UnswAttackFamily.SSDP_REFLECTION:
                return AttackType(AttackTypeName("SSDP_REFLECTION"))
            case UnswAttackFamily.TCP_SYN_DEVICE:
                return AttackType(AttackTypeName("TCP_SYN_DEVICE"))
            case UnswAttackFamily.TCP_SYN_REFLECTION:
                return AttackType(AttackTypeName("TCP_SYN_REFLECTION"))
            case UnswAttackFamily.UDP_DEVICE:
                return AttackType(AttackTypeName("UDP_DEVICE"))


class UnswColumn(StrEnum):
    TIMESTAMP = "Timestamp"
    FROM_DNS_PACKETS = "FromLocalUdpPort53IP192.168.1.1Packet"
    FROM_DNS_BYTES = "FromLocalUdpPort53IP192.168.1.1Byte"
    FROM_DHCP_PACKETS = "FromLocalUdpPort67IP192.168.1.1Packet"
    FROM_DHCP_BYTES = "FromLocalUdpPort67IP192.168.1.1Byte"
    FROM_ARP_PACKETS = "FromLocalArpPortAllPacket"
    FROM_ARP_BYTES = "FromLocalArpPortAllByte"
    TO_DHCP_BROADCAST_PACKETS = "ToLocalUdpPort67IP255.255.255.255/32Packet"
    TO_DHCP_BROADCAST_BYTES = "ToLocalUdpPort67IP255.255.255.255/32Byte"
    TO_DHCP_GATEWAY_PACKETS = "ToLocalUdpPort67IP192.168.1.1Packet"
    TO_DHCP_GATEWAY_BYTES = "ToLocalUdpPort67IP192.168.1.1Byte"
    TO_DNS_PACKETS = "ToLocalUdpPort53IP192.168.1.1Packet"
    TO_DNS_BYTES = "ToLocalUdpPort53IP192.168.1.1Byte"
    TO_EAPOL_PACKETS = "ToLocal0x888ePortAllPacket"
    TO_EAPOL_BYTES = "ToLocal0x888ePortAllByte"
    TO_ARP_PACKETS = "ToLocalArpPortAllPacket"
    TO_ARP_BYTES = "ToLocalArpPortAllByte"
    FLOW_COUNT = "NoOfFlows"

    @property
    def feature(self) -> FeatureName:
        match self:
            case UnswColumn.FROM_DNS_PACKETS:
                return FeatureName("from_local_dns_packets")
            case UnswColumn.FROM_DNS_BYTES:
                return FeatureName("from_local_dns_bytes")
            case UnswColumn.FROM_DHCP_PACKETS:
                return FeatureName("from_local_dhcp_packets")
            case UnswColumn.FROM_DHCP_BYTES:
                return FeatureName("from_local_dhcp_bytes")
            case UnswColumn.FROM_ARP_PACKETS:
                return FeatureName("from_local_arp_packets")
            case UnswColumn.FROM_ARP_BYTES:
                return FeatureName("from_local_arp_bytes")
            case UnswColumn.TO_DHCP_BROADCAST_PACKETS | UnswColumn.TO_DHCP_GATEWAY_PACKETS:
                return FeatureName("to_local_dhcp_packets")
            case UnswColumn.TO_DHCP_BROADCAST_BYTES | UnswColumn.TO_DHCP_GATEWAY_BYTES:
                return FeatureName("to_local_dhcp_bytes")
            case UnswColumn.TO_DNS_PACKETS:
                return FeatureName("to_local_dns_packets")
            case UnswColumn.TO_DNS_BYTES:
                return FeatureName("to_local_dns_bytes")
            case UnswColumn.TO_EAPOL_PACKETS:
                return FeatureName("to_local_eapol_packets")
            case UnswColumn.TO_EAPOL_BYTES:
                return FeatureName("to_local_eapol_bytes")
            case UnswColumn.TO_ARP_PACKETS:
                return FeatureName("to_local_arp_packets")
            case UnswColumn.TO_ARP_BYTES:
                return FeatureName("to_local_arp_bytes")
            case UnswColumn.FLOW_COUNT:
                return FeatureName("flow_count")
            case UnswColumn.TIMESTAMP:
                raise ValueError("The UNSW timestamp column is not a feature counter")

    @classmethod
    def counters(cls, group: StudyStratum) -> tuple["UnswColumn", ...]:
        shared = (
            cls.FROM_DNS_PACKETS,
            cls.FROM_DNS_BYTES,
            cls.FROM_DHCP_PACKETS,
            cls.FROM_DHCP_BYTES,
            cls.FROM_ARP_PACKETS,
            cls.FROM_ARP_BYTES,
        )
        match group:
            case StudyStratum.UNSW_IOT_ATTACK_FLOWS_G1:
                return (
                    *shared,
                    cls.TO_DHCP_BROADCAST_PACKETS,
                    cls.TO_DHCP_BROADCAST_BYTES,
                    cls.TO_DNS_PACKETS,
                    cls.TO_DNS_BYTES,
                    cls.TO_EAPOL_PACKETS,
                    cls.TO_EAPOL_BYTES,
                    cls.TO_ARP_PACKETS,
                    cls.TO_ARP_BYTES,
                    cls.FLOW_COUNT,
                )
            case StudyStratum.UNSW_IOT_ATTACK_FLOWS_G2:
                return (
                    *shared,
                    cls.TO_DHCP_GATEWAY_PACKETS,
                    cls.TO_DHCP_GATEWAY_BYTES,
                    cls.TO_DNS_PACKETS,
                    cls.TO_DNS_BYTES,
                    cls.TO_ARP_PACKETS,
                    cls.TO_ARP_BYTES,
                    cls.FLOW_COUNT,
                )
            case _:
                raise DatasetValidationError(f"Unsupported UNSW feature stratum: {group.name}")


class UnswDevice(StrEnum):
    WEMO_MOTION_SENSOR = "ec1a59832811"
    WEMO_POWER_SWITCH = "ec1a5979f489"
    SAMSUNG_CAMERA = "00166cab6b88"
    TP_LINK_PLUG = "50c7bf005639"
    NETATMO_CAMERA = "70ee50183443"
    PHILIPS_HUE_BULB = "0017882b9a25"
    AMAZON_ECHO = "44650d56ccd3"
    GOOGLE_CHROMECAST = "f4f5d88f0a3c"
    IHOME = "74c63b29d71d"
    LIFX_BULB = "d073d5018308"

    @property
    def identifier(self) -> ClientName:
        return ClientName(self)

    @property
    def group(self) -> StudyStratum:
        match self:
            case (
                UnswDevice.WEMO_MOTION_SENSOR
                | UnswDevice.WEMO_POWER_SWITCH
                | UnswDevice.SAMSUNG_CAMERA
                | UnswDevice.TP_LINK_PLUG
                | UnswDevice.NETATMO_CAMERA
            ):
                return StudyStratum.UNSW_IOT_ATTACK_FLOWS_G1
            case _:
                return StudyStratum.UNSW_IOT_ATTACK_FLOWS_G2


class UnswFileSuffix(StrEnum):
    FLOW_STATS = "_flowstats.csv"
    ANNOTATIONS = ".csv"


class TonIoTStudyHost(IntEnum):
    ROUTER = 1
    MIDDLEWARE_SERVER = 152
    SECURITY_ONION = 180
    OWASP_SHEPHERD = 184
    ORCHESTRATED_SERVER = 190
    WINDOWS_7 = 193
    METASPLOITABLE_3 = 194
    WINDOWS_10 = 195
    HOST_79 = 79

    def host_ip(self, network: LabNetwork) -> IPv4AddressText:
        return lab_host_ip(network, LabHostNumber(self))


class NetworkProtocolCategory(StrEnum):
    TCP = "tcp"
    UDP = "udp"
    ICMP = "icmp"


class ReservoirGroup(IntEnum):
    SKIPPED = -1
    BENIGN = 0
    ATTACK = 1


class MissingValueToken(StrEnum):
    EMPTY = ""
    DASH = "-"
    EMPTY_MARKER = "(empty)"

    @classmethod
    def numeric(cls) -> tuple["MissingValueToken", "MissingValueToken"]:
        return cls.EMPTY, cls.DASH

    @classmethod
    def zeek(cls) -> tuple["MissingValueToken", "MissingValueToken", "MissingValueToken"]:
        return cls.EMPTY, cls.DASH, cls.EMPTY_MARKER


class DerivedColumn(StrEnum):
    ROW = "row"
    KIND = "kind"
    STARTED = "started"
    KNOWN_DIRECTION = "known_direction"
    USABLE = "usable"
    SOURCE_ROW = "source_row"
    DETAIL = "detail"
    LABEL = "label"
    TIMESTAMP = "ts"
    SUPPORTED = "supported"
    CONSISTENT = "consistent"
    SECOND = "second"
    LENGTH = "length"
    PORT = "port"
    ADDRESS = "ip"
    SYN = "syn"
    PACKETS = "packets"
    BYTES = "bytes"
    PORTS = "ports"
    ADDRESSES = "addresses"
    NANOSECONDS = "ns"
    TERMINAL = "_terminal"
    SPLIT = "_split"


class Ctu13LabelMarker(StrEnum):
    NORMAL = "normal"
    BACKGROUND = "background"
    BOTNET = "botnet"
    COMMAND_AND_CONTROL = "c-c"


class Ctu13Column(StrEnum):
    START_TIME = "StartTime"
    DURATION = "Dur"
    PROTOCOL = "Proto"
    DIRECTION = "Dir"
    STATE = "State"
    LABEL = "Label"
    TOTAL_PACKETS = "TotPkts"
    TOTAL_BYTES = "TotBytes"
    SOURCE_BYTES = "SrcBytes"
    SOURCE_TOS = "sTos"
    DESTINATION_TOS = "dTos"
    SOURCE_PORT = "Sport"
    DESTINATION_PORT = "Dport"

    @property
    def feature(self) -> FeatureName:
        match self:
            case Ctu13Column.DURATION:
                return FeatureName("dur")
            case Ctu13Column.TOTAL_PACKETS:
                return FeatureName("totpkts")
            case Ctu13Column.TOTAL_BYTES:
                return FeatureName("totbytes")
            case Ctu13Column.SOURCE_BYTES:
                return FeatureName("srcbytes")
            case Ctu13Column.SOURCE_TOS:
                return FeatureName("stos")
            case Ctu13Column.DESTINATION_TOS:
                return FeatureName("dtos")
            case _:
                raise ValueError(f"{self.name} is not a numeric CTU-13 feature column")

    @property
    def port_features(self) -> tuple[FeatureName, FeatureName]:
        match self:
            case Ctu13Column.SOURCE_PORT:
                return FeatureName("sport_privileged"), FeatureName("sport_value")
            case Ctu13Column.DESTINATION_PORT:
                return FeatureName("dport_privileged"), FeatureName("dport_value")
            case _:
                raise ValueError(f"{self.name} is not a CTU-13 port column")

    @classmethod
    def numeric(cls) -> tuple["Ctu13Column", ...]:
        return (
            cls.DURATION,
            cls.TOTAL_PACKETS,
            cls.TOTAL_BYTES,
            cls.SOURCE_BYTES,
            cls.SOURCE_TOS,
            cls.DESTINATION_TOS,
        )

    @classmethod
    def ports(cls) -> tuple["Ctu13Column", "Ctu13Column"]:
        return cls.SOURCE_PORT, cls.DESTINATION_PORT


class CtuSourceDirectionToken(StrEnum):
    ORIGIN_TO_RESPONDER = "->"
    BIDIRECTIONAL = "<->"
    RESPONDER_TO_ORIGIN = "<-"
    UNKNOWN = "<?>"
    WHO = "who"
    UNKNOWN_REVERSED = "?>"

    @property
    def direction(self) -> "CtuFlowDirection":
        match self:
            case CtuSourceDirectionToken.ORIGIN_TO_RESPONDER:
                return CtuFlowDirection.ORIGIN_TO_RESPONDER
            case CtuSourceDirectionToken.BIDIRECTIONAL:
                return CtuFlowDirection.BIDIRECTIONAL
            case CtuSourceDirectionToken.RESPONDER_TO_ORIGIN:
                return CtuFlowDirection.RESPONDER_TO_ORIGIN
            case CtuSourceDirectionToken.WHO:
                return CtuFlowDirection.WHO
            case CtuSourceDirectionToken.UNKNOWN | CtuSourceDirectionToken.UNKNOWN_REVERSED:
                return CtuFlowDirection.UNKNOWN


class CorpusPath(StrEnum):
    CTU_13_EXTRACTED = "CTU-13/extracted"
    CTU_13 = "CTU-13"
    CTU_13_ARCHIVE = "CTU-13/CTU-13-Dataset.tar.bz2"
    CTU_13_ARCHIVE_FLAT = "CTU-13-Dataset.tar.bz2"
    GOTHAM_PROCESSED = "Gotham2025/extracted/processed"
    TON_IOT_PROCESSED = "TON-IoT/Processed_datasets/Processed_Network_dataset"
    TON_IOT_PROCESSED_DIRECT = "Processed_datasets/Processed_Network_dataset"
    IOT_23 = "IoT-23-v2"
    IOT_23_ARCHIVE = "IoT-23-v2/iot_23_datasets_small.tar.gz"
    N_BAIOT = "N-BaIoT"
    UNSW = "UNSW-IoT-Attack-Flows"
    UNSW_FLOWS = "flowdata/flowdata"
    UNSW_ANNOTATIONS = "annotations/annotations"


class CtuRootAlias(StrEnum):
    CTU_13 = "ctu-13"
    DATASET = "ctu-13-dataset"
    EXTRACTED = "extracted"


class SamplingSeed(IntEnum):
    CTU_13 = 13
    IOT_23_RESERVOIR = 7
    IOT_23_CHUNK = 11
    TON_IOT = 5
    GOTHAM_BASE = 1000


class Ctu13BotnetLabel(StrEnum):
    BOTNET = "BOTNET"
    COMMAND_AND_CONTROL = "COMMAND_AND_CONTROL"


class CtuSourceSuffix(StrEnum):
    FLOW_FILE = ".binetflow"
    CSV_FILE = ".csv"
    COMPRESSED_ARCHIVE = ".tar.bz2"


class GothamColumn(StrEnum):
    FRAME_TIME = "frame.time"
    FRAME_LENGTH = "frame.len"
    FRAME_PROTOCOLS = "frame.protocols"
    IP_TTL = "ip.ttl"
    IP_PROTOCOL = "ip.proto"
    IP_TOS = "ip.tos"
    IP_FLAGS = "ip.flags"
    IP_DESTINATION = "ip.dst"
    TCP_SOURCE_PORT = "tcp.srcport"
    TCP_DESTINATION_PORT = "tcp.dstport"
    TCP_FLAGS = "tcp.flags"
    TCP_WINDOW_SIZE = "tcp.window_size_value"
    TCP_PDU_SIZE = "tcp.pdu.size"
    UDP_SOURCE_PORT = "udp.srcport"
    UDP_DESTINATION_PORT = "udp.dstport"
    LABEL = "label"


class GothamBaseFeature(StrEnum):
    FRAME_LENGTH = "frame_length"
    TTL = "ttl"
    PROTOCOL = "protocol"
    TOS = "tos"
    IP_FLAGS = "ip_flags"
    TCP_SYN = "tcp_syn"
    TCP_ACK = "tcp_ack"
    TCP_FIN = "tcp_fin"
    TCP_RST = "tcp_rst"
    TCP_PSH = "tcp_psh"
    TCP_WINDOW = "tcp_window"
    TCP_PAYLOAD = "tcp_payload"
    IS_TCP = "is_tcp"
    IS_UDP = "is_udp"
    IS_ICMP = "is_icmp"
    IS_GRE = "is_gre"
    SOURCE_PORT_PRIVILEGED = "source_port_privileged"
    DESTINATION_PORT_PRIVILEGED = "destination_port_privileged"
    SOURCE_PORT = "source_port"
    DESTINATION_PORT = "destination_port"

    @property
    def position(self) -> ColumnIndex:
        return ColumnIndex(list(GothamBaseFeature).index(self))


class GothamAggregateFeature(StrEnum):
    PACKETS_PER_SECOND = "packets_per_second"
    BYTES_PER_SECOND = "bytes_per_second"
    DISTINCT_DESTINATION_PORTS_PER_SECOND = "distinct_destination_ports_per_second"
    DISTINCT_DESTINATION_IPS_PER_SECOND = "distinct_destination_ips_per_second"
    SYN_FRACTION_PER_SECOND = "syn_fraction_per_second"


class GothamLabel(StrEnum):
    BENIGN = "benign"
    NORMAL = "normal"
    UNKNOWN = "unknown"
    UNLABELLED = "unlabelled"
    UNLABELED = "unlabeled"
    UNASSIGNED = "-"

    @property
    def is_benign(self) -> bool:
        return self in (GothamLabel.BENIGN, GothamLabel.NORMAL)

    @classmethod
    def known(cls, text: "SourceCell") -> "GothamLabel | None":
        try:
            return cls(text.lower())
        except ValueError:
            return None


class TcpFlagMask(IntEnum):
    FIN = 0x01
    SYN = 0x02
    RST = 0x04
    PSH = 0x08
    ACK = 0x10

    @classmethod
    def feature_order(cls) -> tuple["TcpFlagMask", ...]:
        return cls.SYN, cls.ACK, cls.FIN, cls.RST, cls.PSH


class IpProtocolNumber(IntEnum):
    ICMP = 1
    TCP = 6
    UDP = 17
    GRE = 47

    @classmethod
    def feature_order(cls) -> tuple["IpProtocolNumber", ...]:
        return cls.TCP, cls.UDP, cls.ICMP, cls.GRE


class CalendarMonth(StrEnum):
    JANUARY = "Jan"
    FEBRUARY = "Feb"
    MARCH = "Mar"
    APRIL = "Apr"
    MAY = "May"
    JUNE = "Jun"
    JULY = "Jul"
    AUGUST = "Aug"
    SEPTEMBER = "Sep"
    OCTOBER = "Oct"
    NOVEMBER = "Nov"
    DECEMBER = "Dec"

    @property
    def number(self) -> MonthNumber:
        return MonthNumber(list(CalendarMonth).index(self) + 1)


class TimestampPart(StrEnum):
    MONTH = "month"
    DAY = "day"
    YEAR = "year"
    HOUR = "hour"
    MINUTE = "minute"
    SECOND = "second"
    FRACTION = "fraction"


class Iot23Column(StrEnum):
    TIMESTAMP = "ts"
    LABEL = "label"
    DETAILED_LABEL = "detailed-label"
    TUNNEL_PARENTS = "tunnel_parents"
    DURATION = "duration"
    ORIGIN_BYTES = "orig_bytes"
    RESPONDER_BYTES = "resp_bytes"
    MISSED_BYTES = "missed_bytes"
    ORIGIN_PACKETS = "orig_pkts"
    ORIGIN_IP_BYTES = "orig_ip_bytes"
    RESPONDER_PACKETS = "resp_pkts"
    RESPONDER_IP_BYTES = "resp_ip_bytes"
    ORIGIN_PORT = "id.orig_p"
    RESPONDER_PORT = "id.resp_p"
    CONNECTION_STATE = "conn_state"
    PROTOCOL = "proto"
    SERVICE = "service"
    HISTORY = "history"

    @classmethod
    def numeric(cls) -> tuple["Iot23Column", ...]:
        return (
            cls.DURATION,
            cls.ORIGIN_BYTES,
            cls.RESPONDER_BYTES,
            cls.MISSED_BYTES,
            cls.ORIGIN_PACKETS,
            cls.ORIGIN_IP_BYTES,
            cls.RESPONDER_PACKETS,
            cls.RESPONDER_IP_BYTES,
        )

    @classmethod
    def ports(cls) -> tuple["Iot23Column", "Iot23Column"]:
        return cls.ORIGIN_PORT, cls.RESPONDER_PORT

    @classmethod
    def expanded_terminal(cls) -> tuple["Iot23Column", "Iot23Column", "Iot23Column"]:
        return cls.TUNNEL_PARENTS, cls.LABEL, cls.DETAILED_LABEL

    @property
    def port_features(self) -> tuple[FeatureName, FeatureName]:
        match self:
            case Iot23Column.ORIGIN_PORT:
                return FeatureName("orig_p_privileged"), FeatureName("orig_p_value")
            case Iot23Column.RESPONDER_PORT:
                return FeatureName("resp_p_privileged"), FeatureName("resp_p_value")
            case _:
                raise ValueError(f"{self.name} is not an IoT-23 port column")


class Iot23Label(StrEnum):
    BENIGN = "benign"
    MALICIOUS = "malicious"


class Iot23AttackName(StrEnum):
    UNSPECIFIED = "UNSPECIFIED"
    HORIZONTAL_PORT_SCAN = "HORIZONTAL_PORT_SCAN"
    VERTICAL_PORT_SCAN = "VERTICAL_PORT_SCAN"
    COMMAND_AND_CONTROL = "COMMAND_AND_CONTROL"
    COMMAND_AND_CONTROL_HEARTBEAT = "COMMAND_AND_CONTROL_HEARTBEAT"


class Iot23DetailAlias(StrEnum):
    PART_OF_A_HORIZONTAL_PORT_SCAN = "PART_OF_A_HORIZONTAL_PORT_SCAN"
    PART_OF_A_VERTICAL_PORT_SCAN = "PART_OF_A_VERTICAL_PORT_SCAN"
    C_C = "C_C"
    C_C_HEARTBEAT = "C_C_HEARTBEAT"
    C_C_HEART_BEAT = "C_C_HEART_BEAT"
    C_C_CHANNEL = "C_C_CHANNEL"

    @property
    def attack_name(self) -> Iot23AttackName:
        match self:
            case Iot23DetailAlias.PART_OF_A_HORIZONTAL_PORT_SCAN:
                return Iot23AttackName.HORIZONTAL_PORT_SCAN
            case Iot23DetailAlias.PART_OF_A_VERTICAL_PORT_SCAN:
                return Iot23AttackName.VERTICAL_PORT_SCAN
            case Iot23DetailAlias.C_C | Iot23DetailAlias.C_C_CHANNEL:
                return Iot23AttackName.COMMAND_AND_CONTROL
            case Iot23DetailAlias.C_C_HEARTBEAT | Iot23DetailAlias.C_C_HEART_BEAT:
                return Iot23AttackName.COMMAND_AND_CONTROL_HEARTBEAT


class Iot23SourceName(StrEnum):
    CONNECTION_LOG = "conn.log.labeled"
    ARCHIVE_SUFFIX = ".tar.gz"
    ARCHIVE = "iot_23_datasets_small.tar.gz"


class ZeekHeaderToken(StrEnum):
    FIELDS = "#fields"


class ChunkThinning(Enum):
    NONE = "NONE"
    HISTORICAL = "HISTORICAL"


class TonIotColumn(StrEnum):
    TIMESTAMP = "ts"
    SOURCE_IP = "src_ip"
    DESTINATION_IP = "dst_ip"
    LABEL = "label"
    TYPE = "type"
    DURATION = "duration"
    SOURCE_BYTES = "src_bytes"
    DESTINATION_BYTES = "dst_bytes"
    MISSED_BYTES = "missed_bytes"
    SOURCE_PACKETS = "src_pkts"
    SOURCE_IP_BYTES = "src_ip_bytes"
    DESTINATION_PACKETS = "dst_pkts"
    DESTINATION_IP_BYTES = "dst_ip_bytes"
    SOURCE_PORT = "src_port"
    DESTINATION_PORT = "dst_port"
    CONNECTION_STATE = "conn_state"
    PROTOCOL = "proto"
    SERVICE = "service"

    @classmethod
    def numeric(cls) -> tuple["TonIotColumn", ...]:
        return (
            cls.DURATION,
            cls.SOURCE_BYTES,
            cls.DESTINATION_BYTES,
            cls.MISSED_BYTES,
            cls.SOURCE_PACKETS,
            cls.SOURCE_IP_BYTES,
            cls.DESTINATION_PACKETS,
            cls.DESTINATION_IP_BYTES,
        )

    @classmethod
    def ports(cls) -> tuple["TonIotColumn", "TonIotColumn"]:
        return cls.SOURCE_PORT, cls.DESTINATION_PORT

    @property
    def port_features(self) -> tuple[FeatureName, FeatureName]:
        match self:
            case TonIotColumn.SOURCE_PORT:
                return FeatureName("src_port_privileged"), FeatureName("src_port_value")
            case TonIotColumn.DESTINATION_PORT:
                return FeatureName("dst_port_privileged"), FeatureName("dst_port_value")
            case _:
                raise ValueError(f"{self.name} is not a TON-IoT port column")


class TonIotBinaryLabel(StrEnum):
    NORMAL = "0"
    ATTACK = "1"


class TonIotType(StrEnum):
    NORMAL = "normal"
    DDOS = "ddos"
    DOS = "dos"
    SCANNING = "scanning"
    PASSWORD = "password"
    BACKDOOR = "backdoor"
    INJECTION = "injection"
    RANSOMWARE = "ransomware"
    MITM = "mitm"
    XSS = "xss"

    @property
    def attack_type_name(self) -> "AttackTypeName | None":
        match self:
            case TonIotType.NORMAL:
                return None
            case TonIotType.DDOS:
                return AttackTypeName("DDOS")
            case TonIotType.DOS:
                return AttackTypeName("DOS")
            case TonIotType.SCANNING:
                return AttackTypeName("SCANNING")
            case TonIotType.PASSWORD:
                return AttackTypeName("PASSWORD")
            case TonIotType.BACKDOOR:
                return AttackTypeName("BACKDOOR")
            case TonIotType.INJECTION:
                return AttackTypeName("INJECTION")
            case TonIotType.RANSOMWARE:
                return AttackTypeName("RANSOMWARE")
            case TonIotType.MITM:
                return AttackTypeName("MAN_IN_THE_MIDDLE")
            case TonIotType.XSS:
                return AttackTypeName("CROSS_SITE_SCRIPTING")

    @property
    def binary_label(self) -> TonIotBinaryLabel:
        if self is TonIotType.NORMAL:
            return TonIotBinaryLabel.NORMAL
        return TonIotBinaryLabel.ATTACK

    @classmethod
    def known(cls, text: "SourceCell") -> "TonIotType | None":
        try:
            return cls(text)
        except ValueError:
            return None


class IntegerBase(IntEnum):
    AUTO = 0
    DECIMAL = 10
    HEXADECIMAL = 16


@dataclass(frozen=True, slots=True)
class SourceRecord:
    cells: dict[SourceColumn, SourceCell]

    def cell(self, column: StrEnum) -> SourceCell:
        return self.cells[SourceColumn(column)]


class RecordCommentPolicy(Enum):
    NONE = "NONE"
    ZEEK = "ZEEK"

    @property
    def marker(self) -> "SourceLine | None":
        match self:
            case RecordCommentPolicy.NONE:
                return None
            case RecordCommentPolicy.ZEEK:
                return SourceLine(b"#")


class FieldDelimiter(Enum):
    COMMA = "COMMA"
    TAB = "TAB"

    @property
    def text(self) -> "SourceCell":
        match self:
            case FieldDelimiter.COMMA:
                return SourceCell(",")
            case FieldDelimiter.TAB:
                return SourceCell("\t")

    @property
    def token(self) -> "SourceLine":
        return SourceLine(self.text.encode(TextEncoding.ASCII))


class SearchBound(Enum):
    SHARE_CEILING = 0.999

    def __init__(self, bound: FalseAlarmBudget) -> None:
        self.bound = bound


class TimeScale(IntEnum):
    MICROSECONDS_PER_SECOND = 1_000_000
    NANOSECONDS_PER_SECOND = 1_000_000_000


class LineBreak(Enum):
    LINE_FEED = "LINE_FEED"
    CARRIAGE_RETURN = "CARRIAGE_RETURN"

    @property
    def token(self) -> "SourceLine":
        match self:
            case LineBreak.LINE_FEED:
                return SourceLine(b"\n")
            case LineBreak.CARRIAGE_RETURN:
                return SourceLine(b"\r")


class ChronologicalOrder(Enum):
    ALREADY_ORDERED = "ALREADY_ORDERED"
    REORDERED = "REORDERED"


class PopulationCounting(Enum):
    COUNTED = "COUNTED"
    UNCOUNTED = "UNCOUNTED"


class StreamBufferSize(IntEnum):
    READER = 1 << 20
    READ_AHEAD_CHUNK = 1 << 24
    RECORD_CHUNK = 1 << 25


class QueueDepth(IntEnum):
    READ_AHEAD = 4


class RecordBlockRows(IntEnum):
    STANDARD = 500_000


class DecompressionWorkers(IntEnum):
    MAXIMUM = 8


class PollInterval(Enum):
    SOURCE_QUEUE = timedelta(milliseconds=100)

    def __init__(self, interval: timedelta) -> None:
        self.interval = interval


class SharedFeature(StrEnum):
    SERVICE_KNOWN = "service_known"
    HISTORY_LENGTH = "history_length"


class FeatureGroup(StrEnum):
    STATE = "state"
    PROTOCOL = "proto"
    DIRECTION = "dir"
    HISTORY = "history"
    STACK = "stack"

    def feature(self, member: StrEnum) -> FeatureName:
        return FeatureName(f"{self}_{member}")


class Iot23Capture(StrEnum):
    CAPTURE_1_1 = "1-1"
    CAPTURE_3_1 = "3-1"
    CAPTURE_7_1 = "7-1"
    CAPTURE_8_1 = "8-1"
    CAPTURE_9_1 = "9-1"
    CAPTURE_20_1 = "20-1"
    CAPTURE_21_1 = "21-1"
    CAPTURE_34_1 = "34-1"
    CAPTURE_35_1 = "35-1"
    CAPTURE_36_1 = "36-1"
    CAPTURE_42_1 = "42-1"
    CAPTURE_48_1 = "48-1"
    CAPTURE_49_1 = "49-1"
    CAPTURE_60_1 = "60-1"
    CAPTURE_4_1 = "4-1"
    CAPTURE_5_1 = "5-1"

    @property
    def cohort_role(self) -> "TargetCohortRole":
        match self:
            case (
                Iot23Capture.CAPTURE_20_1
                | Iot23Capture.CAPTURE_21_1
                | Iot23Capture.CAPTURE_42_1
                | Iot23Capture.CAPTURE_4_1
                | Iot23Capture.CAPTURE_5_1
            ):
                return TargetCohortRole.PEER_ONLY
            case _:
                return TargetCohortRole.REPORTED_TARGET


class GothamDevice(StrEnum):
    AIR_QUALITY = "iotsim-air-quality-1"
    BUILDING_MONITOR = "iotsim-building-monitor-1"
    CITY_POWER = "iotsim-city-power-1"
    COMBINED_CYCLE = "iotsim-combined-cycle-1"
    DOMOTIC_MONITOR = "iotsim-domotic-monitor-1"
    IP_CAMERA_MUSEUM = "iotsim-ip-camera-museum-1"
    IP_CAMERA_STREET = "iotsim-ip-camera-street-1"


class NBaIoTDevice(StrEnum):
    DANMINI_DOORBELL = "Danmini_Doorbell"
    ECOBEE_THERMOSTAT = "Ecobee_Thermostat"
    ENNIO_DOORBELL = "Ennio_Doorbell"
    PHILIPS_BABY_MONITOR = "Philips_B120N10_Baby_Monitor"
    PROVISION_737E_CAMERA = "Provision_PT_737E_Security_Camera"
    PROVISION_838_CAMERA = "Provision_PT_838_Security_Camera"
    SAMSUNG_WEBCAM = "Samsung_SNH_1011_N_Webcam"
    SIMPLEHOME_1002_CAMERA = "SimpleHome_XCS7_1002_WHT_Security_Camera"
    SIMPLEHOME_1003_CAMERA = "SimpleHome_XCS7_1003_WHT_Security_Camera"


class ZeekConnectionState(StrEnum):
    S0 = "S0"
    S1 = "S1"
    SF = "SF"
    REJ = "REJ"
    S2 = "S2"
    S3 = "S3"
    RSTO = "RSTO"
    RSTR = "RSTR"
    RSTOS0 = "RSTOS0"
    RSTRH = "RSTRH"
    SH = "SH"
    SHR = "SHR"
    OTH = "OTH"


class CtuArgusState(StrEnum):
    S_RA = "S_RA"
    SR_A = "SR_A"
    SRPA_SPA = "SRPA_SPA"
    FSPA_FSPA = "FSPA_FSPA"
    SPA_SPA = "SPA_SPA"
    S = "S_"
    SR_SA = "SR_SA"
    A_A = "A_A"
    RA_A = "RA_A"
    PA_PA = "PA_PA"
    SPA_SA = "SPA_SA"
    URP = "URP"
    CON = "CON"
    INT = "INT"
    REQ = "REQ"
    RST = "RST"
    ECO = "ECO"
    FPA_FPA = "FPA_FPA"
    S_SA = "S_SA"
    SA_A = "SA_A"


class CtuFlowDirection(StrEnum):
    ORIGIN_TO_RESPONDER = "->"
    BIDIRECTIONAL = "<->"
    RESPONDER_TO_ORIGIN = "<-"
    UNKNOWN = "<?>"
    WHO = "who"


class Ctu13Scenario(StrEnum):
    SCENARIO_01 = "01"
    SCENARIO_02 = "02"
    SCENARIO_03 = "03"
    SCENARIO_04 = "04"
    SCENARIO_05 = "05"
    SCENARIO_06 = "06"
    SCENARIO_07 = "07"
    SCENARIO_08 = "08"
    SCENARIO_09 = "09"
    SCENARIO_10 = "10"
    SCENARIO_11 = "11"
    SCENARIO_12 = "12"
    SCENARIO_13 = "13"

    @property
    def client_name(self) -> ClientName:
        return ClientName(f"CTU13-scenario-{self}")

    @property
    def cohort_role(self) -> TargetCohortRole:
        if self is Ctu13Scenario.SCENARIO_07:
            return TargetCohortRole.PEER_ONLY
        return TargetCohortRole.REPORTED_TARGET

    @property
    def attack_family(self) -> "Ctu13AttackFamily":
        match self:
            case Ctu13Scenario.SCENARIO_01 | Ctu13Scenario.SCENARIO_02 | Ctu13Scenario.SCENARIO_09:
                return Ctu13AttackFamily.NERIS
            case (
                Ctu13Scenario.SCENARIO_03
                | Ctu13Scenario.SCENARIO_04
                | Ctu13Scenario.SCENARIO_10
                | Ctu13Scenario.SCENARIO_11
            ):
                return Ctu13AttackFamily.RBOT
            case Ctu13Scenario.SCENARIO_05 | Ctu13Scenario.SCENARIO_13:
                return Ctu13AttackFamily.VIRUT
            case Ctu13Scenario.SCENARIO_06:
                return Ctu13AttackFamily.MENTI
            case Ctu13Scenario.SCENARIO_07:
                return Ctu13AttackFamily.SOGOU
            case Ctu13Scenario.SCENARIO_08:
                return Ctu13AttackFamily.MURLO
            case Ctu13Scenario.SCENARIO_12:
                return Ctu13AttackFamily.NSIS_AY


class Ctu13AttackFamily(StrEnum):
    NERIS = "NERIS"
    RBOT = "RBOT"
    VIRUT = "VIRUT"
    MENTI = "MENTI"
    SOGOU = "SOGOU"
    MURLO = "MURLO"
    NSIS_AY = "NSIS.AY"


class PacketProtocolToken(StrEnum):
    COAP = "coap"
    MQTT = "mqtt"
    RTSP = "rtsp"
    TLS = "tls"
    HTTP = "http"
    TELNET = "telnet"
    DNS = "dns"
    GRE = "gre"
    ICMP = "icmp"
    SSH = "ssh"
    RTP = "rtp"
    SDP = "sdp"


class TcpHistoryFlag(StrEnum):
    SYN = "S"
    HARDWARE = "h"
    ACK = "a"
    DATA = "d"
    FIN = "f"
    RESET = "r"


@dataclass(frozen=True, slots=True)
class SyntheticFixtureShape:
    feature_count: FeatureCount
    benign_rows: ObservationCount
    attack_rows: ObservationCount
    peer_benign_rows: ObservationCount
    peer_attack_rows: ObservationCount
    attack_shift: FixtureShift

    @classmethod
    def standard(cls) -> "SyntheticFixtureShape":
        return cls(
            feature_count=8,
            benign_rows=1250,
            attack_rows=40,
            peer_benign_rows=400,
            peer_attack_rows=200,
            attack_shift=2.0,
        )


@dataclass(frozen=True, slots=True)
class ReservoirCaps:
    benign: ObservationCount
    attack: ObservationCount

    @classmethod
    def standard(cls, dataset: DatasetName) -> "ReservoirCaps":
        match dataset:
            case DatasetName.CTU_13:
                return cls(benign=200_000, attack=60_000)
            case DatasetName.GOTHAM_2025:
                return cls(benign=200_000, attack=40_000)
            case _:
                raise ValueError(f"{dataset.name} has no standard reservoir caps")


@dataclass(frozen=True, slots=True)
class Iot23Sampling:
    benign_cap: ObservationCount
    malicious_cap_per_detail: ObservationCount
    chunk_rows: ObservationCount
    thin_group_min_rows: ObservationCount
    thin_fraction: RetentionFraction
    large_file_bytes: SourceByteCount

    @classmethod
    def standard(cls) -> "Iot23Sampling":
        return cls(
            benign_cap=200_000,
            malicious_cap_per_detail=60_000,
            chunk_rows=1_000_000,
            thin_group_min_rows=20_000,
            thin_fraction=0.02,
            large_file_bytes=1_500_000_000,
        )


@dataclass(frozen=True, slots=True)
class TonIotSampling:
    benign_cap: ObservationCount
    attack_cap_per_label: ObservationCount
    source_chunk_rows: ObservationCount
    chunk_group_threshold: ObservationCount
    chunk_keep_fraction: RetentionFraction
    chunk_reservoir_cap: ObservationCount

    @classmethod
    def standard(cls) -> "TonIotSampling":
        return cls(
            benign_cap=150_000,
            attack_cap_per_label=60_000,
            source_chunk_rows=2_000_000,
            chunk_group_threshold=20_000,
            chunk_keep_fraction=0.03,
            chunk_reservoir_cap=60_000,
        )


@dataclass(frozen=True, slots=True)
class AttackCapture:
    attack_type: AttackType
    source_file: Path
    features: FloatMatrix
    source_order: RowIndexArray
    source_population_rows: ObservationCount | None = None

    def __post_init__(self) -> None:
        _validate_feature_matrix(self.features)
        _validate_source_order(self.source_order, self.features.shape[0])
        if (
            self.source_population_rows is not None
            and self.source_population_rows < self.features.shape[0]
        ):
            raise ValueError("Attack source population cannot be smaller than its retained sample")
        self.features.setflags(write=False)
        self.source_order.setflags(write=False)


@dataclass(frozen=True, slots=True)
class DatasetClientData:
    client: DatasetClientId
    study_stratum: StudyStratum
    feature_schema: FeatureSchemaId
    feature_names: tuple[FeatureName, ...]
    benign_features: FloatMatrix
    benign_source_order: RowIndexArray
    attack_captures: tuple[AttackCapture, ...]
    target_cohort_role: TargetCohortRole = TargetCohortRole.ELIGIBILITY_DRIVEN
    benign_source_population_rows: ObservationCount | None = None

    def __post_init__(self) -> None:
        if self.study_stratum.dataset is not self.client.dataset:
            raise ValueError("Study stratum must belong to the client's dataset")
        if self.feature_schema.dataset is not self.client.dataset:
            raise ValueError("Feature schema must belong to the client's dataset")
        _validate_feature_matrix(self.benign_features)
        _validate_source_order(self.benign_source_order, self.benign_features.shape[0])
        if (
            self.benign_source_population_rows is not None
            and self.benign_source_population_rows < self.benign_features.shape[0]
        ):
            raise ValueError("Benign source population cannot be smaller than its retained sample")
        if len(self.feature_names) != self.benign_features.shape[1]:
            raise ValueError("Feature names must align with benign feature columns")
        if len(set(self.feature_names)) != len(self.feature_names):
            raise ValueError("Feature names must be unique")
        attack_types = tuple(capture.attack_type for capture in self.attack_captures)
        if len(set(attack_types)) != len(attack_types):
            raise ValueError("Dataset clients cannot repeat attack type captures")
        if any(
            capture.features.shape[1] != self.benign_features.shape[1]
            for capture in self.attack_captures
        ):
            raise ValueError("Benign and attack feature schemas must align")
        self.benign_features.setflags(write=False)
        self.benign_source_order.setflags(write=False)


DatasetClientLoader = Callable[[Path, StudyStratum, LabNetwork], tuple[DatasetClientData, ...]]
DatasetSourceDiscovery = Callable[[Path, StudyStratum], tuple[Path, ...]]


@dataclass(frozen=True, slots=True)
class ChronologicalIndices:
    training: RowIndexArray
    reference: RowIndexArray
    calibration: RowIndexArray
    test: RowIndexArray

    def by_role(self) -> tuple[tuple[SplitRole, RowIndexArray], ...]:
        return (
            (SplitRole.TRAINING, self.training),
            (SplitRole.REFERENCE, self.reference),
            (SplitRole.CALIBRATION, self.calibration),
            (SplitRole.TEST, self.test),
        )

    def __post_init__(self) -> None:
        roles = (self.training, self.reference, self.calibration, self.test)
        if any(
            indices.ndim != 1
            or indices.size == 0
            or not np.issubdtype(indices.dtype, np.integer)
            or np.any(indices < 0)
            for indices in roles
        ):
            raise ValueError("Every chronological role requires non-negative row indices")
        combined = np.concatenate(roles)
        if np.unique(combined).size != combined.size:
            raise ValueError("Chronological roles must contain distinct rows")
        for indices in roles:
            indices.setflags(write=False)


@dataclass(frozen=True, slots=True)
class AttackHalfIndices:
    peer_development: RowIndexArray
    held_out: RowIndexArray

    def __post_init__(self) -> None:
        halves = (self.peer_development, self.held_out)
        if any(
            indices.ndim != 1
            or indices.size == 0
            or not np.issubdtype(indices.dtype, np.integer)
            or np.any(indices < 0)
            for indices in halves
        ):
            raise ValueError("Attack halves require non-empty non-negative row indices")
        if np.intersect1d(*halves).size:
            raise ValueError("Attack development and held-out rows must be disjoint")
        for indices in halves:
            indices.setflags(write=False)


@dataclass(frozen=True, slots=True)
class AttackCaptureSplit:
    attack_type: AttackType
    indices: AttackHalfIndices


@dataclass(frozen=True, slots=True)
class DatasetSplitPlan:
    client: DatasetClientId
    benign_roles: ChronologicalIndices
    attack_splits: tuple[AttackCaptureSplit, ...]
    excluded_attack_types: tuple[AttackType, ...]

    def __post_init__(self) -> None:
        attack_types = (
            tuple(split.attack_type for split in self.attack_splits) + self.excluded_attack_types
        )
        if len(set(attack_types)) != len(attack_types):
            raise ValueError("Every attack type must appear exactly once in a split plan")


@dataclass(frozen=True, slots=True)
class DatasetInventoryReport:
    data_root: Path
    sources: tuple[DatasetSourceInventory, ...]

    def __post_init__(self) -> None:
        discovered = tuple(source.dataset for source in self.sources)
        if len(discovered) != len(DatasetName) or set(discovered) != set(DatasetName):
            raise ValueError("Dataset inventories must cover every supported source exactly once")

    @property
    def is_complete(self) -> bool:
        return all(source.is_available for source in self.sources)


CURRENT_ARTIFACT_SCHEMA: ArtifactSchemaVersion = 16


@dataclass(frozen=True, slots=True)
class InputDigestMaterial:
    chunks: tuple[ArtifactPayload, ...]


@dataclass(frozen=True, slots=True)
class ArtifactPath:
    path: Path


@dataclass(frozen=True, slots=True)
class ArtifactVerification:
    state: VerificationState
    content_digest: ScientificDigest


@dataclass(frozen=True, slots=True)
class SyntheticWorkflowResult:
    artifact_path: ArtifactPath
    verification: ArtifactVerification
    reuse_state: ArtifactReuseState


@dataclass(frozen=True, slots=True)
class CampaignWorkflowResult:
    artifact_path: ArtifactPath
    verification: ArtifactVerification
    reuse_state: ArtifactReuseState
    target_count: ObservationCount


@dataclass(frozen=True, slots=True)
class CampaignStatus:
    dataset: DatasetName
    study_stratum: StudyStratum
    artifact_path: ArtifactPath
    state: CampaignCompletionState
    target_count: ObservationCount | None


@dataclass(frozen=True, slots=True, kw_only=True)
class DetectionOutcome:
    expert_ids: tuple[ExpertId, ...]
    local_threshold: Threshold
    expert_thresholds: tuple[Threshold, ...]
    joint_multiplier: JointMultiplier
    peer_block_alert_rates: tuple[FalseAlertRate, ...]
    evaluation_count: ObservationCount
    alert_count: ObservationCount
    benign_alert_rate: FalseAlertRate
    attack_detection_rate: DetectionRate
    local_alerts: tuple[AlertDecision, ...]
    peer_alerts: tuple[AlertDecision, ...]
    alerts: tuple[AlertDecision, ...]


@dataclass(frozen=True, slots=True, kw_only=True)
class AttackTypePerformance:
    attack_type: AttackType
    test_rows: ObservationCount
    detection_rate: DetectionRate
    local_detection_rate: DetectionRate
    max_fusion_detection_rate: DetectionRate
    peer_union_detection_rate: DetectionRate
    bonferroni_detection_rate: DetectionRate


@dataclass(frozen=True, slots=True, kw_only=True)
class ArmPerformance:
    arm: StudyArm
    false_alert_rate: FalseAlertRate
    detection_rates: tuple[DetectionRate, ...]


@dataclass(frozen=True, slots=True)
class ModelSeedProvenance:
    owner: DatasetClientId
    seed: LearningSeed
    domain: SeedDomain


@dataclass(frozen=True, slots=True, kw_only=True)
class TargetPerformance:
    target: DatasetClientId
    seed: LearningSeed
    split_digest: ScientificDigest
    peer_model_seeds: tuple[ModelSeedProvenance, ...]
    local_model_seed: ModelSeedProvenance
    benign_test_rows: ObservationCount
    realized_false_alert_rate: FalseAlertRate
    attack_types: tuple[AttackTypePerformance, ...]
    local_false_alert_rate: FalseAlertRate
    max_fusion_false_alert_rate: FalseAlertRate
    peer_union_false_alert_rate: FalseAlertRate
    bonferroni_false_alert_rate: FalseAlertRate
    peer_count: MinimumPeerCount
    budget_sweep: tuple[BudgetSweepPerformance, ...]
    study_stratum: StudyStratum
    arms: tuple[ArmPerformance, ...] = ()

    def __post_init__(self) -> None:
        self._validate_arms()
        self._validate_coverage()
        self._validate_seed_provenance()

    def _validate_arms(self) -> None:
        arm_names = tuple(performance.arm for performance in self.arms)
        if len(set(arm_names)) != len(arm_names):
            raise ValueError("Target performance cannot repeat study arms")
        if any(
            len(performance.detection_rates) != len(self.attack_types) for performance in self.arms
        ):
            raise ValueError("Arm detection rates must cover every held-out attack type")

    def _validate_coverage(self) -> None:
        if self.study_stratum.dataset is not self.target.dataset:
            raise ValueError("Target study stratum must belong to its dataset")
        if not self.attack_types:
            raise ValueError("Target performance requires held-out attack types")
        if len(self.budget_sweep) < 2:
            raise ValueError("Target performance requires a budget sweep with at least two points")
        multipliers = tuple(point.multiplier for point in self.budget_sweep)
        if tuple(sorted(set(multipliers))) != multipliers:
            raise ValueError("Budget sweep multipliers must be unique and increasing")
        attack_types = tuple(result.attack_type for result in self.attack_types)
        if len(set(attack_types)) != len(attack_types):
            raise ValueError("Target performance cannot repeat attack types")

    def _validate_seed_provenance(self) -> None:
        if len(self.peer_model_seeds) != self.peer_count:
            raise ValueError("Peer seed provenance must cover every contributing expert")
        peer_owners = tuple(seed.owner for seed in self.peer_model_seeds)
        if len(set(peer_owners)) != len(peer_owners) or self.target in peer_owners:
            raise ValueError("Peer seed provenance must identify distinct non-target peers")
        if any(owner.dataset is not self.target.dataset for owner in peer_owners):
            raise ValueError("Peer seed provenance must come from the target's dataset")
        if any(seed.domain is not SeedDomain.PEER_MODEL for seed in self.peer_model_seeds):
            raise ValueError("Peer model seeds must use the peer-model seed domain")
        if (
            self.local_model_seed.owner != self.target
            or self.local_model_seed.domain is not SeedDomain.LOCAL_MODEL
        ):
            raise ValueError("Local model seed provenance must belong to the target")


@dataclass(frozen=True, slots=True, kw_only=True)
class TargetMetricSummary:
    target: DatasetClientId
    method_detection_rate: DetectionRate
    local_detection_rate: DetectionRate
    max_fusion_detection_rate: DetectionRate
    bonferroni_detection_rate: DetectionRate
    detection_difference: RateDifference
    max_fusion_detection_difference: RateDifference
    method_max_fusion_difference: RateDifference
    realized_false_alert_rate: FalseAlertRate
    local_false_alert_rate: FalseAlertRate
    max_fusion_false_alert_rate: FalseAlertRate
    bonferroni_false_alert_rate: FalseAlertRate
    matched_fpr_difference: RateDifference | None


@dataclass(frozen=True, slots=True)
class TargetPairedDifference:
    target: DatasetClientId
    difference: RateDifference


class BudgetAnalysisKind(Enum):
    UNIVERSAL_PRIMARY = "UNIVERSAL_PRIMARY"
    UNIVERSAL_ROBUSTNESS = "UNIVERSAL_ROBUSTNESS"
    BROAD_SCALING = "BROAD_SCALING"
    HIGH_RESOURCE = "HIGH_RESOURCE"

    @property
    def title(self) -> "DisplayText":
        match self:
            case BudgetAnalysisKind.UNIVERSAL_PRIMARY:
                return DisplayText("Universal primary analysis")
            case BudgetAnalysisKind.UNIVERSAL_ROBUSTNESS:
                return DisplayText("Universal robustness analysis")
            case BudgetAnalysisKind.BROAD_SCALING:
                return DisplayText("Broad scaling analysis")
            case BudgetAnalysisKind.HIGH_RESOURCE:
                return DisplayText("Extreme high-resource analysis")


class BudgetOutcome(Enum):
    PACE_TPR = "PACE_TPR"
    LOCAL_TPR = "LOCAL_TPR"
    MAX_FUSION_TPR = "MAX_FUSION_TPR"
    PACE_FALSE_ALERT_RATE = "PACE_FALSE_ALERT_RATE"

    @property
    def display_label(self) -> "DisplayText":
        match self:
            case BudgetOutcome.PACE_TPR:
                return DisplayText("PACE TPR")
            case BudgetOutcome.LOCAL_TPR:
                return DisplayText("local TPR")
            case BudgetOutcome.MAX_FUSION_TPR:
                return DisplayText("max-fusion TPR")
            case BudgetOutcome.PACE_FALSE_ALERT_RATE:
                return DisplayText("PACE FPR")


class BudgetAnalysisScope(Enum):
    POOLED = "POOLED"
    COMPLETE_STRATUM = "COMPLETE_STRATUM"


@dataclass(frozen=True, slots=True)
class BudgetOutcomeDifference:
    outcome: BudgetOutcome
    mean_difference: RateDifference
    interval: RateConfidenceInterval | None


@dataclass(frozen=True, slots=True)
class StratumBudgetEffect:
    stratum: StudyStratum
    target_count: ObservationCount
    differences: tuple[BudgetOutcomeDifference, ...]


@dataclass(frozen=True, slots=True)
class BudgetComparison:
    alpha: AlphaLevel
    from_rows: LocalTrainingRowCount
    to_rows: LocalTrainingRowCount
    target_count: ObservationCount
    pooled: tuple[BudgetOutcomeDifference, ...]
    strata: tuple[StratumBudgetEffect, ...]


@dataclass(frozen=True, slots=True)
class BudgetEffectAnalysis:
    kind: BudgetAnalysisKind
    scope: BudgetAnalysisScope
    strata: tuple[StudyStratum, ...]
    budgets: tuple[LocalTrainingRowCount, ...]
    target_count: ObservationCount
    comparisons: tuple[BudgetComparison, ...]


@dataclass(frozen=True, slots=True)
class StratumFeasibility:
    stratum: StudyStratum
    eligible_targets: ObservationCount
    cohort_sizes: tuple[ObservationCount, ...]


@dataclass(frozen=True, slots=True)
class BudgetEffectReport:
    kinds: tuple[BudgetAnalysisKind, ...]
    feasibility: tuple[StratumFeasibility, ...]
    analyses: tuple[BudgetEffectAnalysis, ...]


@dataclass(frozen=True, slots=True)
class PowerPoint:
    difference_sd: PairedDifferenceSd
    minimum_detectable_effect: RateDifference
    power_at_meaningful_effect: StatisticalPower


@dataclass(frozen=True, slots=True)
class AnalysisPower:
    kind: BudgetAnalysisKind
    target_count: ObservationCount
    points: tuple[PowerPoint, ...]


@dataclass(frozen=True, slots=True)
class SeedRateDifference:
    seed: LearningSeed
    difference: RateDifference


@dataclass(frozen=True, slots=True)
class SeedFalseAlertRates:
    seed: LearningSeed
    method: FalseAlertRate
    arm: FalseAlertRate


@dataclass(frozen=True, slots=True)
class ArmTargetSummary:
    target: DatasetClientId
    nominal_difference: RateDifference
    matched_difference: RateDifference
    method_false_alert_rate: FalseAlertRate
    arm_false_alert_rate: FalseAlertRate
    seed_differences: tuple[SeedRateDifference, ...]
    seed_false_alert_rates: tuple[SeedFalseAlertRates, ...]


@dataclass(frozen=True, slots=True, kw_only=True)
class ArmMetricSummary:
    arm: StudyArm
    target_count: ObservationCount
    macro_detection_rate: DetectionRate
    macro_detection_difference: RateDifference
    method_arm_difference: RateDifference
    method_arm_confidence_interval: RateConfidenceInterval
    worst_detection_difference: RateDifference
    catastrophic_cell_count: ObservationCount
    harmed_target_count: ObservationCount
    mean_false_alert_rate: FalseAlertRate
    worst_false_alert_rate: FalseAlertRate
    targets: tuple[ArmTargetSummary, ...] = ()


@dataclass(frozen=True, slots=True, kw_only=True)
class CampaignMetricSummary:
    target_count: ObservationCount
    targets: tuple[TargetMetricSummary, ...]
    macro_detection_rate: DetectionRate
    local_macro_detection_rate: DetectionRate
    max_fusion_macro_detection_rate: DetectionRate
    bonferroni_macro_detection_rate: DetectionRate
    macro_detection_difference: RateDifference
    method_local_confidence_interval: RateConfidenceInterval
    max_fusion_macro_detection_difference: RateDifference
    method_max_fusion_confidence_interval: RateConfidenceInterval
    matched_fpr_target_count: ObservationCount
    matched_fpr_macro_difference: RateDifference | None
    matched_fpr_confidence_interval: RateConfidenceInterval | None
    mean_method_max_fusion_difference: RateDifference
    worst_method_max_fusion_difference: RateDifference
    improved_target_share: FalseAlertRate
    worst_detection_difference: RateDifference
    tenth_percentile_detection_difference: RateDifference
    catastrophic_cell_count: ObservationCount
    harmed_target_count: ObservationCount
    mean_false_alert_rate: FalseAlertRate
    median_false_alert_rate: FalseAlertRate
    ninetieth_percentile_false_alert_rate: FalseAlertRate
    worst_false_alert_rate: FalseAlertRate
    exceedance_share: FalseAlertRate
    local_mean_false_alert_rate: FalseAlertRate
    max_fusion_mean_false_alert_rate: FalseAlertRate
    max_fusion_worst_false_alert_rate: FalseAlertRate
    bonferroni_mean_false_alert_rate: FalseAlertRate
    bonferroni_worst_false_alert_rate: FalseAlertRate
    arms: tuple[ArmMetricSummary, ...] = ()


@dataclass(frozen=True, slots=True)
class CampaignReportResult:
    artifact_path: ArtifactPath
    metrics: CampaignMetricSummary
    reuse_state: ArtifactReuseState
    results: tuple["TargetPerformance", ...] = ()


@dataclass(frozen=True, slots=True, kw_only=True)
class ChannelDecisions:
    local_threshold: Threshold
    expert_thresholds: tuple[Threshold, ...]
    joint_multiplier: JointMultiplier
    peer_block_alert_rates: tuple[FalseAlertRate, ...]
    local_alerts: BooleanArray
    peer_alerts: BooleanArray
    combined_alerts: BooleanArray
    bank_thresholds: tuple[Threshold, ...] = ()

    def __post_init__(self) -> None:
        decisions = (self.local_alerts, self.peer_alerts, self.combined_alerts)
        if any(mask.ndim != 1 or mask.size == 0 for mask in decisions):
            raise ValueError("Channel decisions must contain non-empty one-dimensional masks")
        if len({mask.size for mask in decisions}) != 1:
            raise ValueError("Local, peer, and combined decision masks must align")
        if not np.array_equal(self.combined_alerts, self.local_alerts | self.peer_alerts):
            raise ValueError("Combined alerts must be the union of local and peer alerts")
        for mask in decisions:
            mask.setflags(write=False)


@dataclass(frozen=True, slots=True, kw_only=True)
class MaxFusionDecisions:
    local_threshold: Threshold
    peer_threshold: Threshold
    local_alerts: BooleanArray
    peer_alerts: BooleanArray
    combined_alerts: BooleanArray

    def __post_init__(self) -> None:
        decisions = (self.local_alerts, self.peer_alerts, self.combined_alerts)
        if any(mask.ndim != 1 or mask.size == 0 for mask in decisions):
            raise ValueError("Collapsed peer decisions require non-empty one-dimensional masks")
        if len({mask.size for mask in decisions}) != 1:
            raise ValueError("Collapsed peer decision masks must align")
        if not np.array_equal(self.combined_alerts, self.local_alerts | self.peer_alerts):
            raise ValueError("Collapsed peer alerts must be the union of local and peer alerts")
        for mask in decisions:
            mask.setflags(write=False)


@dataclass(frozen=True, slots=True)
class TargetEligibility:
    issues: tuple[TargetEligibilityIssue, ...]

    @property
    def is_eligible(self) -> bool:
        return not self.issues


@dataclass(frozen=True, slots=True)
class TargetBudgetSupport:
    client: DatasetClientId
    benign_rows: ObservationCount
    training_rows: ObservationCount
    calibration_rows: ObservationCount
    attack_type_count: ObservationCount
    eligibility: TargetEligibility

    def is_supported_at(self, budget: LocalTrainingRowCount) -> bool:
        return self.eligibility.is_eligible and self.training_rows >= budget


@dataclass(frozen=True, slots=True)
class BudgetAnalysis:
    kind: BudgetAnalysisKind
    budgets: tuple[LocalTrainingRowCount, ...]
    targets: tuple[ClientName, ...]

    def __post_init__(self) -> None:
        if not self.budgets or tuple(sorted(set(self.budgets))) != self.budgets:
            raise ValueError("Analysis budgets must be unique, increasing and non-empty")
        if not self.targets or len(set(self.targets)) != len(self.targets):
            raise ValueError("Analysis targets must be unique and non-empty")


@dataclass(frozen=True, slots=True)
class TargetTrainingRows:
    client: ClientName
    training_rows: ObservationCount
    attack_type_count: ObservationCount


@dataclass(frozen=True, slots=True)
class BudgetPlanSummary:
    analyses: tuple[BudgetAnalysis, ...]
    eligible_targets: ObservationCount
    targets: tuple[TargetTrainingRows, ...] = ()


@dataclass(frozen=True, slots=True)
class StratumBudgetPlan:
    stratum: StudyStratum
    requested_budgets: tuple[LocalTrainingRowCount, ...]
    candidates: tuple[TargetBudgetSupport, ...]
    analyses: tuple[BudgetAnalysis, ...]
    unavailable_budgets: tuple[LocalTrainingRowCount, ...]
    unobserved_budgets: tuple[LocalTrainingRowCount, ...]

    @property
    def eligible(self) -> tuple[TargetBudgetSupport, ...]:
        return tuple(target for target in self.candidates if target.eligibility.is_eligible)

    def supported_count(self, budget: LocalTrainingRowCount) -> ObservationCount:
        return sum(target.is_supported_at(budget) for target in self.candidates)


OverwriteOption = Annotated[bool, typer.Option("--overwrite")]
SmokeOption = Annotated[bool, typer.Option("--smoke")]
TargetArgument = Annotated[CommandTarget, typer.Argument(case_sensitive=False)]
OptionalTargetArgument = Annotated[CommandTarget, typer.Argument(case_sensitive=False)]


@dataclass(frozen=True, slots=True)
class CriterionVerdict:
    criterion: AcceptanceCriterion
    satisfied: CriterionSatisfied


@dataclass(frozen=True, slots=True, kw_only=True)
class AcceptanceCell:
    kind: BudgetAnalysisKind
    alpha: AlphaLevel
    local_training_rows: LocalTrainingRowCount
    target_count: ObservationCount
    nominal_gain: RateDifference
    nominal_interval: RateConfidenceInterval
    matched_gain: RateDifference
    matched_interval: RateConfidenceInterval
    gain_without_excluded: RateDifference | None
    gain_without_excluded_interval: RateConfidenceInterval | None
    harmed_target_count: ObservationCount
    severe_target_count: ObservationCount
    worst_target_difference: RateDifference
    false_alert_increase: RateDifference
    exceedance_increase: RateDifference
    seed_gains: tuple[SeedRateDifference, ...]
    minimum_stratum_left_out_gain: RateDifference
    verdicts: tuple[CriterionVerdict, ...]

    @property
    def is_gating(self) -> bool:
        return self.kind is BudgetAnalysisKind.UNIVERSAL_PRIMARY

    @property
    def is_accepted(self) -> bool:
        return all(verdict.satisfied for verdict in self.verdicts)


@dataclass(frozen=True, slots=True)
class AcceptanceReport:
    comparator: StudyArm
    cells: tuple[AcceptanceCell, ...]

    @property
    def is_accepted(self) -> bool:
        gating = tuple(cell for cell in self.cells if cell.is_gating)
        return len(gating) > 0 and all(cell.is_accepted for cell in gating)
