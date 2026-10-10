import copy
import shutil
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from pace import cli
from pace.config import (
    AnalysisSpec,
    PowerSettings,
    ProtocolMatrixSettings,
    ResolvedConfiguration,
    load_configuration,
)
from pace.datasets.adapters.ctu13 import load_ctu13_clients
from pace.datasets.adapters.gotham import load_gotham_clients
from pace.experiment.artifacts import (
    CampaignOutcome,
    EvidenceArtifact,
    Provenance,
    read_verified_artifact,
    reuse_or_write_artifact,
)
from pace.experiment.runner import run_dataset_study
from pace.types import (
    AlphaLevel,
    ArtifactKind,
    AttackCapture,
    AttackType,
    AttackTypeName,
    BudgetAnalysisKind,
    CampaignMetricSummary,
    ClientName,
    CorpusPath,
    DatasetClientData,
    DatasetClientId,
    DatasetName,
    ExecutionDevice,
    ExitStatus,
    FeatureName,
    FeatureSchemaId,
    LearningSeed,
    LocalTrainingRowCount,
    ModelSeedProvenance,
    RateConfidenceInterval,
    ReplacementPolicy,
    ReservoirCaps,
    ScientificDigest,
    SeedDomain,
    StudyStratum,
    TargetMetricSummary,
    TargetPerformance,
)

REPOSITORY_ROOT = Path(__file__).parents[1]


def campaign_configuration() -> ResolvedConfiguration:
    source = REPOSITORY_ROOT / "configs" / "default.yaml"
    base = load_configuration(source, Path.cwd())
    local = base.scientific.local_detectors.model_copy(
        update={
            "autoencoder_epochs": 1,
            "local_training_rows": 32,
            "detector_sample_limit": 32,
            "lof_neighbors": 5,
            "nearest_neighbors": 4,
            "isolation_forest_trees": 8,
        }
    )
    peer = base.scientific.peer_training.model_copy(update={"epochs": 1})
    scientific = base.scientific.model_copy(
        update={
            "minimum_peer_count": 4,
            "study_arms": (),
            "calibration_block_size": 10,
            "peer_training": peer,
            "local_detectors": local,
        }
    )
    split = base.splitting.model_copy(
        update={
            "minimum_calibration_rows": 20,
            "minimum_attack_rows": 30,
        }
    )
    return base.model_copy(update={"scientific": scientific, "splitting": split})


def campaign_devices() -> tuple[DatasetClientData, ...]:
    generator = np.random.default_rng(141)
    names = tuple(FeatureName(f"feature_{index}") for index in range(115))
    captures_by_device: list[DatasetClientData] = []
    attack_types = (
        AttackType(AttackTypeName("GAFGYT/COMBO")),
        AttackType(AttackTypeName("MIRAI/ACK")),
    )
    for client_index in range(5):
        benign = generator.normal(client_index * 0.1, 1.0, size=(120, 115))
        captures = tuple(
            AttackCapture(
                attack_type,
                Path(
                    f"{attack_type.identifier.split('/')[0].lower()}_attacks/"
                    f"{attack_type.identifier.split('/')[1].lower()}.csv"
                ),
                generator.normal(1.5 + client_index * 0.1, 1.0, size=(30, 115)),
                np.arange(30, dtype=np.int64),
            )
            for attack_type in attack_types
        )
        captures_by_device.append(
            DatasetClientData(
                DatasetClientId(DatasetName.N_BAIOT, ClientName(f"client-{client_index}")),
                StudyStratum.N_BAIOT,
                FeatureSchemaId.N_BAIOT_KITSUNE_SIGNED_LOG_V1,
                names,
                benign,
                np.arange(120, dtype=np.int64),
                captures,
            )
        )
    return tuple(captures_by_device)


INPUT = "1" * 64


IMPLEMENTATION = "2" * 64


def two_seed_configuration() -> ResolvedConfiguration:
    configuration = campaign_configuration()
    scientific = configuration.scientific.model_copy(update={"learning_seeds": (0, 1)})
    return configuration.model_copy(update={"scientific": scientific})


def build_campaign_artifact(tmp_path: Path) -> tuple[EvidenceArtifact, ResolvedConfiguration]:
    configuration = two_seed_configuration()
    outcome = CampaignOutcome.from_targets(run_dataset_study(campaign_devices(), configuration))
    path = tmp_path / "campaign.json"
    reuse_or_write_artifact(
        path,
        outcome,
        Provenance(
            configuration.scientific,
            configuration.splitting,
            configuration.scientific_digest,
            INPUT,
            IMPLEMENTATION,
            ArtifactKind.DATASET_CAMPAIGN,
            DatasetName.N_BAIOT,
            ExecutionDevice.CPU,
            None,
            StudyStratum.N_BAIOT,
        ),
        ReplacementPolicy.REPLACE_REQUESTED,
    )
    artifact = read_verified_artifact(
        path,
        Provenance(
            configuration.scientific,
            configuration.splitting,
            configuration.scientific_digest,
            INPUT,
            IMPLEMENTATION,
            ArtifactKind.DATASET_CAMPAIGN,
            DatasetName.N_BAIOT,
            ExecutionDevice.CPU,
            None,
            StudyStratum.N_BAIOT,
        ),
    )
    return artifact, configuration


def run_single_seed_campaign(
    clients: tuple[DatasetClientData, ...],
    configuration: ResolvedConfiguration,
    seed: LearningSeed | None = None,
) -> tuple[TargetPerformance, ...]:
    selected = configuration.scientific.seed if seed is None else seed
    scientific = configuration.scientific.model_copy(
        update={"seed": selected, "learning_seeds": (selected,)}
    )
    return run_dataset_study(clients, configuration.model_copy(update={"scientific": scientific}))


def campaign_results() -> tuple[TargetPerformance, ...]:
    configuration = two_seed_configuration()
    return run_dataset_study(campaign_devices(), configuration)


def with_attributes(result: TargetPerformance, **updates: object) -> TargetPerformance:
    clone = copy.copy(result)
    for name, value in updates.items():
        object.__setattr__(clone, name, value)
    return clone


def target_provenance(
    target: DatasetClientId, seed: LearningSeed
) -> tuple[ScientificDigest, tuple[ModelSeedProvenance, ...], ModelSeedProvenance]:
    peer_seeds = tuple(
        ModelSeedProvenance(
            DatasetClientId(target.dataset, ClientName(f"peer-{peer_index}")),
            seed + peer_index + 1,
            SeedDomain.PEER_MODEL,
        )
        for peer_index in range(4)
    )
    local_seed = ModelSeedProvenance(
        target,
        seed + 100,
        SeedDomain.LOCAL_MODEL,
    )
    return "a" * 64, peer_seeds, local_seed


def _staged_corpus_root(source: Path, corpus_directory: Path) -> Path:
    corpus_root = source.parent / f"staged-{source.stem}"
    staged = corpus_root / corpus_directory
    staged.mkdir(parents=True, exist_ok=True)
    if source.exists():
        shutil.copyfile(source, staged / source.name)
    return corpus_root


def load_gotham_capture(source: Path, caps: ReservoirCaps | None = None) -> DatasetClientData:
    (client,) = load_gotham_clients(
        _staged_corpus_root(source, Path(CorpusPath.GOTHAM_PROCESSED)), caps
    )
    return client


def load_ctu13_scenario(source: Path, caps: ReservoirCaps | None = None) -> DatasetClientData:
    (client,) = load_ctu13_clients(
        _staged_corpus_root(source, Path(CorpusPath.CTU_13_EXTRACTED)), caps
    )
    return client


def main(arguments: Sequence[str]) -> ExitStatus:
    original = sys.argv
    sys.argv = ["pace", *arguments]
    try:
        return cli.main()
    finally:
        sys.argv = original


def minimal_campaign_metrics(
    targets: tuple[TargetMetricSummary, ...] | None = None,
) -> CampaignMetricSummary:
    target = TargetMetricSummary(
        target=DatasetClientId(DatasetName.N_BAIOT, ClientName("target")),
        method_detection_rate=0.8,
        local_detection_rate=0.7,
        max_fusion_detection_rate=0.6,
        bonferroni_detection_rate=0.7,
        detection_difference=0.1,
        max_fusion_detection_difference=-0.1,
        method_max_fusion_difference=0.2,
        realized_false_alert_rate=0.03,
        local_false_alert_rate=0.02,
        max_fusion_false_alert_rate=0.04,
        bonferroni_false_alert_rate=0.045,
        matched_fpr_difference=0.15,
    )
    return CampaignMetricSummary(
        target_count=len(targets or (target,)),
        targets=targets or (target,),
        macro_detection_rate=0.8,
        local_macro_detection_rate=0.7,
        max_fusion_macro_detection_rate=0.6,
        bonferroni_macro_detection_rate=0.7,
        macro_detection_difference=0.1,
        method_local_confidence_interval=RateConfidenceInterval(
            lower_bound=0.05, upper_bound=0.15, confidence_level=0.95
        ),
        max_fusion_macro_detection_difference=-0.1,
        method_max_fusion_confidence_interval=RateConfidenceInterval(
            lower_bound=0.1, upper_bound=0.3, confidence_level=0.95
        ),
        matched_fpr_target_count=1,
        matched_fpr_macro_difference=0.15,
        matched_fpr_confidence_interval=RateConfidenceInterval(
            lower_bound=0.05, upper_bound=0.25, confidence_level=0.95
        ),
        mean_method_max_fusion_difference=0.2,
        worst_method_max_fusion_difference=0.2,
        improved_target_share=1.0,
        worst_detection_difference=0.1,
        tenth_percentile_detection_difference=0.1,
        catastrophic_cell_count=0,
        harmed_target_count=0,
        mean_false_alert_rate=0.03,
        median_false_alert_rate=0.03,
        ninetieth_percentile_false_alert_rate=0.03,
        worst_false_alert_rate=0.03,
        exceedance_share=0.0,
        local_mean_false_alert_rate=0.02,
        max_fusion_mean_false_alert_rate=0.04,
        max_fusion_worst_false_alert_rate=0.04,
        bonferroni_mean_false_alert_rate=0.045,
        bonferroni_worst_false_alert_rate=0.045,
    )


def protocol_matrix(
    budgets: tuple[LocalTrainingRowCount, ...],
    alphas: tuple[AlphaLevel, ...] = (AlphaLevel.ALPHA_001, AlphaLevel.ALPHA_005),
    analyses: tuple[AnalysisSpec, ...] | None = None,
) -> ProtocolMatrixSettings:
    return ProtocolMatrixSettings(
        alpha_levels=alphas,
        local_training_row_counts=budgets,
        analyses=analyses
        or (AnalysisSpec(kind=BudgetAnalysisKind.UNIVERSAL_PRIMARY, budgets=budgets),),
        power=PowerSettings(
            minimum_meaningful_effect=0.02,
            difference_sds=(0.05, 0.1),
            significance_level=0.05,
            power=0.8,
        ),
    )
