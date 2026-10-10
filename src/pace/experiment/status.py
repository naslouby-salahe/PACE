from pathlib import Path

import structlog

from pace.config import ResolvedConfiguration, configuration_for_protocol_variant
from pace.datasets.registry import (
    dataset_source_files,
    inspect_dataset_sources,
)
from pace.experiment.artifacts import (
    CampaignOutcome,
    Provenance,
    campaign_artifact_path,
    digest_implementation,
    digest_source_files,
    execution_log_path,
    read_verified_artifact,
)
from pace.experiment.logging_setup import configure_logging
from pace.experiment.preflight import matrix_variants, read_budget_plan_summary
from pace.types import (
    ArtifactConflictError,
    ArtifactKind,
    ArtifactValidationError,
    CampaignCompletionState,
    CampaignStatus,
    CommandName,
    DatasetInventoryReport,
    DatasetName,
    DatasetValidationError,
    LogEvent,
    ObservationCount,
    StudyStratum,
    default_study_stratum,
)


def inspect_dataset_campaign_status(
    dataset: DatasetName,
    source_files: tuple[Path, ...],
    configuration: ResolvedConfiguration,
    stratum: StudyStratum | None = None,
) -> CampaignStatus:
    selected_stratum = stratum or default_study_stratum(dataset)
    if selected_stratum.dataset is not dataset:
        raise DatasetValidationError("Study stratum does not belong to the requested dataset")
    configure_logging(
        execution_log_path(configuration.runtime.output_root, CommandName.STATUS),
        configuration.runtime.log_level,
    )
    destination = campaign_artifact_path(
        configuration.runtime.output_root,
        selected_stratum,
        configuration.scientific_digest,
    )

    def status(
        state: CampaignCompletionState, target_count: ObservationCount | None = None
    ) -> CampaignStatus:
        return _record_status(
            CampaignStatus(dataset, selected_stratum, destination, state, target_count)
        )

    if not destination.path.exists():
        return status(CampaignCompletionState.MISSING)
    try:
        artifact = read_verified_artifact(
            destination.path,
            Provenance(
                configuration.scientific,
                configuration.splitting,
                configuration.scientific_digest,
                digest_source_files(source_files, configuration.runtime.raw_data_root),
                digest_implementation(),
                ArtifactKind.DATASET_CAMPAIGN,
                dataset,
                configuration.runtime.execution_device,
                campaign_stratum=selected_stratum,
                provenance_active=configuration.runtime.provenance_active,
            ),
        )
    except (ArtifactConflictError, DatasetValidationError):
        return status(CampaignCompletionState.STALE)
    except ArtifactValidationError:
        return status(CampaignCompletionState.INVALID)
    if not isinstance(artifact.outcome, CampaignOutcome):
        return status(CampaignCompletionState.INVALID)
    return status(CampaignCompletionState.COMPLETE, artifact.outcome.target_count)


def inspect_registered_dataset_campaign_status(
    dataset: DatasetName,
    configuration: ResolvedConfiguration,
    stratum: StudyStratum | None = None,
) -> CampaignStatus:
    selected_stratum = stratum or default_study_stratum(dataset)
    source_files: tuple[Path, ...] = ()
    if campaign_artifact_path(
        configuration.runtime.output_root, selected_stratum, configuration.scientific_digest
    ).path.exists():
        try:
            source_files = dataset_source_files(
                configuration.runtime.raw_data_root, dataset, selected_stratum
            )
        except DatasetValidationError:
            source_files = ()
    return inspect_dataset_campaign_status(dataset, source_files, configuration, selected_stratum)


def inspect_matrix_status(
    stratum: StudyStratum, configuration: ResolvedConfiguration
) -> tuple[CampaignStatus, ...]:
    dataset = stratum.dataset
    root = configuration.runtime.raw_data_root
    source_files = dataset_source_files(root, dataset, stratum)
    try:
        summary = read_budget_plan_summary(
            configuration, stratum, digest_source_files(source_files, root)
        )
    except ArtifactValidationError:
        return (inspect_registered_dataset_campaign_status(dataset, configuration, stratum),)
    return tuple(
        inspect_dataset_campaign_status(
            dataset,
            source_files,
            configuration_for_protocol_variant(
                configuration, variant.alpha, variant.local_training_rows, variant.targets
            ),
            stratum,
        )
        for variant in matrix_variants(summary.analyses, configuration.protocol_matrix.alpha_levels)
    )


def _record_status(status: CampaignStatus) -> CampaignStatus:
    structlog.get_logger().bind(
        dataset=status.dataset.name,
        study_stratum=status.study_stratum.name,
        artifact_path=status.artifact_path.path.as_posix(),
        state=status.state.name,
        target_count=status.target_count,
    ).info(LogEvent.CAMPAIGN_STATUS)
    return status


def run_doctor(configuration: ResolvedConfiguration) -> DatasetInventoryReport:
    configure_logging(
        execution_log_path(configuration.runtime.output_root, CommandName.DOCTOR),
        configuration.runtime.log_level,
    )
    logger = structlog.get_logger().bind(
        command=CommandName.DOCTOR,
        configuration_digest=configuration.scientific_digest,
    )
    logger.info(LogEvent.DOCTOR_STARTED, data_root=configuration.runtime.raw_data_root.as_posix())
    report = inspect_dataset_sources(configuration.runtime.raw_data_root)
    for source in report.sources:
        logger.info(
            LogEvent.DATASET_INSPECTED,
            dataset=source.dataset.name,
            availability=source.availability.name,
            file_count=source.file_count,
            byte_count=source.byte_count,
        )
    logger.info(
        LogEvent.DOCTOR_COMPLETED,
        dataset_count=len(report.sources),
        available=report.is_complete,
    )
    return report
