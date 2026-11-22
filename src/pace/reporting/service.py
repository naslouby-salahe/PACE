import csv
from dataclasses import dataclass
from pathlib import Path

import structlog
from pydantic import ValidationError, model_validator
from structlog.stdlib import BoundLogger

from pace.config import ReportingSettings, ResolvedConfiguration, configuration_for_protocol_variant
from pace.datasets.registry import (
    dataset_source_files,
    registered_strata,
    resolve_study_stratum,
)
from pace.experiment.artifacts import (
    CampaignOutcome,
    Provenance,
    campaign_artifact_path,
    digest_implementation,
    digest_source_files,
    execution_log_path,
    read_verified_artifact,
    write_bytes_atomically,
)
from pace.experiment.logging_setup import configure_logging
from pace.experiment.preflight import MatrixVariant, matrix_variants, read_budget_plan_summary
from pace.reporting.acceptance import summarize_acceptance
from pace.reporting.figures import write_figures
from pace.reporting.metrics import summarize_campaign
from pace.reporting.records import (
    RecordStore,
    ReportInputs,
    ResultTable,
    StratumBudgetEvidence,
    StratumTraining,
    TargetRecord,
    VariantMetrics,
    VariantPerformances,
    records_of,
)
from pace.reporting.tables import build_tables
from pace.types import (
    CURRENT_ARTIFACT_SCHEMA,
    AcceptanceReport,
    AlphaLevel,
    ArtifactConflictError,
    ArtifactFileName,
    ArtifactKind,
    ArtifactPath,
    ArtifactPayload,
    ArtifactReuseState,
    ArtifactSchemaVersion,
    ArtifactSerializationMode,
    ArtifactValidationError,
    BudgetAnalysisKind,
    CampaignMetricSummary,
    CampaignReportResult,
    CommandName,
    CsvNewline,
    DatasetClientId,
    DatasetName,
    DatasetValidationError,
    DisplayText,
    FigureName,
    FrozenStrictModel,
    ImplementationScope,
    LatexAlignment,
    LocalTrainingRowCount,
    LogEvent,
    OutputDirectory,
    PopulationScope,
    ProvenanceActive,
    ReplacementPolicy,
    ScientificDigest,
    StudyStratum,
    TableFormat,
    TextEncoding,
    canonical_json,
    payload_digest,
    placeholder_digest,
)


class DatasetSourceProvenance(FrozenStrictModel, frozen=True):
    dataset: DatasetName
    study_stratum: StudyStratum
    input_digest: ScientificDigest
    artifact_digest: ScientificDigest


class PersistedCampaignReport(FrozenStrictModel, frozen=True):
    schema_version: ArtifactSchemaVersion
    configuration_digest: ScientificDigest
    reporting_configuration: ReportingSettings
    reporting_digest: ScientificDigest
    input_digest: ScientificDigest
    implementation_digest: ScientificDigest
    source_artifact_digest: ScientificDigest
    sources: tuple[DatasetSourceProvenance, ...]
    metrics: CampaignMetricSummary
    content_digest: ScientificDigest

    @model_validator(mode="after")
    def validate_sources(self) -> "PersistedCampaignReport":
        if len(self.sources) != 1:
            raise ValueError("Campaign reports require exactly one source campaign")
        (source,) = self.sources
        if source.study_stratum.dataset is not source.dataset:
            raise ValueError("Campaign report source stratum must belong to its dataset")
        input_digest, artifact_digest = digest_source_manifest(self.sources)
        if input_digest != self.input_digest or artifact_digest != self.source_artifact_digest:
            raise ValueError("Campaign report digests do not match their source manifest")
        return self


@dataclass(frozen=True, slots=True)
class ReportProvenance:
    configuration_digest: ScientificDigest
    reporting_configuration: ReportingSettings
    reporting_digest: ScientificDigest
    input_digest: ScientificDigest
    implementation_digest: ScientificDigest
    source_artifact_digest: ScientificDigest
    sources: tuple[DatasetSourceProvenance, ...]
    provenance_active: ProvenanceActive = True


def _content_digest(report: PersistedCampaignReport) -> ScientificDigest:
    return payload_digest(canonical_json(report, ArtifactSerializationMode.OMIT_DIGEST))


def digest_source_manifest(
    sources: tuple[DatasetSourceProvenance, ...],
) -> tuple[ScientificDigest, ScientificDigest]:
    (source,) = sources
    return source.input_digest, source.artifact_digest


def build_campaign_report(
    provenance: ReportProvenance, metrics: CampaignMetricSummary
) -> PersistedCampaignReport:
    def report(content_digest: ScientificDigest) -> PersistedCampaignReport:
        return PersistedCampaignReport(
            schema_version=CURRENT_ARTIFACT_SCHEMA,
            configuration_digest=provenance.configuration_digest,
            reporting_configuration=provenance.reporting_configuration,
            reporting_digest=provenance.reporting_digest,
            input_digest=provenance.input_digest,
            implementation_digest=provenance.implementation_digest,
            source_artifact_digest=provenance.source_artifact_digest,
            sources=provenance.sources,
            metrics=metrics,
            content_digest=content_digest,
        )

    return report(_content_digest(report(placeholder_digest())))


def validate_campaign_report(report: PersistedCampaignReport, provenance: ReportProvenance) -> None:
    if report.schema_version != CURRENT_ARTIFACT_SCHEMA:
        raise ArtifactValidationError("Campaign report schema is not supported")
    if provenance.provenance_active and (
        report.configuration_digest != provenance.configuration_digest
        or report.reporting_configuration != provenance.reporting_configuration
        or report.reporting_digest != provenance.reporting_digest
        or report.input_digest != provenance.input_digest
        or report.implementation_digest != provenance.implementation_digest
        or report.source_artifact_digest != provenance.source_artifact_digest
        or report.sources != provenance.sources
    ):
        raise ArtifactConflictError("Campaign report provenance does not match verified evidence")
    if report.content_digest != _content_digest(report):
        raise ArtifactValidationError("Campaign report content hash does not match its contents")


def _report_path(
    output_root: Path,
    stratum: StudyStratum,
    configuration_digest: ScientificDigest,
    reporting_digest: ScientificDigest,
) -> ArtifactPath:
    return ArtifactPath(
        output_root.resolve()
        / OutputDirectory.REPORTS
        / stratum.name.lower()
        / configuration_digest
        / reporting_digest
        / ArtifactFileName.REPORT
    )


def _read_report(path: Path, provenance: ReportProvenance) -> PersistedCampaignReport:
    try:
        report = PersistedCampaignReport.model_validate_json(path.read_bytes(), strict=True)
    except (OSError, ValidationError, ValueError) as error:
        raise ArtifactValidationError(f"Existing campaign report is invalid: {path}") from error
    validate_campaign_report(report, provenance)
    return report


def _write_report(
    path: Path, metrics: CampaignMetricSummary, provenance: ReportProvenance
) -> PersistedCampaignReport:
    report = build_campaign_report(provenance, metrics)

    def validate_written(path_to_validate: Path) -> None:
        if _read_report(path_to_validate, provenance) != report:
            raise ArtifactValidationError("Serialized campaign report changed during validation")

    write_bytes_atomically(path, ArtifactPayload(canonical_json(report) + b"\n"), validate_written)
    return report


@dataclass(frozen=True, slots=True)
class MatrixReportResult:
    variant: MatrixVariant
    result: CampaignReportResult


def report_registered_dataset_matrix(
    dataset: DatasetName,
    configuration: ResolvedConfiguration,
    stratum: StudyStratum | None = None,
    replacement_policy: ReplacementPolicy = ReplacementPolicy.PRESERVE_VALID,
) -> tuple[MatrixReportResult, ...]:
    selected_stratum = resolve_study_stratum(dataset, stratum)
    source_files = dataset_source_files(
        configuration.runtime.raw_data_root, dataset, selected_stratum
    )
    summary = read_budget_plan_summary(
        configuration,
        selected_stratum,
        digest_source_files(source_files, configuration.runtime.raw_data_root),
    )
    return tuple(
        MatrixReportResult(
            variant,
            report_dataset_campaign(
                selected_stratum,
                source_files,
                configuration_for_protocol_variant(
                    configuration, variant.alpha, variant.local_training_rows, variant.targets
                ),
                replacement_policy,
            ),
        )
        for variant in matrix_variants(summary.analyses, configuration.protocol_matrix.alpha_levels)
    )


def _verified_campaign(
    stratum: StudyStratum,
    source_files: tuple[Path, ...],
    configuration: ResolvedConfiguration,
    implementation_digest: ScientificDigest,
) -> tuple[ScientificDigest, ScientificDigest, CampaignOutcome]:
    dataset = stratum.dataset
    input_digest = digest_source_files(source_files, configuration.runtime.raw_data_root)
    artifact = read_verified_artifact(
        campaign_artifact_path(
            configuration.runtime.output_root, stratum, configuration.scientific_digest
        ).path,
        Provenance(
            configuration.scientific,
            configuration.splitting,
            configuration.scientific_digest,
            input_digest,
            implementation_digest,
            ArtifactKind.DATASET_CAMPAIGN,
            dataset,
            configuration.runtime.execution_device,
            campaign_stratum=stratum,
            provenance_active=configuration.runtime.provenance_active,
        ),
    )
    if not isinstance(artifact.outcome, CampaignOutcome):
        raise DatasetValidationError("Verified campaign artifact has the wrong result schema")
    if any(
        target.target.dataset is not dataset or target.study_stratum is not stratum
        for target in artifact.outcome.targets
    ):
        raise DatasetValidationError("Verified campaign contains targets from another stratum")
    return input_digest, artifact.content_digest, artifact.outcome


def _persist_report(
    stratum: StudyStratum,
    metrics: CampaignMetricSummary,
    configuration: ResolvedConfiguration,
    provenance: ReportProvenance,
    replacement_policy: ReplacementPolicy,
) -> tuple[ArtifactPath, PersistedCampaignReport, ArtifactReuseState]:
    destination = _report_path(
        configuration.runtime.output_root,
        stratum,
        configuration.scientific_digest,
        configuration.reporting_digest,
    )
    if destination.path.exists() and replacement_policy is ReplacementPolicy.PRESERVE_VALID:
        try:
            return (
                destination,
                _read_report(destination.path, provenance),
                ArtifactReuseState.REUSED,
            )
        except ArtifactConflictError:
            previous = PersistedCampaignReport.model_validate_json(
                destination.path.read_bytes(), strict=True
            )
            _read_report(
                destination.path,
                ReportProvenance(
                    configuration_digest=provenance.configuration_digest,
                    reporting_configuration=provenance.reporting_configuration,
                    reporting_digest=provenance.reporting_digest,
                    input_digest=provenance.input_digest,
                    implementation_digest=previous.implementation_digest,
                    source_artifact_digest=provenance.source_artifact_digest,
                    sources=provenance.sources,
                    provenance_active=provenance.provenance_active,
                ),
            )
            current = ArtifactPath(
                destination.path.parent / provenance.implementation_digest / destination.path.name
            )
            if current.path.exists():
                return current, _read_report(current.path, provenance), ArtifactReuseState.REUSED
            return (
                current,
                _write_report(current.path, metrics, provenance),
                ArtifactReuseState.CREATED,
            )
    return (
        destination,
        _write_report(destination.path, metrics, provenance),
        ArtifactReuseState.CREATED,
    )


def _report_logger(configuration: ResolvedConfiguration, stratum: StudyStratum) -> BoundLogger:
    configure_logging(
        execution_log_path(configuration.runtime.output_root, CommandName.REPORT),
        configuration.runtime.log_level,
    )
    return structlog.get_logger().bind(
        command=CommandName.REPORT,
        study_stratum=stratum.name,
        execution_device=configuration.runtime.execution_device.name,
        configuration_digest=configuration.scientific_digest,
        reporting_digest=configuration.reporting_digest,
    )


def _provenance(
    configuration: ResolvedConfiguration, sources: tuple[DatasetSourceProvenance, ...]
) -> ReportProvenance:
    input_digest, source_artifact_digest = digest_source_manifest(sources)
    return ReportProvenance(
        configuration.scientific_digest,
        configuration.reporting,
        configuration.reporting_digest,
        input_digest,
        digest_implementation(ImplementationScope.REPORT),
        source_artifact_digest,
        sources,
        configuration.runtime.provenance_active,
    )


def report_dataset_campaign(
    stratum: StudyStratum,
    source_files: tuple[Path, ...],
    configuration: ResolvedConfiguration,
    replacement_policy: ReplacementPolicy = ReplacementPolicy.PRESERVE_VALID,
) -> CampaignReportResult:
    logger = _report_logger(configuration, stratum)
    logger.info(LogEvent.REPORT_STARTED, source_file_count=len(source_files))
    implementation_digest = digest_implementation()
    input_digest, artifact_digest, outcome = _verified_campaign(
        stratum, source_files, configuration, implementation_digest
    )
    provenance = _provenance(
        configuration,
        (
            DatasetSourceProvenance(
                dataset=stratum.dataset,
                study_stratum=stratum,
                input_digest=input_digest,
                artifact_digest=artifact_digest,
            ),
        ),
    )
    destination, report, reuse_state = _persist_report(
        stratum,
        summarize_campaign(
            outcome.targets, configuration.scientific.alpha, configuration.reporting
        ),
        configuration,
        provenance,
        replacement_policy,
    )
    logger.info(
        LogEvent.REPORT_REUSED
        if reuse_state is ArtifactReuseState.REUSED
        else LogEvent.REPORT_COMPLETED,
        input_digest=provenance.input_digest,
        target_count=report.metrics.target_count,
        artifact_path=destination.path.as_posix(),
        content_digest=report.content_digest,
        state=reuse_state.name,
    )
    return CampaignReportResult(destination, report.metrics, reuse_state, outcome.targets)


logger = structlog.get_logger()


@dataclass(frozen=True, slots=True)
class CollectedStratum:
    evidence: StratumBudgetEvidence
    records: tuple[TargetRecord, ...]
    training: StratumTraining
    performances: tuple[VariantPerformances, ...] = ()


@dataclass(frozen=True, slots=True)
class ResultFiles:
    tables: tuple[Path, ...]
    figures: tuple[Path, ...]


def collect_stratum(
    stratum: StudyStratum,
    configuration: ResolvedConfiguration,
    replacement_policy: ReplacementPolicy,
) -> CollectedStratum:
    dataset = stratum.dataset
    root = configuration.runtime.raw_data_root
    matrix = report_registered_dataset_matrix(dataset, configuration, stratum, replacement_policy)
    summary = read_budget_plan_summary(
        configuration,
        stratum,
        digest_source_files(dataset_source_files(root, dataset, stratum), root),
    )
    records: dict[tuple[DatasetClientId, AlphaLevel, LocalTrainingRowCount], TargetRecord] = {}
    for entry in matrix:
        variant = entry.variant
        for record in records_of(entry.result.results, variant.alpha, variant.local_training_rows):
            records.setdefault((record.target, record.alpha, record.rows), record)
    return CollectedStratum(
        StratumBudgetEvidence(
            stratum,
            summary.analyses,
            summary.eligible_targets,
            tuple(VariantMetrics(item.variant, item.result.metrics) for item in matrix),
        ),
        tuple(records.values()),
        StratumTraining(stratum, summary.targets),
        tuple(
            VariantPerformances(
                stratum,
                entry.variant.alpha,
                entry.variant.local_training_rows,
                entry.result.results,
            )
            for entry in matrix
        ),
    )


def _inputs_of(
    collected: tuple[CollectedStratum, ...], configuration: ResolvedConfiguration
) -> ReportInputs:
    evidence = tuple(item.evidence for item in collected)
    inputs = ReportInputs(
        configuration,
        RecordStore(tuple(record for item in collected for record in item.records)),
        evidence,
        tuple(item.training for item in collected),
        tuple(performance for item in collected for performance in item.performances),
    )
    return ReportInputs(
        inputs.configuration,
        inputs.store,
        inputs.evidence,
        inputs.training,
        inputs.performances,
        inputs.sensitivity,
        summarize_acceptance(inputs),
    )


def collect_inputs(
    configuration: ResolvedConfiguration,
    replacement_policy: ReplacementPolicy,
    scope: PopulationScope = PopulationScope.PRIMARY,
) -> ReportInputs:
    return _inputs_of(
        tuple(
            collect_stratum(stratum, configuration, replacement_policy)
            for dataset in DatasetName
            for stratum in registered_strata(dataset)
            if scope.has_stratum(stratum)
        ),
        configuration,
    )


def report_acceptance(
    configuration: ResolvedConfiguration,
    replacement_policy: ReplacementPolicy = ReplacementPolicy.PRESERVE_VALID,
    scope: PopulationScope = PopulationScope.PRIMARY,
) -> AcceptanceReport:
    acceptance = collect_inputs(configuration, replacement_policy, scope).acceptance
    if acceptance is None:
        raise DatasetValidationError("The acceptance report could not be computed")
    expected = {
        (alpha, rows)
        for analysis in configuration.protocol_matrix.analyses
        if analysis.kind is BudgetAnalysisKind.UNIVERSAL_PRIMARY
        for rows in analysis.budgets
        for alpha in configuration.protocol_matrix.alpha_levels
    }
    actual = {(cell.alpha, cell.local_training_rows) for cell in acceptance.cells if cell.is_gating}
    if not expected or actual != expected:
        raise DatasetValidationError("Official acceptance is missing a configured gating cell")
    return acceptance


def results_root(configuration: ResolvedConfiguration) -> Path:
    return configuration.runtime.output_root.resolve().parent / OutputDirectory.RESULTS


def write_results(
    configuration: ResolvedConfiguration,
    replacement_policy: ReplacementPolicy,
    scope: PopulationScope = PopulationScope.PRIMARY,
) -> ResultFiles:
    inputs = collect_inputs(configuration, replacement_policy, scope)
    root = results_root(configuration)
    if scope is PopulationScope.CONFIRMATION:
        root = root / OutputDirectory.CONFIRMATION
    tables = tuple(path for table in build_tables(inputs) for path in write_table(table, root))
    report_figures = tuple(
        name for name in FigureName if name is not FigureName.CALIBRATION_SENSITIVITY
    )
    figures = write_figures(inputs, report_figures, root / OutputDirectory.FIGURES)
    logger.info(LogEvent.RESULTS_WRITTEN, table_files=len(tables), figure_files=len(figures))
    return ResultFiles(tables, figures)


def write_stratum_figures(
    stratum: StudyStratum,
    configuration: ResolvedConfiguration,
    replacement_policy: ReplacementPolicy,
) -> tuple[Path, ...]:
    inputs = _inputs_of(
        (collect_stratum(stratum, configuration, replacement_policy),), configuration
    )
    directory = configuration.runtime.output_root / OutputDirectory.FIGURES / stratum.name.lower()
    return write_figures(
        inputs, tuple(name for name in FigureName if name.is_stratum_level), directory
    )


def _escape(text: DisplayText) -> DisplayText:
    return DisplayText(text.replace("_", "\\_").replace("%", "\\%").replace("&", "\\&"))


def _latex(table: ResultTable) -> DisplayText:
    header = " & ".join(_escape(DisplayText(f"{column}")) for column in table.columns)
    body = "\n".join(" & ".join(_escape(cell) for cell in row) + " \\\\" for row in table.rows)
    alignment = LatexAlignment.LEFT * len(table.columns)
    return DisplayText(
        f"\\begin{{tabular}}{{{alignment}}}\n\\toprule\n{header} \\\\\n\\midrule\n{body}\n"
        "\\bottomrule\n\\end{tabular}\n"
    )


def write_table(table: ResultTable, results_root: Path) -> tuple[Path, Path]:
    directory = results_root / OutputDirectory.TABLES
    directory.mkdir(parents=True, exist_ok=True)
    csv_path = directory / f"{table.name}.{TableFormat.CSV}"
    with csv_path.open("w", newline=CsvNewline.UNIVERSAL, encoding=TextEncoding.UTF8) as handle:
        writer = csv.writer(handle)
        writer.writerow([f"{column}" for column in table.columns])
        writer.writerows(table.rows)
    latex_path = directory / f"{table.name}.{TableFormat.LATEX}"
    latex_path.write_text(_latex(table), encoding=TextEncoding.UTF8)
    logger.info(LogEvent.TABLE_WRITTEN, table=table.name, row_count=len(table.rows))
    return csv_path, latex_path
