from dataclasses import dataclass, replace
from pathlib import Path
from unittest.mock import Mock

import pytest

import pace.reporting.service as reporting_service
from pace.config import ResolvedConfiguration, load_configuration
from pace.datasets.adapters.nbaiot import nbaiot_source_files
from pace.datasets.registry import dataset_source_files
from pace.experiment.artifacts import (
    CampaignOutcome,
    Provenance,
    campaign_artifact_path,
    digest_files,
    digest_implementation,
    reuse_or_write_artifact,
)
from pace.experiment.preflight import matrix_variants
from pace.experiment.runner import run_dataset_study
from pace.reporting.acceptance import summarize_acceptance
from pace.reporting.figures import write_figures
from pace.reporting.metrics import summarize_campaign
from pace.reporting.records import (
    RecordStore,
    ReportInputs,
    StratumBudgetEvidence,
    StratumTraining,
    VariantMetrics,
    records_of,
)
from pace.reporting.service import (
    DatasetSourceProvenance,
    MatrixReportResult,
    PersistedCampaignReport,
    ReportProvenance,
    build_campaign_report,
    report_acceptance,
    report_dataset_campaign,
    report_registered_dataset_matrix,
    validate_campaign_report,
    write_table,
)
from pace.reporting.tables import build_tables
from pace.types import (
    AlphaLevel,
    ArtifactConflictError,
    ArtifactKind,
    ArtifactPath,
    ArtifactReuseState,
    ArtifactValidationError,
    AttackType,
    AttackTypeName,
    AttackTypePerformance,
    BudgetAnalysis,
    BudgetAnalysisKind,
    BudgetPlanSummary,
    BudgetSweepPerformance,
    CampaignReportResult,
    ClientName,
    ConfigurationSection,
    DatasetClientId,
    DatasetName,
    DatasetValidationError,
    DisplayText,
    FigureName,
    LearningSeed,
    LeaveOneOutScope,
    ReplacementPolicy,
    SatisfiedAnswer,
    StudyArm,
    StudyStratum,
    TableName,
    TargetPerformance,
    TargetTrainingRows,
    default_study_stratum,
)
from tests.support import (
    REPOSITORY_ROOT,
    campaign_configuration,
    campaign_devices,
    minimal_campaign_metrics,
    protocol_matrix,
    run_single_seed_campaign,
    target_provenance,
    two_seed_configuration,
)


def _configuration(tmp_path: Path) -> ResolvedConfiguration:
    project_root = REPOSITORY_ROOT
    base = load_configuration(project_root / "configs" / "default.yaml", project_root)
    return base.model_copy(
        update={
            "runtime": base.runtime.model_copy(
                update={
                    "raw_data_root": tmp_path / "raw",
                    "output_root": tmp_path / "outputs",
                }
            )
        }
    )


def _target_performance(
    seed: LearningSeed,
    dataset: DatasetName = DatasetName.N_BAIOT,
    client_name: str = "target",
) -> TargetPerformance:
    target = DatasetClientId(dataset, ClientName(client_name))
    split_digest, peer_model_seeds, local_model_seed = target_provenance(target, seed)
    attack_type = AttackType(AttackTypeName("GAFGYT/COMBO"))
    return TargetPerformance(
        target=target,
        seed=seed,
        split_digest=split_digest,
        peer_model_seeds=peer_model_seeds,
        local_model_seed=local_model_seed,
        benign_test_rows=20,
        realized_false_alert_rate=0.04,
        attack_types=(
            AttackTypePerformance(
                attack_type=attack_type,
                test_rows=20,
                detection_rate=0.8,
                local_detection_rate=0.7,
                max_fusion_detection_rate=0.6,
                peer_union_detection_rate=0.3,
                bonferroni_detection_rate=0.75,
            ),
        ),
        local_false_alert_rate=0.02,
        max_fusion_false_alert_rate=0.03,
        peer_union_false_alert_rate=0.01,
        bonferroni_false_alert_rate=0.035,
        peer_count=4,
        budget_sweep=(
            BudgetSweepPerformance(multiplier=0.4, detection_rate=0.4, false_alert_rate=0.0),
            BudgetSweepPerformance(multiplier=0.75, detection_rate=0.6, false_alert_rate=0.02),
            BudgetSweepPerformance(multiplier=1.0, detection_rate=0.8, false_alert_rate=0.04),
            BudgetSweepPerformance(multiplier=1.4, detection_rate=0.9, false_alert_rate=0.1),
        ),
        study_stratum=StudyStratum[dataset.name],
    )


def _write_verified_campaign(
    configuration: ResolvedConfiguration,
    dataset: DatasetName = DatasetName.N_BAIOT,
    client_name: str = "target",
) -> tuple[Path, ...]:
    raw_root = configuration.runtime.raw_data_root
    dataset_directory = "N-BaIoT" if dataset is DatasetName.N_BAIOT else dataset.name
    source_file = raw_root / dataset_directory / client_name / "benign_traffic.csv"
    source_file.parent.mkdir(parents=True)
    source_file.write_text("stable test source", encoding="utf-8")
    source_files = (
        (source_file,) if dataset is not DatasetName.N_BAIOT else nbaiot_source_files(raw_root)
    )
    source_digest = digest_files(source_files, raw_root)
    implementation_digest = digest_implementation()
    reuse_or_write_artifact(
        campaign_artifact_path(
            configuration.runtime.output_root,
            dataset,
            configuration.scientific_digest,
        ).path,
        CampaignOutcome.from_targets(
            tuple(
                _target_performance(seed, dataset, client_name)
                for seed in configuration.scientific.learning_seeds
            )
        ),
        Provenance(
            configuration.scientific,
            configuration.splitting,
            configuration.scientific_digest,
            source_digest,
            implementation_digest,
            ArtifactKind.DATASET_CAMPAIGN,
            dataset,
        ),
        ReplacementPolicy.PRESERVE_VALID,
    )
    return source_files


def report_registered_dataset_campaign(
    dataset: DatasetName,
    configuration: ResolvedConfiguration,
    replacement_policy: ReplacementPolicy = ReplacementPolicy.PRESERVE_VALID,
) -> CampaignReportResult:
    stratum = default_study_stratum(dataset)
    sources = dataset_source_files(configuration.runtime.raw_data_root, dataset, stratum)
    return report_dataset_campaign(stratum, sources, configuration, replacement_policy)


def test_verified_campaign_report_is_persisted_and_reused_without_rewrite(
    tmp_path: Path,
) -> None:
    configuration = _configuration(tmp_path)
    _write_verified_campaign(configuration)

    first = report_registered_dataset_campaign(DatasetName.N_BAIOT, configuration)
    report_digest = first.artifact_path.path.read_bytes()
    report_mtime = first.artifact_path.path.stat().st_mtime_ns
    assert first.reuse_state is ArtifactReuseState.CREATED
    assert first.artifact_path.path.is_relative_to(tmp_path / "outputs" / "reports")
    persisted = PersistedCampaignReport.model_validate_json(report_digest, strict=True)
    assert persisted.reporting_configuration == configuration.reporting
    assert persisted.reporting_digest == configuration.reporting_digest
    assert tuple(source.dataset for source in persisted.sources) == (DatasetName.N_BAIOT,)
    assert persisted.sources[0].input_digest == persisted.input_digest
    assert persisted.sources[0].artifact_digest == persisted.source_artifact_digest

    second = report_registered_dataset_campaign(DatasetName.N_BAIOT, configuration)

    assert second.reuse_state is ArtifactReuseState.REUSED
    assert second.artifact_path.path.read_bytes() == report_digest
    assert second.artifact_path.path.stat().st_mtime_ns == report_mtime
    assert abs(second.metrics.macro_detection_rate - 0.8) < 1e-12


def test_report_overwrite_replaces_only_the_requested_report(tmp_path: Path) -> None:
    configuration = _configuration(tmp_path)
    _write_verified_campaign(configuration)
    first = report_registered_dataset_campaign(DatasetName.N_BAIOT, configuration)
    report_contents = first.artifact_path.path.read_bytes()
    campaign_path = campaign_artifact_path(
        configuration.runtime.output_root,
        DatasetName.N_BAIOT,
        configuration.scientific_digest,
    ).path
    campaign_contents = campaign_path.read_bytes()

    replaced = report_registered_dataset_campaign(
        DatasetName.N_BAIOT, configuration, replacement_policy=ReplacementPolicy.REPLACE_REQUESTED
    )

    assert replaced.reuse_state is ArtifactReuseState.CREATED
    assert replaced.artifact_path.path.read_bytes() == report_contents
    assert campaign_path.read_bytes() == campaign_contents


def test_report_rejects_source_changes_instead_of_reusing_old_evidence(
    tmp_path: Path,
) -> None:
    configuration = _configuration(tmp_path)
    _write_verified_campaign(configuration)
    report_registered_dataset_campaign(DatasetName.N_BAIOT, configuration)
    source_file = next((configuration.runtime.raw_data_root / "N-BaIoT").rglob("*.csv"))
    source_file.write_text("changed test source", encoding="utf-8")

    with pytest.raises(ArtifactConflictError):
        report_registered_dataset_campaign(DatasetName.N_BAIOT, configuration)


def test_reporting_configuration_changes_get_separate_reusable_artifacts(
    tmp_path: Path,
) -> None:
    configuration = _configuration(tmp_path)
    _write_verified_campaign(configuration)
    first = report_registered_dataset_campaign(DatasetName.N_BAIOT, configuration)
    first_contents = first.artifact_path.path.read_bytes()
    changed_source = (
        (REPOSITORY_ROOT / "configs" / "default.yaml")
        .read_text(encoding="utf-8")
        .replace("bootstrap_seed: 314159", "bootstrap_seed: 271828")
    )
    changed_configuration_path = tmp_path / "alternate.yaml"
    changed_configuration_path.write_text(changed_source, encoding="utf-8")
    changed_configuration = load_configuration(changed_configuration_path, tmp_path).model_copy(
        update={"runtime": configuration.runtime}
    )

    second = report_registered_dataset_campaign(DatasetName.N_BAIOT, changed_configuration)
    repeated = report_registered_dataset_campaign(DatasetName.N_BAIOT, changed_configuration)

    assert second.artifact_path.path != first.artifact_path.path
    assert second.reuse_state is ArtifactReuseState.CREATED
    assert repeated.reuse_state is ArtifactReuseState.REUSED
    assert first.artifact_path.path.read_bytes() == first_contents


def test_corrupt_existing_report_is_rejected(tmp_path: Path) -> None:
    configuration = _configuration(tmp_path)
    _write_verified_campaign(configuration)
    result = report_registered_dataset_campaign(DatasetName.N_BAIOT, configuration)
    result.artifact_path.path.write_text("not-json", encoding="utf-8")

    with pytest.raises(ArtifactValidationError, match="Existing campaign report is invalid"):
        report_registered_dataset_campaign(DatasetName.N_BAIOT, configuration)


def test_registered_report_wrappers_reuse_registry_sources_and_build_matrix(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    configuration = _configuration(tmp_path)
    sources = _write_verified_campaign(configuration)
    result = report_registered_dataset_campaign(DatasetName.N_BAIOT, configuration)
    calls: list[DatasetName | StudyStratum] = []

    def fake_sources(
        _data_root: Path,
        _dataset: DatasetName,
        _stratum: StudyStratum | None = None,
    ) -> tuple[Path, ...]:
        return sources

    def fake_report(
        campaign: DatasetName | StudyStratum,
        _source_files: tuple[Path, ...],
        _configuration: ResolvedConfiguration,
        _replacement_policy: ReplacementPolicy = ReplacementPolicy.PRESERVE_VALID,
    ) -> CampaignReportResult:
        calls.append(campaign)
        return result

    analysis = BudgetAnalysis(
        BudgetAnalysisKind.UNIVERSAL_PRIMARY, (100, 400, 20000), (ClientName("target"),)
    )
    monkeypatch.setattr(reporting_service, "dataset_source_files", fake_sources)
    monkeypatch.setattr(reporting_service, "report_dataset_campaign", fake_report)
    monkeypatch.setattr(
        reporting_service,
        "read_budget_plan_summary",
        Mock(return_value=BudgetPlanSummary((analysis,), 1)),
    )

    registered_matrix = report_registered_dataset_matrix(DatasetName.N_BAIOT, configuration)

    assert tuple(entry.result for entry in registered_matrix) == (result,) * 6
    assert {entry.variant.kind for entry in registered_matrix} == {
        BudgetAnalysisKind.UNIVERSAL_PRIMARY
    }
    assert calls == [StudyStratum.N_BAIOT] * 6


def test_acceptance_collects_every_stratum_before_requiring_the_comparator_arm(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    base = _configuration(tmp_path)
    configuration = base.model_copy(update={"protocol_matrix": protocol_matrix((100, 400))})
    seen: list[StudyStratum] = []

    def analysis_of(stratum: StudyStratum) -> BudgetAnalysis:
        return BudgetAnalysis(
            BudgetAnalysisKind.UNIVERSAL_PRIMARY, (100, 400), (ClientName(stratum.name),)
        )

    def fake_matrix(
        dataset: DatasetName,
        _configuration: ResolvedConfiguration,
        stratum: StudyStratum | None,
        _replacement_policy: ReplacementPolicy,
    ) -> tuple[MatrixReportResult, ...]:
        assert stratum is not None
        seen.append(stratum)
        target = replace(
            minimal_campaign_metrics().targets[0],
            target=DatasetClientId(dataset, ClientName(stratum.name)),
        )
        report = CampaignReportResult(
            ArtifactPath(tmp_path / "report.json"),
            minimal_campaign_metrics((target,)),
            ArtifactReuseState.REUSED,
        )
        alphas = configuration.protocol_matrix.alpha_levels
        return tuple(
            MatrixReportResult(variant, report)
            for variant in matrix_variants((analysis_of(stratum),), alphas)
        )

    def fake_summary(
        _configuration: ResolvedConfiguration, stratum: StudyStratum, _digest: str
    ) -> BudgetPlanSummary:
        return BudgetPlanSummary((analysis_of(stratum),), 1)

    monkeypatch.setattr(reporting_service, "report_registered_dataset_matrix", fake_matrix)
    monkeypatch.setattr(
        reporting_service, "dataset_source_files", Mock(return_value=(tmp_path / "source.csv",))
    )
    monkeypatch.setattr(reporting_service, "digest_source_files", Mock(return_value="5" * 64))
    monkeypatch.setattr(reporting_service, "read_budget_plan_summary", fake_summary)

    with pytest.raises(DatasetValidationError, match="MEAN_ENSEMBLE_LOCAL_BRANCH"):
        report_acceptance(configuration)

    assert len(seen) == len(set(seen)) == 7


@dataclass(frozen=True, slots=True)
class ReportWorld:
    configuration: ResolvedConfiguration
    source: Path
    outcome: CampaignOutcome


@pytest.fixture(scope="module")
def world(tmp_path_factory: pytest.TempPathFactory) -> ReportWorld:
    root = tmp_path_factory.mktemp("report-world")
    base = two_seed_configuration()
    configuration = base.model_copy(
        update={
            "runtime": base.runtime.model_copy(
                update={"raw_data_root": root / "raw", "output_root": root / "outputs"}
            )
        }
    )
    source = root / "raw" / "N-BaIoT" / "device" / "benign_traffic.csv"
    source.parent.mkdir(parents=True)
    source.write_text("stable source", encoding="utf-8")
    outcome = CampaignOutcome.from_targets(run_dataset_study(campaign_devices(), configuration))
    reuse_or_write_artifact(
        campaign_artifact_path(
            configuration.runtime.output_root, StudyStratum.N_BAIOT, configuration.scientific_digest
        ).path,
        outcome,
        Provenance(
            configuration.scientific,
            configuration.splitting,
            configuration.scientific_digest,
            digest_files((source,), configuration.runtime.raw_data_root),
            digest_implementation(),
            ArtifactKind.DATASET_CAMPAIGN,
            DatasetName.N_BAIOT,
            configuration.runtime.execution_device,
            None,
            StudyStratum.N_BAIOT,
        ),
        ReplacementPolicy.REPLACE_REQUESTED,
    )
    return ReportWorld(configuration, source, outcome)


def test_corrupt_and_stale_reports_are_rejected(world: ReportWorld) -> None:
    configuration = world.configuration
    created = report_dataset_campaign(
        StudyStratum.N_BAIOT, (world.source,), configuration, ReplacementPolicy.REPLACE_REQUESTED
    )
    report_file = created.artifact_path.path
    original = report_file.read_bytes()
    report_file.write_bytes(b"{broken")
    with pytest.raises(ArtifactValidationError, match="Existing campaign report is invalid"):
        report_dataset_campaign(StudyStratum.N_BAIOT, (world.source,), configuration)
    report_file.write_bytes(original.replace(b'"schema_version":', b'"schema_version":0,"x":'))
    with pytest.raises(ArtifactValidationError):
        report_dataset_campaign(StudyStratum.N_BAIOT, (world.source,), configuration)
    report_file.write_bytes(original)
    assert (
        report_dataset_campaign(StudyStratum.N_BAIOT, (world.source,), configuration).reuse_state
        is ArtifactReuseState.REUSED
    )


def _report(
    world: ReportWorld,
) -> tuple[PersistedCampaignReport, tuple[DatasetSourceProvenance, ...]]:
    result = report_dataset_campaign(
        StudyStratum.N_BAIOT,
        (world.source,),
        world.configuration,
        ReplacementPolicy.REPLACE_REQUESTED,
    )
    sources = (
        DatasetSourceProvenance(
            dataset=DatasetName.N_BAIOT,
            study_stratum=StudyStratum.N_BAIOT,
            input_digest="3" * 64,
            artifact_digest="4" * 64,
        ),
    )
    report = build_campaign_report(
        ReportProvenance(
            world.configuration.scientific_digest,
            world.configuration.reporting,
            world.configuration.reporting_digest,
            "3" * 64,
            "5" * 64,
            "4" * 64,
            sources,
        ),
        result.metrics,
    )
    return report, sources


def test_report_provenance_is_validated(world: ReportWorld) -> None:
    configuration = world.configuration
    report, sources = _report(world)

    def validate(candidate: PersistedCampaignReport, implementation: str = "5" * 64) -> None:
        validate_campaign_report(
            candidate,
            ReportProvenance(
                configuration.scientific_digest,
                configuration.reporting,
                configuration.reporting_digest,
                "3" * 64,
                implementation,
                "4" * 64,
                sources,
            ),
        )

    validate(report)
    with pytest.raises(ArtifactConflictError):
        validate(report, "6" * 64)
    variant = report.model_copy(update={"content_digest": ("7" * 64)})
    with pytest.raises(ArtifactValidationError, match="content hash"):
        validate(variant)
    variant = report.model_copy(update={"schema_version": type(report.schema_version)(1)})
    with pytest.raises(ArtifactValidationError, match="schema is not supported"):
        validate(variant)


def test_report_sources_must_be_canonical_and_match_their_digests(world: ReportWorld) -> None:
    report, sources = _report(world)
    fields: dict[str, object] = {
        name: getattr(report, name) for name in PersistedCampaignReport.model_fields
    }
    with pytest.raises(ValueError, match="exactly one source campaign"):
        PersistedCampaignReport.model_validate({**fields, "sources": ()})
    with pytest.raises(ValueError, match="exactly one source campaign"):
        PersistedCampaignReport.model_validate({**fields, "sources": (sources[0], sources[0])})
    with pytest.raises(ValueError, match="digests do not match"):
        PersistedCampaignReport.model_validate({**fields, "input_digest": ("8" * 64)})


def test_inactive_provenance_reports_evidence_despite_implementation_drift(
    world: ReportWorld, monkeypatch: pytest.MonkeyPatch
) -> None:
    configuration = world.configuration
    created = report_dataset_campaign(
        StudyStratum.N_BAIOT, (world.source,), configuration, ReplacementPolicy.REPLACE_REQUESTED
    )
    monkeypatch.setattr(reporting_service, "digest_implementation", Mock(return_value="7" * 64))

    with pytest.raises(ArtifactConflictError):
        report_dataset_campaign(StudyStratum.N_BAIOT, (world.source,), configuration)

    relaxed = configuration.model_copy(
        update={"runtime": configuration.runtime.model_copy(update={"provenance_active": False})}
    )
    reused = report_dataset_campaign(StudyStratum.N_BAIOT, (world.source,), relaxed)

    assert reused.metrics == created.metrics
    assert reused.reuse_state is ArtifactReuseState.REUSED


ROWS = 32


def _arms_configuration() -> ResolvedConfiguration:
    base = campaign_configuration()
    scientific = base.scientific.model_copy(update={"study_arms": tuple(StudyArm)})
    return base.model_copy(update={"scientific": scientific})


def _outcomes(configuration: ResolvedConfiguration) -> tuple[TargetPerformance, ...]:
    return tuple(
        result
        for seed in (1, 2)
        for result in run_single_seed_campaign(campaign_devices(), configuration, seed)
    )


@pytest.fixture(scope="module")
def inputs() -> ReportInputs:
    configuration = _arms_configuration()
    alpha = configuration.scientific.alpha
    outcomes = _outcomes(configuration)
    stratum = StudyStratum.N_BAIOT
    names = tuple(sorted({result.target.name for result in outcomes}))
    analysis = BudgetAnalysis(BudgetAnalysisKind.UNIVERSAL_PRIMARY, (ROWS,), names)
    metrics = summarize_campaign(outcomes, alpha, configuration.reporting)
    evidence = StratumBudgetEvidence(
        stratum,
        (analysis,),
        len(names),
        tuple(
            VariantMetrics(variant, metrics) for variant in matrix_variants((analysis,), (alpha,))
        ),
    )
    return ReportInputs(
        configuration,
        RecordStore(records_of(outcomes, alpha, ROWS)),
        summarize_acceptance((evidence,), configuration.reporting),
        (evidence,),
        (StratumTraining(stratum, tuple(TargetTrainingRows(name, 500, 3) for name in names)),),
    )


def test_every_table_is_built_once_with_aligned_rows(inputs: ReportInputs) -> None:
    tables = build_tables(inputs)
    assert tuple(table.name for table in tables) == tuple(TableName)
    for table in tables:
        assert all(len(row) == len(table.columns) for row in table.rows), table.name
    populated = {table.name: len(table.rows) for table in tables}
    assert populated[TableName.TARGETS] == 5
    assert populated[TableName.ACCEPTANCE] == 10
    assert populated[TableName.CENSUS] == 5
    assert populated[TableName.SEEDS] == 2
    by_name = {table.name: table for table in tables}
    answers = {row[-1] for row in by_name[TableName.HEADLINE].rows}
    assert answers <= {DisplayText(SatisfiedAnswer.YES), DisplayText(SatisfiedAnswer.NO)}
    scopes = {row[2] for row in by_name[TableName.SENSITIVITY].rows}
    assert scopes == {LeaveOneOutScope.TARGET.column_label}
    prefixes = {row[0].split(".")[0] for row in by_name[TableName.CONFIGURATION].rows}
    assert prefixes == {str(section) for section in ConfigurationSection}


def test_tables_are_exported_as_csv_and_latex(inputs: ReportInputs, tmp_path: Path) -> None:
    table = build_tables(inputs)[0]
    csv_path, latex_path = write_table(table, tmp_path)
    header = csv_path.read_text(encoding="utf-8").splitlines()[0].split(",")
    assert header == [f"{column}" for column in table.columns]
    latex = latex_path.read_text(encoding="utf-8")
    assert latex.startswith("\\begin{tabular}")
    assert "\\toprule" in latex
    assert "\\_" in latex


def test_every_figure_is_written_as_vector_and_raster(inputs: ReportInputs, tmp_path: Path) -> None:
    paths = write_figures(inputs, tuple(FigureName), tmp_path)
    assert len(paths) == 2 * len(FigureName)
    assert all(path.stat().st_size > 0 for path in paths)
    assert paths[0].read_bytes().startswith(b"%PDF")


def test_results_are_written_under_the_results_root(
    inputs: ReportInputs, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def collect(_configuration: ResolvedConfiguration, _policy: ReplacementPolicy) -> ReportInputs:
        return inputs

    monkeypatch.setattr(reporting_service, "collect_inputs", collect)
    configuration = inputs.configuration.model_copy(
        update={
            "runtime": inputs.configuration.runtime.model_copy(
                update={"output_root": tmp_path / "outputs"}
            )
        }
    )
    files = reporting_service.write_results(configuration, ReplacementPolicy.PRESERVE_VALID)
    assert len(files.tables) == 2 * len(TableName)
    assert len(files.figures) == 2 * len(FigureName)
    assert (tmp_path / "results" / "tables" / "headline.csv").is_file()
    assert (tmp_path / "results" / "figures" / "headline-gain.pdf").is_file()


def test_stratum_figures_use_the_stratum_level_subset_only(
    inputs: ReportInputs, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def collect(
        _stratum: StudyStratum, _configuration: ResolvedConfiguration, _policy: ReplacementPolicy
    ) -> reporting_service.CollectedStratum:
        return stratum_of(inputs)

    monkeypatch.setattr(reporting_service, "collect_stratum", collect)
    configuration = inputs.configuration.model_copy(
        update={
            "runtime": inputs.configuration.runtime.model_copy(
                update={"output_root": tmp_path / "outputs"}
            )
        }
    )
    written = reporting_service.write_stratum_figures(
        StudyStratum.N_BAIOT, configuration, ReplacementPolicy.PRESERVE_VALID
    )
    expected = sum(name.is_stratum_level for name in FigureName)
    assert len(written) == 2 * expected
    assert all("n_baiot" in path.parts for path in written)


def stratum_of(inputs: ReportInputs) -> reporting_service.CollectedStratum:
    return reporting_service.CollectedStratum(
        inputs.evidence[0], inputs.store.records, inputs.training[0]
    )


def test_alpha_levels_are_those_present_in_the_evidence(inputs: ReportInputs) -> None:
    assert inputs.alphas() == (AlphaLevel.ALPHA_005,)


def _matrix_of(inputs: ReportInputs, tmp_path: Path) -> tuple[MatrixReportResult, ...]:
    configuration = inputs.configuration
    alpha = configuration.scientific.alpha
    outcomes = _outcomes(configuration)
    analysis = inputs.evidence[0].analyses[0]
    metrics = summarize_campaign(outcomes, alpha, configuration.reporting)
    result = CampaignReportResult(
        ArtifactPath(tmp_path / "report.json"), metrics, ArtifactReuseState.REUSED, outcomes
    )
    return tuple(
        MatrixReportResult(variant, result) for variant in matrix_variants((analysis,), (alpha,))
    )


def _patch_collection(
    monkeypatch: pytest.MonkeyPatch, inputs: ReportInputs, tmp_path: Path
) -> None:
    analysis = inputs.evidence[0].analyses[0]
    summary = BudgetPlanSummary((analysis,), 5, inputs.training[0].targets)
    monkeypatch.setattr(
        reporting_service,
        "report_registered_dataset_matrix",
        Mock(return_value=_matrix_of(inputs, tmp_path)),
    )
    monkeypatch.setattr(reporting_service, "read_budget_plan_summary", Mock(return_value=summary))
    monkeypatch.setattr(reporting_service, "dataset_source_files", Mock(return_value=()))
    monkeypatch.setattr(reporting_service, "digest_source_files", Mock(return_value="1" * 64))


def test_collect_stratum_keeps_one_record_per_target_alpha_and_budget(
    inputs: ReportInputs, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_collection(monkeypatch, inputs, tmp_path)
    collected = reporting_service.collect_stratum(
        StudyStratum.N_BAIOT, inputs.configuration, ReplacementPolicy.PRESERVE_VALID
    )
    assert len(collected.records) == 5
    assert collected.training == inputs.training[0]
    assert collected.evidence.stratum is StudyStratum.N_BAIOT
    assert len(collected.evidence.variants) == len(_matrix_of(inputs, tmp_path))


def test_collect_inputs_pools_every_registered_stratum_and_scores_acceptance(
    inputs: ReportInputs, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_collection(monkeypatch, inputs, tmp_path)

    def strata(dataset: DatasetName) -> tuple[StudyStratum, ...]:
        return (StudyStratum.N_BAIOT,) if dataset is DatasetName.N_BAIOT else ()

    monkeypatch.setattr(reporting_service, "registered_strata", strata)
    pooled = reporting_service.collect_inputs(
        inputs.configuration, ReplacementPolicy.PRESERVE_VALID
    )
    assert len(pooled.store.records) == 5
    assert pooled.acceptance.cells
    assert pooled.alphas() == (AlphaLevel.ALPHA_005,)
