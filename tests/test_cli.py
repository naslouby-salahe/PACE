import os
from collections.abc import Callable, Iterable, Iterator
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import pytest
import typer
from typer.core import TyperGroup

from pace import cli
from pace.config import CalibrationStudySettings
from pace.experiment.calibration_study import study_cells
from pace.experiment.preflight import MatrixVariant
from pace.experiment.runner import MatrixRunResult
from pace.reporting.service import ResultFiles
from pace.types import (
    AcceptanceCell,
    AcceptanceCriterion,
    AcceptanceReport,
    AlphaLevel,
    ArtifactPath,
    ArtifactReuseState,
    ArtifactVerification,
    BudgetAnalysis,
    BudgetAnalysisKind,
    CampaignCompletionState,
    CampaignStatus,
    CampaignWorkflowResult,
    ClientName,
    CommandTarget,
    ComputeMeasurement,
    ComputePhase,
    ComputeSubject,
    CriterionVerdict,
    DatasetClientId,
    DatasetLocationKind,
    DatasetName,
    ExitStatus,
    FprLevel,
    LevelGain,
    OperatingMethod,
    PaceError,
    PopulationScope,
    RateConfidenceInterval,
    RoleAudit,
    SeedRateDifference,
    StratumBudgetPlan,
    StudyStratum,
    TargetBudgetSupport,
    TargetEligibility,
    TargetEligibilityIssue,
    VerificationState,
)
from tests.support import REPOSITORY_ROOT, main


def _config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "configuration.yaml"
    path.write_text(
        (REPOSITORY_ROOT / "configs" / "default.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PACE_CONFIG", str(path))
    return path


def test_doctor_smoke_runs_the_synthetic_workflow_and_reuses_its_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _config(tmp_path, monkeypatch)
    assert main(["doctor", "--smoke"]) == ExitStatus.FAILURE
    created_output = capsys.readouterr().out
    assert "Synthetic evidence created:" in created_output
    assert (tmp_path / "outputs" / "smoke").is_dir()
    assert main(["doctor", "--smoke"]) == ExitStatus.FAILURE
    assert "Synthetic evidence reused:" in capsys.readouterr().out


def test_cli_reports_a_missing_configuration_as_a_usage_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PACE_CONFIG", "missing.yaml")
    with pytest.raises(SystemExit) as error:
        main(["doctor"])
    assert error.value.code == 2


def test_doctor_cli_checks_dataset_sources_without_writing_into_raw_data(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    raw_root = tmp_path / "data" / "raw"
    for source in DatasetName:
        location = raw_root / source.source_location
        if source.location_kind is DatasetLocationKind.FILE:
            location.parent.mkdir(parents=True, exist_ok=True)
            location.write_bytes(b"fixture")
        else:
            location.mkdir(parents=True, exist_ok=True)
            (location / "sample.csv").write_bytes(b"fixture")
    _config(tmp_path, monkeypatch)
    before = {
        path.relative_to(raw_root): path.read_bytes()
        for path in raw_root.rglob("*")
        if path.is_file()
    }

    assert main(["doctor"]) == ExitStatus.SUCCESS
    assert "Dataset sources ready" in capsys.readouterr().out
    assert main(["doctor"]) == ExitStatus.SUCCESS
    assert "Dataset sources ready" in capsys.readouterr().out
    after = {
        path.relative_to(raw_root): path.read_bytes()
        for path in raw_root.rglob("*")
        if path.is_file()
    }
    assert after == before
    log_lines = (
        (tmp_path / "outputs" / "logs" / "doctor.jsonl").read_text(encoding="utf-8").splitlines()
    )
    assert len(log_lines) == 22


def test_doctor_cli_returns_failure_for_missing_datasets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _config(tmp_path, monkeypatch)
    assert main(["doctor"]) == ExitStatus.FAILURE
    assert "Dataset sources incomplete" in capsys.readouterr().out


def _variants() -> tuple[MatrixVariant, ...]:
    return tuple(
        MatrixVariant(BudgetAnalysisKind.UNIVERSAL_PRIMARY, alpha, rows, (ClientName("target"),))
        for alpha in (AlphaLevel.ALPHA_001, AlphaLevel.ALPHA_005)
        for rows in (100, 400)
    )


def _cell_status(tmp_path: Path, state: CampaignCompletionState) -> CampaignStatus:
    return CampaignStatus(
        DatasetName.N_BAIOT,
        StudyStratum.N_BAIOT,
        ArtifactPath(tmp_path / "cell.json"),
        state,
        None,
    )


def test_status_cli_counts_complete_matrix_cells_and_sets_the_exit_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _config(tmp_path, monkeypatch)
    cells = (
        _cell_status(tmp_path, CampaignCompletionState.COMPLETE),
        _cell_status(tmp_path, CampaignCompletionState.STALE),
        _cell_status(tmp_path, CampaignCompletionState.MISSING),
    )
    monkeypatch.setattr("pace.cli.inspect_matrix_status", Mock(return_value=cells))
    assert main(["status", "n_baiot"]) == ExitStatus.FAILURE
    rendered = capsys.readouterr().out
    assert "N-BaIoT: 1 of 3 matrix cells complete" in rendered
    assert "provenance is stale" in rendered
    assert "artifact missing" in rendered
    unavailable = (_cell_status(tmp_path, CampaignCompletionState.SOURCES_MISSING),)
    monkeypatch.setattr("pace.cli.inspect_matrix_status", Mock(return_value=unavailable))
    assert main(["status", "n_baiot"]) == ExitStatus.FAILURE
    rendered = capsys.readouterr().out
    assert "raw data missing" in rendered
    assert "stale" not in rendered
    monkeypatch.setattr("pace.cli.inspect_matrix_status", Mock(return_value=cells[:1]))
    assert main(["status"]) == ExitStatus.SUCCESS
    assert capsys.readouterr().out.count("1 of 1 matrix cells complete") == 7


def test_run_cli_runs_each_stratum_matrix_and_writes_its_figures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _config(tmp_path, monkeypatch)
    campaign_result = CampaignWorkflowResult(
        ArtifactPath(tmp_path / "outputs" / "results.json"),
        ArtifactVerification(VerificationState.VERIFIED, "a" * 64),
        ArtifactReuseState.REUSED,
        27,
    )
    workflow = Mock(
        return_value=tuple(MatrixRunResult(variant, campaign_result) for variant in _variants())
    )
    figures = Mock(return_value=(tmp_path / "a.pdf", tmp_path / "a.png"))
    monkeypatch.setattr("pace.cli.run_registered_dataset_matrix_workflow", workflow)
    monkeypatch.setattr("pace.cli.write_stratum_figures", figures)

    assert main(["run", "n_baiot"]) == ExitStatus.SUCCESS

    rendered = capsys.readouterr().out
    assert rendered.count("N-BaIoT campaign") == 4
    assert "alpha=0.010, local rows=100" in rendered
    assert "alpha=0.050, local rows=400" in rendered
    assert "N-BaIoT figures: 2 files written" in rendered
    assert workflow.call_args.args[2] is StudyStratum.N_BAIOT
    assert figures.call_count == 1
    assert main(["run", "all", "--overwrite"]) == ExitStatus.SUCCESS
    assert figures.call_count == 8


def test_report_cli_writes_tables_and_figures_without_options(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _config(tmp_path, monkeypatch)
    files = ResultFiles(
        (tmp_path / "a.csv", tmp_path / "a.tex"), (tmp_path / "f.pdf", tmp_path / "f.png")
    )
    monkeypatch.setattr("pace.cli.write_results", Mock(return_value=files))
    assert main(["report"]) == ExitStatus.SUCCESS
    output = capsys.readouterr().out
    assert "Results written: 2 table files, 2 figure files" in output
    assert "pace calibration-study" in output
    assert main(["report", "--population", "confirmation"]) == ExitStatus.SUCCESS
    assert "pace calibration-study" not in capsys.readouterr().out


def _failing(*_arguments: object, **_keywords: object) -> None:
    raise PaceError("synthetic failure")


@pytest.mark.parametrize(
    ("target", "arguments"),
    [
        ("pace.cli.run_synthetic_workflow", ["doctor", "--smoke"]),
        ("pace.cli.run_registered_dataset_matrix_workflow", ["run", "iot_23"]),
        ("pace.cli.write_results", ["report"]),
        ("pace.cli.report_acceptance", ["accept"]),
    ],
)
def test_cli_reports_runtime_failures_with_their_message_and_a_dedicated_exit_code(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    target: str,
    arguments: list[str],
) -> None:
    _config(tmp_path, monkeypatch)
    monkeypatch.setattr(target, _failing)
    monkeypatch.setattr(
        "pace.cli.run_doctor", Mock(return_value=Mock(sources=(), is_complete=True))
    )
    assert main(arguments) == ExitStatus.RUNTIME_FAILURE
    assert "Error: synthetic failure" in capsys.readouterr().err
    log_path = tmp_path / "outputs" / "logs" / f"{arguments[0]}.jsonl"
    assert "synthetic failure" in log_path.read_text(encoding="utf-8")


def test_an_interrupted_command_exits_with_130_and_one_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _config(tmp_path, monkeypatch)
    monkeypatch.setattr("pace.cli.write_results", Mock(side_effect=KeyboardInterrupt))
    assert main(["report"]) == ExitStatus.INTERRUPTED == 130
    assert "Interrupted; completed cells are kept" in capsys.readouterr().err.splitlines()


def test_cli_requires_a_command() -> None:
    with pytest.raises(SystemExit) as error:
        main([])
    assert error.value.code == 2


def _support(name: str, training_rows: int, weak: bool = False) -> TargetBudgetSupport:
    return TargetBudgetSupport(
        DatasetClientId(DatasetName.TON_IOT, ClientName(name)),
        benign_rows=training_rows * 5 // 2,
        training_rows=training_rows,
        calibration_rows=training_rows // 2,
        attack_type_count=3,
        eligibility=TargetEligibility((TargetEligibilityIssue.INSUFFICIENT_PEERS,) if weak else ()),
    )


def _budget_plan() -> StratumBudgetPlan:
    candidates = (
        _support("big", 30000),
        _support("mid", 5000),
        _support("small", 600),
        _support("weak", 90000, weak=True),
    )
    return StratumBudgetPlan(
        StudyStratum.TON_IOT,
        (100, 400, 20000),
        candidates,
        (
            BudgetAnalysis(
                BudgetAnalysisKind.UNIVERSAL_PRIMARY,
                (100, 400),
                (ClientName("big"), ClientName("mid"), ClientName("small")),
            ),
            BudgetAnalysis(
                BudgetAnalysisKind.HIGH_RESOURCE, (100, 400, 20000), (ClientName("big"),)
            ),
        ),
        (20000,),
        (),
    )


def test_preflight_cli_prints_the_per_target_table_and_the_cohort_decision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _config(tmp_path, monkeypatch)
    service = Mock(return_value=_budget_plan())
    monkeypatch.setattr("pace.cli.preflight_registered_dataset", service)

    assert main(["preflight", "ton_iot"]) == ExitStatus.SUCCESS

    rendered = capsys.readouterr().out
    assert "TON_IoT budget feasibility" in rendered
    assert (
        "target big: benign 75000, train 30000, calibration 15000, "
        "attack types 3, >=100: yes, >=400: yes, >=20000: yes"
    ) in rendered
    assert "target mid: benign 12500, train 5000" in rendered
    assert ">=20000: no" in rendered
    assert "[INSUFFICIENT_PEERS]" in rendered
    assert ("summary targets_total=3 | >=100: 3 | >=400: 3 | >=20000: 1") in rendered
    assert "analysis universal_primary: budgets 100/400 on 3 targets" in rendered
    assert "analysis high_resource: budgets 100/400/20000 on 1 targets" in rendered
    assert "Outcome-independent power for the pooled paired mean difference" in rendered
    assert "Universal primary analysis: 3 targets" in rendered
    assert "Extreme high-resource analysis: 1 targets" in rendered
    assert "minimum detectable effect" in rendered
    assert "unavailable for full-stratum inference: 20000" in rendered
    assert "no valid observations" not in rendered
    assert service.call_count == 1


def test_preflight_cli_covers_every_registered_stratum_when_no_experiment_is_given(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _config(tmp_path, monkeypatch)
    service = Mock(return_value=_budget_plan())
    monkeypatch.setattr("pace.cli.preflight_registered_dataset", service)

    assert main(["preflight"]) == ExitStatus.SUCCESS

    assert service.call_count == 7
    assert {call.args[0] for call in service.call_args_list} == set(DatasetName)
    assert capsys.readouterr().out.count("budget feasibility") == 7


def test_preflight_cli_distinguishes_unavailable_from_unobserved_budgets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _config(tmp_path, monkeypatch)
    plan = replace(
        _budget_plan(), analyses=_budget_plan().analyses[:1], unobserved_budgets=(20000,)
    )
    monkeypatch.setattr("pace.cli.preflight_registered_dataset", Mock(return_value=plan))

    assert main(["preflight", "ton_iot"]) == ExitStatus.SUCCESS

    rendered = capsys.readouterr().out
    assert "unavailable for full-stratum inference: 20000" in rendered
    assert "no valid observations: 20000" in rendered


class _InlineExecutor:
    def __init__(self, max_workers: int, mp_context: object) -> None:
        self.max_workers = max_workers
        self.mp_context = mp_context

    def __enter__(self) -> "_InlineExecutor":
        return self

    def __exit__(self, *_arguments: object) -> None:
        return None

    def map[Item, Result](
        self, job: Callable[..., Result], *iterables: Iterable[Item]
    ) -> Iterator[Result]:
        return map(job, *iterables)


def test_run_cli_fans_strata_out_to_bounded_workers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _config(tmp_path, monkeypatch)
    ran = Mock()
    monkeypatch.setattr("pace.cli._run_stratum", ran)
    monkeypatch.setattr("pace.cli.ProcessPoolExecutor", _InlineExecutor)

    assert main(["run", "all", "--workers", "3"]) == ExitStatus.SUCCESS

    assert ran.call_count == len(CommandTarget.ALL.strata)
    assert capsys.readouterr().out.count("worker finished") == len(CommandTarget.ALL.strata)
    assert os.environ["OMP_NUM_THREADS"] == "1"
    assert os.environ["MKL_NUM_THREADS"] == "1"
    assert os.environ["OPENBLAS_NUM_THREADS"] == "1"


def test_run_cli_stays_sequential_for_a_single_stratum_even_with_workers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _config(tmp_path, monkeypatch)
    ran = Mock()
    monkeypatch.setattr("pace.cli._run_stratum", ran)
    assert main(["run", "n_baiot", "--workers", "4"]) == ExitStatus.SUCCESS
    assert ran.call_count == 1
    assert "--workers has no effect for a single stratum" in capsys.readouterr().err


def _measurement() -> ComputeMeasurement:
    return ComputeMeasurement(
        subject=ComputeSubject.PACE,
        phase=ComputePhase.INFERENCE,
        stratum=StudyStratum.N_BAIOT,
        target=DatasetClientId(DatasetName.N_BAIOT, ClientName("client-0")),
        seconds=1.0,
        seconds_sd=0.0,
        peak_megabytes=1.0,
        model_bytes=10,
        processed_rows=5,
    )


def test_benchmark_cli_writes_the_compute_cost_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _config(tmp_path, monkeypatch)
    service = Mock(return_value=(_measurement(),))
    monkeypatch.setattr("pace.reporting.studies.benchmark_registered_stratum", service)

    assert main(["benchmark", "n_baiot", "--repeats", "2"]) == ExitStatus.SUCCESS

    assert "Compute cost: 1 measurements" in capsys.readouterr().out
    assert service.call_args.args[2] == 2
    assert (tmp_path / "results" / "tables" / "compute-cost.csv").is_file()


def _role_audit(disjoint: bool) -> RoleAudit:
    return RoleAudit(
        stratum=StudyStratum.N_BAIOT,
        target=DatasetClientId(DatasetName.N_BAIOT, ClientName("client-0")),
        benign_rows=10,
        training_rows=4,
        reference_rows=2,
        calibration_rows=2,
        test_rows=2,
        attack_types=1,
        development_rows=5,
        held_out_rows=5,
        roles_disjoint=disjoint,
    )


@pytest.mark.parametrize(
    ("disjoint", "expected"), [(True, ExitStatus.SUCCESS), (False, ExitStatus.FAILURE)]
)
def test_audit_cli_fails_on_any_role_violation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    disjoint: bool,
    expected: ExitStatus,
) -> None:
    _config(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "pace.reporting.studies.audit_registered_stratum",
        Mock(return_value=(_role_audit(disjoint),)),
    )

    assert main(["audit", "n_baiot"]) == expected

    rendered = capsys.readouterr().out
    assert "Data roles: 1 clients audited" in rendered
    assert ("role violation: N-BaIoT client-0" in rendered) is (not disjoint)
    assert (tmp_path / "results" / "tables" / "data-roles.csv").is_file()


def _study_cells_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = CalibrationStudySettings(
        rows=(300,),
        budgets=(0.01,),
        expert_counts=(1,),
        replicates=1,
        test_rows=1500,
        attack_rows=300,
        seed=2,
    )
    cells = study_cells(settings)[:4]
    monkeypatch.setattr("pace.experiment.calibration_study.study_cells", Mock(return_value=cells))


@pytest.mark.parametrize("workers", [1, 3])
def test_calibration_study_cli_writes_the_sensitivity_table_and_figure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    workers: int,
) -> None:
    _config(tmp_path, monkeypatch)
    _study_cells_stub(monkeypatch)
    monkeypatch.setattr("pace.experiment.calibration_study.ProcessPoolExecutor", _InlineExecutor)

    assert main(["calibration-study", "--workers", f"{workers}"]) == ExitStatus.SUCCESS

    assert "Calibration study: 4 records" in capsys.readouterr().out
    assert (tmp_path / "results" / "tables" / "calibration-sensitivity.csv").is_file()
    assert (tmp_path / "results" / "figures" / "calibration-sensitivity.pdf").is_file()


def _acceptance_report(
    accepted: bool,
    reference_detection: float = 0.87,
    diagnostic: bool = False,
    gain: float = 0.03,
    harmed_targets: int = 0,
    gain_interval: tuple[float, float] = (0.01, 0.05),
) -> AcceptanceReport:
    interval = RateConfidenceInterval(
        lower_bound=gain_interval[0], upper_bound=gain_interval[1], confidence_level=0.95
    )
    levels = tuple(
        LevelGain(
            level=level,
            matched_targets=40,
            gain=gain,
            interval=interval,
            subject_detection=0.9,
            reference_detection=reference_detection,
        )
        for level in FprLevel
    )
    cell = AcceptanceCell(
        kind=(
            BudgetAnalysisKind.BROAD_SCALING if diagnostic else BudgetAnalysisKind.UNIVERSAL_PRIMARY
        ),
        alpha=AlphaLevel.ALPHA_001,
        local_training_rows=100,
        target_count=58,
        levels=levels,
        gain_without_excluded_interval=interval,
        harmed_target_count=harmed_targets,
        severe_target_count=0,
        worst_target_difference=0.0,
        false_alert_difference=0.001,
        false_alert_interval=RateConfidenceInterval(
            lower_bound=-0.002, upper_bound=0.001, confidence_level=0.95
        ),
        exceedance_increase=0.01,
        seed_gains=(SeedRateDifference(6, 0.03),),
        minimum_stratum_left_out_gain=0.01,
        verdicts=tuple(
            CriterionVerdict(
                criterion, accepted or criterion is not AcceptanceCriterion.FIXED_FPR_GAIN
            )
            for criterion in AcceptanceCriterion
        ),
    )
    return AcceptanceReport(OperatingMethod.MEAN_ENSEMBLE_SAME_CALIBRATION, (cell,))


def _mixed_report() -> AcceptanceReport:
    gating = _acceptance_report(True).cells[0]
    diagnostic = _acceptance_report(False, diagnostic=True).cells[0]
    return AcceptanceReport(OperatingMethod.MEAN_ENSEMBLE_SAME_CALIBRATION, (gating, diagnostic))


@pytest.mark.parametrize(
    ("accepted", "status", "verdict"),
    [(True, ExitStatus.SUCCESS, "ACCEPTED"), (False, ExitStatus.FAILURE, "REJECTED")],
)
def test_accept_cli_reports_the_verdict_and_sets_the_exit_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    accepted: bool,
    status: ExitStatus,
    verdict: str,
) -> None:
    _config(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "pace.cli.report_acceptance", Mock(return_value=_acceptance_report(accepted))
    )
    assert main(["accept"]) == status
    rendered = capsys.readouterr().out
    assert "universal_primary alpha=0.01 rows=100 targets=58" in rendered
    assert "fixed-FPR gain 0.5 alpha +0.0300 [0.0100, 0.0500] n=40" in rendered
    assert "Official protocol; population: primary" in rendered
    assert "same-calibration mean-ensemble local branch" in rendered
    assert "comparable realised FPR" in rendered
    assert f"PACE SCIENTIFIC ACCEPTANCE: {verdict}" in rendered
    assert ("failed [fixed_fpr_gain]" in rendered) is (not accepted)


def test_accept_cli_flags_development_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    path = _config(tmp_path, monkeypatch)
    path.write_text(
        path.read_text(encoding="utf-8").replace(
            "provenance_active: true", "provenance_active: false"
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("pace.cli.report_acceptance", Mock(return_value=_acceptance_report(True)))
    main(["accept"])
    captured = capsys.readouterr()
    assert "Official protocol" not in captured.out
    assert "not an official verdict: source verification is DEVELOPMENT; population" in captured.out
    assert "Warning: DEVELOPMENT source verification" in captured.err


@pytest.mark.parametrize(
    ("old", "new", "reason"),
    [
        (
            "evaluation_role: HELD_OUT",
            "evaluation_role: DEVELOPMENT_CALIBRATION_SPLIT",
            "evaluation role is DEVELOPMENT_CALIBRATION_SPLIT",
        ),
        (
            "calibration_mode: CROSS_FITTED",
            "calibration_mode: IN_SAMPLE",
            "calibration mode is IN_SAMPLE",
        ),
    ],
)
def test_accept_cli_names_why_a_configuration_is_not_official(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    old: str,
    new: str,
    reason: str,
) -> None:
    path = _config(tmp_path, monkeypatch)
    path.write_text(path.read_text(encoding="utf-8").replace(old, new), encoding="utf-8")
    monkeypatch.setattr("pace.cli.report_acceptance", Mock(return_value=_acceptance_report(True)))
    main(["accept"])
    captured = capsys.readouterr()
    assert f"not an official verdict: {reason}; population" in captured.out
    assert "Official protocol" not in captured.out
    assert "Warning" not in captured.err


def test_every_command_warns_on_stderr_under_development_verification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    path = _config(tmp_path, monkeypatch)
    path.write_text(
        path.read_text(encoding="utf-8").replace(
            "provenance_active: true", "provenance_active: false"
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("pace.cli.write_results", Mock(return_value=ResultFiles((), ())))
    main(["report"])
    assert "Warning: DEVELOPMENT source verification" in capsys.readouterr().err
    log_path = tmp_path / "outputs" / "logs" / "report.jsonl"
    assert '"verification": "DEVELOPMENT"' in log_path.read_text(encoding="utf-8")


def test_diagnostic_cells_carry_measured_limitations_and_no_verdict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _config(tmp_path, monkeypatch)
    monkeypatch.setattr("pace.cli.report_acceptance", Mock(return_value=_mixed_report()))
    assert main(["accept"]) == ExitStatus.SUCCESS
    rendered = capsys.readouterr().out
    assert (
        "Scaling and robustness diagnostics (measured evidence; no project-level verdict)"
        in rendered
    )
    diagnostic = next(line for line in rendered.splitlines() if line.startswith("broad_scaling"))
    assert "measured limitations: fixed_fpr_gain (fixed-FPR gain lower bound" in diagnostic
    assert "failed" not in diagnostic
    assert "REJECTED" not in rendered
    assert rendered.rstrip().endswith("PACE SCIENTIFIC ACCEPTANCE: ACCEPTED")


def test_accept_cli_passes_the_population_scope_to_the_reporting_service(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _config(tmp_path, monkeypatch)
    reported = Mock(return_value=_acceptance_report(True))
    monkeypatch.setattr("pace.cli.report_acceptance", reported)
    assert main(["accept", "--population", "confirmation"]) == ExitStatus.SUCCESS
    assert reported.call_args.args[2] is PopulationScope.CONFIRMATION
    assert "INDEPENDENT CONFIRMATION: CONFIRMED" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("accepted", "reference_detection", "gain", "label", "status"),
    [
        (True, 0.87, 0.03, "CONFIRMED", ExitStatus.SUCCESS),
        (True, 0.999, -0.002, "LIMITED EVIDENCE — DETECTION CEILING", ExitStatus.INCONCLUSIVE),
        (False, 0.87, 0.0, "SUPERIORITY NOT ESTABLISHED", ExitStatus.FAILURE),
    ],
)
def test_confirmation_exit_status_separates_confirmed_inconclusive_and_unestablished(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    accepted: bool,
    reference_detection: float,
    gain: float,
    label: str,
    status: ExitStatus,
) -> None:
    _config(tmp_path, monkeypatch)
    report = _acceptance_report(accepted, reference_detection, gain=gain)
    monkeypatch.setattr("pace.cli.report_acceptance", Mock(return_value=report))
    assert main(["accept", "--population", "confirmation"]) == status
    rendered = capsys.readouterr().out
    assert f"INDEPENDENT CONFIRMATION: {label}" in rendered
    assert "REJECTED" not in rendered
    assert ("Measured negative evidence" in rendered) is (gain < 0.0)


@pytest.mark.parametrize(
    ("harmed_targets", "gain_interval"),
    [(5, (-0.01, 0.0)), (0, (-0.03, -0.004))],
)
def test_a_detection_ceiling_never_hides_harm_or_statistically_supported_inferiority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    harmed_targets: int,
    gain_interval: tuple[float, float],
) -> None:
    _config(tmp_path, monkeypatch)
    report = _acceptance_report(
        False,
        0.999,
        gain=-0.005,
        harmed_targets=harmed_targets,
        gain_interval=gain_interval,
    )
    monkeypatch.setattr("pace.cli.report_acceptance", Mock(return_value=report))
    assert main(["accept", "--population", "confirmation"]) == ExitStatus.FAILURE
    rendered = capsys.readouterr().out
    assert "INDEPENDENT CONFIRMATION: SUPERIORITY NOT ESTABLISHED" in rendered
    assert "Measured negative evidence" in rendered


def test_every_command_and_parameter_has_help_text() -> None:
    group = typer.main.get_command(cli.app)
    assert isinstance(group, TyperGroup)
    assert set(group.commands) == {
        "doctor",
        "status",
        "preflight",
        "run",
        "report",
        "accept",
        "benchmark",
        "audit",
        "calibration-study",
    }
    for name, command in group.commands.items():
        assert command.help, name
        for parameter in command.params:
            assert getattr(parameter, "help", None), f"{name} {parameter.name}"
