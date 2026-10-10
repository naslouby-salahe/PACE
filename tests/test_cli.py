from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import pytest

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
    CriterionVerdict,
    DatasetClientId,
    DatasetLocationKind,
    DatasetName,
    ExitStatus,
    PaceError,
    RateConfidenceInterval,
    SeedRateDifference,
    StratumBudgetPlan,
    StudyArm,
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
    assert "Results written: 2 table files, 2 figure files" in capsys.readouterr().out


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
def test_cli_translates_domain_errors_into_parser_errors(
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
    with pytest.raises(SystemExit) as error:
        main(arguments)
    assert error.value.code == 2
    assert "synthetic failure" in capsys.readouterr().err


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
    assert "TON_IOT budget feasibility" in rendered
    assert (
        "target big: benign 75000, train 30000, available 30000, calibration 15000, "
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


def _acceptance_report(accepted: bool) -> AcceptanceReport:
    interval = RateConfidenceInterval(lower_bound=0.01, upper_bound=0.05, confidence_level=0.95)
    cell = AcceptanceCell(
        kind=BudgetAnalysisKind.UNIVERSAL_PRIMARY,
        alpha=AlphaLevel.ALPHA_001,
        local_training_rows=100,
        target_count=58,
        nominal_gain=0.03,
        nominal_interval=interval,
        matched_gain=0.02,
        matched_interval=interval,
        gain_without_excluded=0.01,
        gain_without_excluded_interval=interval,
        harmed_target_count=0,
        severe_target_count=0,
        worst_target_difference=0.0,
        false_alert_increase=0.002,
        exceedance_increase=0.01,
        seed_gains=(SeedRateDifference(6, 0.03),),
        minimum_stratum_left_out_gain=0.01,
        verdicts=(CriterionVerdict(AcceptanceCriterion.NOMINAL_GAIN, accepted),),
    )
    return AcceptanceReport(StudyArm.MEAN_ENSEMBLE_LOCAL_BRANCH, (cell,))


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
    assert "nominal +0.0300 95% CI [0.0100, 0.0500]" in rendered
    assert "matched +0.0200 95% CI [0.0100, 0.0500]" in rendered
    assert f"acceptance against MEAN_ENSEMBLE_LOCAL_BRANCH: {verdict}" in rendered
    assert ("failed [nominal_gain]" in rendered) is (not accepted)
