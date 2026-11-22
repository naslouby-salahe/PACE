from pace.config import CalibrationStudySettings
from pace.experiment.calibration_study import run_study_cell, study_cells
from pace.types import CalibrationMode, CalibrationScenario

SETTINGS = CalibrationStudySettings(
    rows=(300, 1200),
    budgets=(0.01, 0.05),
    expert_counts=(1, 4),
    replicates=2,
    test_rows=3000,
    attack_rows=600,
    seed=5,
)


def test_the_grid_covers_every_scenario_mode_size_and_expert_count() -> None:
    cells = study_cells(SETTINGS)
    assert len(cells) == len(CalibrationScenario) * len(tuple(CalibrationMode)) * 2 * 2
    assert {cell.mode for cell in cells} == set(tuple(CalibrationMode))


def test_a_cell_records_one_result_per_budget() -> None:
    cell = next(
        item for item in study_cells(SETTINGS) if item.mode is CalibrationMode.POOLED_CONFORMAL
    )
    records = run_study_cell(cell)
    assert tuple(item.tier_budget for item in records) == SETTINGS.budgets
    for item in records:
        assert 0.0 <= item.mean_false_alert <= 1.0
        assert 0.0 <= item.mean_detection <= 1.0
        assert item.budget_ratio == item.mean_false_alert / item.tier_budget
        assert item.replicates == SETTINGS.replicates


def test_a_cell_is_deterministic() -> None:
    cell = study_cells(SETTINGS)[3]
    first = run_study_cell(cell)
    second = run_study_cell(cell)
    assert first == second


def test_attacks_are_detected_more_often_than_benign_traffic_alerts() -> None:
    for cell in study_cells(SETTINGS)[:8]:
        for item in run_study_cell(cell):
            assert item.mean_detection >= item.mean_false_alert
