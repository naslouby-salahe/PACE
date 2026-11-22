from pace.config import CalibrationStudySettings
from pace.experiment.calibration_study import run_study_cell, study_cells
from pace.reporting.calibration import (
    calibration_sensitivity_figure,
    calibration_sensitivity_table,
)
from pace.types import SensitivityRecord, TableName

SETTINGS = CalibrationStudySettings(
    rows=(300, 1200),
    budgets=(0.01, 0.05),
    expert_counts=(1, 4),
    replicates=2,
    test_rows=2000,
    attack_rows=400,
    seed=3,
)


def _records() -> tuple[SensitivityRecord, ...]:
    return tuple(item for cell in study_cells(SETTINGS)[:16] for item in run_study_cell(cell))


def test_sensitivity_table_has_one_aligned_row_per_record() -> None:
    records = _records()
    table = calibration_sensitivity_table(records)
    assert table.name is TableName.CALIBRATION_SENSITIVITY
    assert len(table.rows) == len(records)
    assert all(len(row) == len(table.columns) for row in table.rows)


def test_sensitivity_figure_draws_one_panel_per_scenario() -> None:
    figure = calibration_sensitivity_figure(_records())
    assert len(figure.axes) >= 1


def test_an_empty_study_draws_an_explicit_placeholder() -> None:
    figure = calibration_sensitivity_figure(())
    assert len(figure.axes) == 1
