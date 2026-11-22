from pace.reporting.records import ResultTable, count, number
from pace.types import (
    ColumnHeading,
    ComputeMeasurement,
    DisplayText,
    RoleAudit,
    SatisfiedAnswer,
    TableName,
)


def _throughput(measurement: ComputeMeasurement) -> DisplayText:
    if measurement.seconds <= 0.0:
        return DisplayText("")
    return DisplayText(f"{measurement.processed_rows / measurement.seconds:.1f}")


def compute_cost_table(measurements: tuple[ComputeMeasurement, ...]) -> ResultTable:
    rows = tuple(
        (
            item.stratum.display_label,
            DisplayText(item.target.name),
            item.subject.display_label,
            item.phase.display_label,
            count(item.processed_rows),
            number(item.seconds),
            number(item.seconds_sd),
            _throughput(item),
            number(item.peak_megabytes),
            count(item.model_bytes),
        )
        for item in measurements
    )
    return ResultTable(
        TableName.COMPUTE_COST,
        (
            ColumnHeading.STRATUM,
            ColumnHeading.TARGET,
            ColumnHeading.METHOD,
            ColumnHeading.PHASE,
            ColumnHeading.PROCESSED_ROWS,
            ColumnHeading.SECONDS,
            ColumnHeading.SECONDS_SD,
            ColumnHeading.THROUGHPUT,
            ColumnHeading.PEAK_MEMORY,
            ColumnHeading.MODEL_BYTES,
        ),
        rows,
    )


def data_roles_table(audits: tuple[RoleAudit, ...]) -> ResultTable:
    rows = tuple(
        (
            item.stratum.display_label,
            DisplayText(item.target.name),
            count(item.benign_rows),
            count(item.training_rows),
            count(item.reference_rows),
            count(item.calibration_rows),
            count(item.test_rows),
            count(item.attack_types),
            count(item.development_rows),
            count(item.held_out_rows),
            DisplayText(SatisfiedAnswer.YES if item.roles_disjoint else SatisfiedAnswer.NO),
        )
        for item in audits
    )
    return ResultTable(
        TableName.DATA_ROLES,
        (
            ColumnHeading.STRATUM,
            ColumnHeading.TARGET,
            ColumnHeading.BENIGN_ROWS,
            ColumnHeading.TRAINING_ROWS,
            ColumnHeading.REFERENCE_ROWS,
            ColumnHeading.CALIBRATION_ROWS,
            ColumnHeading.BENIGN_TEST_ROWS,
            ColumnHeading.ATTACK_FAMILIES,
            ColumnHeading.DEVELOPMENT_ROWS,
            ColumnHeading.HELD_OUT_ROWS,
            ColumnHeading.DISJOINT,
        ),
        rows,
    )
