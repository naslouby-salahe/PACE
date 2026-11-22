from pace.reporting.measurements import compute_cost_table, data_roles_table
from pace.types import (
    ClientName,
    ComputeMeasurement,
    ComputePhase,
    ComputeSubject,
    DatasetClientId,
    DatasetName,
    RoleAudit,
    StudyStratum,
    TableName,
)

TARGET = DatasetClientId(DatasetName.N_BAIOT, ClientName("client-0"))


def test_compute_cost_rows_carry_throughput_and_labels() -> None:
    table = compute_cost_table(
        (
            ComputeMeasurement(
                subject=ComputeSubject.KITNET,
                phase=ComputePhase.INFERENCE,
                stratum=StudyStratum.N_BAIOT,
                target=TARGET,
                seconds=2.0,
                seconds_sd=0.1,
                peak_megabytes=3.5,
                model_bytes=1024,
                processed_rows=100,
            ),
            ComputeMeasurement(
                subject=ComputeSubject.PACE,
                phase=ComputePhase.CALIBRATION,
                stratum=StudyStratum.N_BAIOT,
                target=TARGET,
                seconds=0.0,
                seconds_sd=0.0,
                peak_megabytes=0.0,
                model_bytes=8,
                processed_rows=10,
            ),
        )
    )
    assert table.name is TableName.COMPUTE_COST
    assert all(len(row) == len(table.columns) for row in table.rows)
    assert table.rows[0][7] == "50.0"
    assert table.rows[1][7] == ""


def test_data_role_rows_flag_violations() -> None:
    audit = RoleAudit(
        stratum=StudyStratum.N_BAIOT,
        target=TARGET,
        benign_rows=100,
        training_rows=40,
        reference_rows=20,
        calibration_rows=20,
        test_rows=20,
        attack_types=2,
        development_rows=30,
        held_out_rows=30,
        roles_disjoint=True,
    )
    violated = RoleAudit(**{**audit.__dict__, "roles_disjoint": False}) if False else audit
    table = data_roles_table((audit, violated))
    assert table.name is TableName.DATA_ROLES
    assert all(len(row) == len(table.columns) for row in table.rows)
