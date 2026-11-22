from dataclasses import dataclass
from pathlib import Path

from pace.config import ResolvedConfiguration
from pace.experiment.audit import audit_registered_stratum
from pace.experiment.benchmark import benchmark_registered_stratum
from pace.experiment.calibration_study import run_study
from pace.reporting.calibration import (
    calibration_sensitivity_figure,
    calibration_sensitivity_table,
)
from pace.reporting.figures import save_figure
from pace.reporting.measurements import compute_cost_table, data_roles_table
from pace.reporting.service import results_root, write_table
from pace.types import (
    FigureName,
    ObservationCount,
    OutputDirectory,
    RepeatCount,
    RoleAudit,
    StudyStratum,
    WorkerCount,
)


@dataclass(frozen=True, slots=True)
class StudyOutput:
    path: Path
    records: ObservationCount


def write_compute_cost(
    configuration: ResolvedConfiguration, strata: tuple[StudyStratum, ...], repeats: RepeatCount
) -> StudyOutput:
    measurements = tuple(
        measurement
        for stratum in strata
        for measurement in benchmark_registered_stratum(
            stratum, configuration, repeats, configuration.reporting.benchmark_targets
        )
    )
    paths = write_table(compute_cost_table(measurements), results_root(configuration))
    return StudyOutput(paths[0], len(measurements))


def write_data_roles(
    configuration: ResolvedConfiguration, strata: tuple[StudyStratum, ...]
) -> tuple[StudyOutput, tuple[RoleAudit, ...]]:
    audits = tuple(
        item for stratum in strata for item in audit_registered_stratum(stratum, configuration)
    )
    paths = write_table(data_roles_table(audits), results_root(configuration))
    return StudyOutput(paths[0], len(audits)), tuple(
        item for item in audits if not item.roles_disjoint
    )


def write_calibration_study(
    configuration: ResolvedConfiguration, workers: WorkerCount
) -> StudyOutput:
    records = run_study(configuration.reporting.calibration_study, workers)
    root = results_root(configuration)
    paths = write_table(calibration_sensitivity_table(records), root)
    save_figure(
        calibration_sensitivity_figure(records),
        FigureName.CALIBRATION_SENSITIVITY,
        root / OutputDirectory.FIGURES,
    )
    return StudyOutput(paths[0], len(records))
