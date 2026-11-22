import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np
import structlog

from pace.config import CalibrationStudySettings
from pace.core.decision import evaluate_peer_union
from pace.types import (
    CalibrationMode,
    CalibrationScenario,
    CriterionSatisfied,
    DetectionRate,
    DetectorCount,
    EnvironmentVariable,
    FalseAlarmBudget,
    FalseAlertRate,
    FloatMatrix,
    LearningSeed,
    LogEvent,
    ObservationCount,
    SensitivityRecord,
    WorkerCount,
)

logger = structlog.get_logger()


@dataclass(frozen=True, slots=True)
class StudyCell:
    scenario: CalibrationScenario
    mode: CalibrationMode
    rows: ObservationCount
    experts: DetectorCount
    settings: CalibrationStudySettings


def study_cells(settings: CalibrationStudySettings) -> tuple[StudyCell, ...]:
    return tuple(
        StudyCell(scenario, mode, rows, experts, settings)
        for scenario in CalibrationScenario
        for experts in settings.expert_counts
        for rows in settings.rows
        for mode in CalibrationMode
    )


def _scores(
    scenario: CalibrationScenario,
    experts: DetectorCount,
    rows: ObservationCount,
    generator: np.random.Generator,
    drifted: CriterionSatisfied,
) -> FloatMatrix:
    match scenario:
        case CalibrationScenario.HEAVY_TAILED:
            return np.exp(generator.normal(size=(experts, rows)))
        case CalibrationScenario.CORRELATED_EXPERTS:
            factor = generator.exponential(size=(1, rows))
            return generator.exponential(size=(experts, rows)) + 0.8 * factor
        case CalibrationScenario.DRIFT:
            scale = 1.25 if drifted else 1.0
            return scale * generator.exponential(size=(experts, rows))
        case CalibrationScenario.IID_EXPONENTIAL:
            return generator.exponential(size=(experts, rows))


def _attack(
    scenario: CalibrationScenario,
    experts: DetectorCount,
    rows: ObservationCount,
    generator: np.random.Generator,
) -> FloatMatrix:
    scores = _scores(scenario, experts, rows, generator, False)
    informative = max(1, experts // 2)
    scores[:informative] = scores[:informative] * 3.0 + 4.0
    return scores


def _replicate(
    cell: StudyCell, budget: FalseAlarmBudget, seed: LearningSeed
) -> tuple[FalseAlertRate, DetectionRate]:
    settings = cell.settings
    generator = np.random.default_rng(seed)
    calibration = _scores(cell.scenario, cell.experts, cell.rows, generator, False)
    benign = _scores(cell.scenario, cell.experts, settings.test_rows, generator, True)
    attack = _attack(cell.scenario, cell.experts, settings.attack_rows, generator)
    _, alerts = evaluate_peer_union(
        calibration,
        np.concatenate((benign, attack), axis=1),
        250,
        budget,
        64.0,
        20,
        None,
        cell.mode,
        5,
        2,
    )
    return np.mean(alerts[: settings.test_rows]).item(), np.mean(
        alerts[settings.test_rows :]
    ).item()


def run_study_cell(cell: StudyCell) -> tuple[SensitivityRecord, ...]:
    records: list[SensitivityRecord] = []
    for budget in cell.settings.budgets:
        outcomes = [
            _replicate(cell, budget, cell.settings.seed + 1009 * index + cell.rows)
            for index in range(cell.settings.replicates)
        ]
        false_alert = np.mean([item[0] for item in outcomes]).item()
        records.append(
            SensitivityRecord(
                scenario=cell.scenario,
                mode=cell.mode,
                rows=cell.rows,
                experts=cell.experts,
                tier_budget=budget,
                replicates=cell.settings.replicates,
                mean_false_alert=false_alert,
                mean_detection=np.mean([item[1] for item in outcomes]).item(),
                budget_ratio=false_alert / budget,
            )
        )
    logger.info(
        LogEvent.CALIBRATION_STUDY_RECORDED,
        scenario=cell.scenario.name,
        mode=cell.mode.name,
        rows=cell.rows,
        experts=cell.experts,
    )
    return tuple(records)


def run_study(
    settings: CalibrationStudySettings, workers: WorkerCount
) -> tuple[SensitivityRecord, ...]:
    cells = study_cells(settings)
    if workers <= 1:
        return tuple(item for cell in cells for item in run_study_cell(cell))
    single_thread: WorkerCount = 1
    for variable in (
        EnvironmentVariable.OMP_THREADS,
        EnvironmentVariable.MKL_THREADS,
        EnvironmentVariable.OPENBLAS_THREADS,
    ):
        os.environ[variable] = f"{single_thread}"
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        return tuple(item for batch in pool.map(run_study_cell, cells) for item in batch)
