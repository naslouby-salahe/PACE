import pickle
import time
import tracemalloc
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import structlog

from pace.config import ResolvedConfiguration
from pace.core.decision import block_max_threshold, consecutive_blocks
from pace.datasets.cache import resolve_clients
from pace.datasets.registry import dataset_source_files, load_dataset
from pace.experiment.arms import protected_channel_decisions
from pace.experiment.artifacts import derive_learning_seed
from pace.experiment.baselines.diot import fine_tune_detector, fit_alphabet, fit_federated_profile
from pace.experiment.baselines.kitnet import fit_kitnet
from pace.experiment.baselines.mcde import McdeDetector, PeerSupervision, fit_modes, fit_supervised
from pace.experiment.inputs import (
    PreparedDevice,
    build_target_frame,
    compute_local_evidence,
    compute_peer_evidence,
    fit_peer_experts,
    train_peer_mlp,
)
from pace.experiment.models import fit_local_ensemble, select_local_training_rows
from pace.experiment.runner import prepare_clients
from pace.experiment.sources import digest_source_files
from pace.types import (
    ComputeMeasurement,
    ComputePhase,
    ComputeSubject,
    ElapsedSeconds,
    FloatMatrix,
    LearningSeed,
    LogEvent,
    MegabyteCount,
    ObservationCount,
    RepeatCount,
    SeedDomain,
    StudyStratum,
)

logger = structlog.get_logger()


@dataclass(frozen=True, slots=True)
class _Timing:
    seconds: ElapsedSeconds
    seconds_sd: ElapsedSeconds
    peak_megabytes: MegabyteCount
    model_bytes: ObservationCount


def _measure[Result](
    operation: Callable[[], Result], repeats: RepeatCount
) -> tuple[Result, _Timing]:
    durations: list[ElapsedSeconds] = []
    result = operation()
    for _ in range(repeats):
        started = time.perf_counter()
        result = operation()
        durations.append(time.perf_counter() - started)
    tracemalloc.start()
    operation()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, _Timing(
        np.mean(durations).item(),
        np.std(durations).item(),
        peak / 1_048_576.0,
        len(pickle.dumps(result)),
    )


@dataclass(frozen=True, slots=True)
class _Bench:
    stratum: StudyStratum
    target: PreparedDevice
    repeats: RepeatCount
    rows: ObservationCount

    def record(
        self, subject: ComputeSubject, phase: ComputePhase, timing: _Timing
    ) -> ComputeMeasurement:
        logger.info(
            LogEvent.BENCHMARK_RECORDED,
            subject=subject.name,
            phase=phase.name,
            seconds=timing.seconds,
        )
        return ComputeMeasurement(
            subject=subject,
            phase=phase,
            stratum=self.stratum,
            target=self.target.client,
            seconds=timing.seconds,
            seconds_sd=timing.seconds_sd,
            peak_megabytes=timing.peak_megabytes,
            model_bytes=timing.model_bytes,
            processed_rows=self.rows,
        )


def _pace_measurements(
    bench: _Bench,
    prepared: tuple[PreparedDevice, ...],
    configuration: ResolvedConfiguration,
    seed: LearningSeed,
) -> tuple[ComputeMeasurement, ...]:
    scientific = configuration.scientific
    peer = next(
        device for device in prepared if device.client != bench.target.client and device.development
    )
    _, offline = _measure(
        lambda: train_peer_mlp(
            peer.training,
            np.concatenate([features for _, features in peer.development]),
            seed,
            scientific.peer_training,
        ),
        1,
    )
    experts = tuple(
        expert
        for expert in fit_peer_experts(prepared, configuration, seed)
        if expert.client != bench.target.client
    )
    frame = build_target_frame(bench.target)
    peers = compute_peer_evidence(experts, frame)
    local_rows = select_local_training_rows(
        bench.target.training, scientific.local_detectors.local_training_rows
    )
    device = configuration.runtime.execution_device
    _, fit = _measure(
        lambda: fit_local_ensemble(local_rows, scientific.local_detectors, seed, device),
        bench.repeats,
    )
    local = compute_local_evidence(bench.target, frame, configuration, seed, peers)
    reserve = scientific.reserve_fraction
    budget = scientific.alpha.fraction
    _, calibration = _measure(
        lambda: protected_channel_decisions(
            local,
            peers.calibration_evidence,
            peers.query_evidence,
            scientific,
            budget * reserve,
            budget * (1.0 - reserve),
        ),
        bench.repeats,
    )
    _, inference = _measure(lambda: compute_peer_evidence(experts, frame), bench.repeats)
    return (
        bench.record(ComputeSubject.PACE, ComputePhase.OFFLINE_TRAINING, offline),
        bench.record(ComputeSubject.PACE, ComputePhase.ONBOARDING_FIT, fit),
        bench.record(ComputeSubject.PACE, ComputePhase.CALIBRATION, calibration),
        bench.record(ComputeSubject.PACE, ComputePhase.INFERENCE, inference),
    )


def _baseline_measurements(
    bench: _Bench,
    prepared: tuple[PreparedDevice, ...],
    configuration: ResolvedConfiguration,
    seed: LearningSeed,
) -> tuple[ComputeMeasurement, ...]:
    scientific = configuration.scientific
    target = bench.target
    frame = build_target_frame(target)
    peers = tuple(
        device for device in prepared if device.client != target.client and device.development
    )
    derived = derive_learning_seed(seed, target.client, SeedDomain.BASELINE_MODEL)
    local = select_local_training_rows(
        target.training, scientific.local_detectors.local_training_rows
    )
    blocks = consecutive_blocks(frame.calibration[:, 0], scientific.calibration_block_size)
    kitnet, kit_fit = _measure(
        lambda: fit_kitnet(local, scientific.baselines.kitnet, derived), bench.repeats
    )
    _, kit_inference = _measure(lambda: kitnet.scores(frame.query), bench.repeats)
    _, threshold = _measure(
        lambda: block_max_threshold(blocks, scientific.alpha.fraction), bench.repeats
    )
    supervision = PeerSupervision(
        np.concatenate([peer.training for peer in peers]),
        np.concatenate([features for peer in peers for _, features in peer.development]),
    )
    supervised, mcde_offline = _measure(
        lambda: fit_supervised(supervision, scientific.baselines.mcde, derived), 1
    )
    modes, mcde_fit = _measure(
        lambda: fit_modes(local, scientific.baselines.mcde, derived), bench.repeats
    )
    mcde = McdeDetector(supervised, modes, scientific.baselines.mcde)
    _, mcde_inference = _measure(lambda: mcde.evidence(frame.reference, frame.query), bench.repeats)
    return (
        bench.record(ComputeSubject.KITNET, ComputePhase.ONBOARDING_FIT, kit_fit),
        bench.record(ComputeSubject.KITNET, ComputePhase.CALIBRATION, threshold),
        bench.record(ComputeSubject.KITNET, ComputePhase.INFERENCE, kit_inference),
        bench.record(ComputeSubject.MCDE, ComputePhase.OFFLINE_TRAINING, mcde_offline),
        bench.record(ComputeSubject.MCDE, ComputePhase.ONBOARDING_FIT, mcde_fit),
        bench.record(ComputeSubject.MCDE, ComputePhase.INFERENCE, mcde_inference),
        *_diot_measurements(bench, peers, local, frame.query, configuration, derived),
    )


def _diot_measurements(
    bench: _Bench,
    peers: tuple[PreparedDevice, ...],
    local: FloatMatrix,
    query: FloatMatrix,
    configuration: ResolvedConfiguration,
    seed: LearningSeed,
) -> tuple[ComputeMeasurement, ...]:
    settings = configuration.scientific.baselines.diot
    pooled = np.concatenate([peer.training[: settings.symbol_fit_row_limit] for peer in peers])
    profile, offline = _measure(
        lambda: fit_federated_profile(
            tuple(peer.training for peer in peers),
            fit_alphabet(pooled, settings, seed),
            settings,
            seed,
        ),
        1,
    )
    detector, fit = _measure(
        lambda: fine_tune_detector(profile, local, settings, seed), bench.repeats
    )
    _, inference = _measure(lambda: detector.surprisal(query), bench.repeats)
    return (
        bench.record(ComputeSubject.DIOT, ComputePhase.OFFLINE_TRAINING, offline),
        bench.record(ComputeSubject.DIOT, ComputePhase.ONBOARDING_FIT, fit),
        bench.record(ComputeSubject.DIOT, ComputePhase.INFERENCE, inference),
    )


def benchmark_devices(
    stratum: StudyStratum,
    prepared: tuple[PreparedDevice, ...],
    configuration: ResolvedConfiguration,
    repeats: RepeatCount,
    target_count: ObservationCount,
) -> tuple[ComputeMeasurement, ...]:
    seed = configuration.scientific.learning_seeds[0]
    rows = configuration.scientific.local_detectors.local_training_rows
    eligible = tuple(
        device
        for device in prepared
        if device.development
        and device.held_out
        and device.training.shape[0] >= rows
        and device.reference.shape[0] > 0
    )
    measurements: list[ComputeMeasurement] = []
    for target in eligible[:target_count]:
        bench = _Bench(stratum, target, repeats, build_target_frame(target).query.shape[0])
        measurements.extend(_pace_measurements(bench, prepared, configuration, seed))
        measurements.extend(_baseline_measurements(bench, prepared, configuration, seed))
    return tuple(measurements)


def benchmark_registered_stratum(
    stratum: StudyStratum,
    configuration: ResolvedConfiguration,
    repeats: RepeatCount,
    target_count: ObservationCount,
) -> tuple[ComputeMeasurement, ...]:
    root = configuration.runtime.raw_data_root
    files = dataset_source_files(root, stratum.dataset, stratum)
    clients = resolve_clients(
        lambda: (
            load_dataset(root, configuration.runtime.lab_network, stratum.dataset, stratum).clients
        ),
        configuration,
        (stratum.dataset, stratum),
        digest_source_files(files, root),
        logger,
    )
    return benchmark_devices(
        stratum, prepare_clients(clients, configuration), configuration, repeats, target_count
    )
