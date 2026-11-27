from dataclasses import dataclass, field

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from pace.config import ScientificSettings
from pace.experiment.artifacts import derive_learning_seed
from pace.experiment.baselines.diot import (
    FederatedProfile,
    fine_tune_detector,
    fit_alphabet,
    fit_federated_profile,
)
from pace.experiment.baselines.kitnet import fit_kitnet
from pace.experiment.baselines.mcde import (
    McdeDetector,
    PeerSupervision,
    fit_modes,
    fit_supervised,
)
from pace.experiment.curves import score_threshold_curve
from pace.experiment.inputs import PreparedDevice, TargetFrame
from pace.experiment.models import select_local_training_rows
from pace.types import (
    DatasetClientId,
    ExternalBaseline,
    FloatArray,
    LearningSeed,
    LocalTrainingRowCount,
    OperatingCurve,
    SeedDomain,
)


@dataclass(frozen=True, slots=True)
class BaselineScores:
    calibration: FloatArray
    query: FloatArray


@dataclass(frozen=True, slots=True)
class _ScoreKey:
    client: DatasetClientId
    method: ExternalBaseline
    local_rows: LocalTrainingRowCount


@dataclass(slots=True)
class BaselineCache:
    supervised: dict[DatasetClientId, HistGradientBoostingClassifier] = field(
        default_factory=dict[DatasetClientId, HistGradientBoostingClassifier]
    )
    profiles: dict[tuple[DatasetClientId, LocalTrainingRowCount], FederatedProfile] = field(
        default_factory=dict[tuple[DatasetClientId, LocalTrainingRowCount], FederatedProfile]
    )
    scores: dict[_ScoreKey, BaselineScores] = field(default_factory=dict[_ScoreKey, BaselineScores])


def _query_segments(frame: TargetFrame) -> tuple[slice, ...]:
    return (slice(0, frame.benign_count), *frame.attack_slices())


def _kitnet_scores(
    target: PreparedDevice, frame: TargetFrame, scientific: ScientificSettings, seed: LearningSeed
) -> BaselineScores:
    model = fit_kitnet(
        select_local_training_rows(target.training, scientific.local_detectors.local_training_rows),
        scientific.baselines.kitnet,
        seed,
    )
    return BaselineScores(
        np.concatenate((model.scores(frame.reference), model.scores(frame.calibration))),
        model.scores(frame.query),
    )


def _peer_supervision(peers: tuple[PreparedDevice, ...]) -> PeerSupervision:
    benign = np.concatenate([peer.training for peer in peers], axis=0)
    attack = np.concatenate(
        [features for peer in peers for _, features in peer.development], axis=0
    )
    return PeerSupervision(benign, attack)


def _mcde_scores(
    target: PreparedDevice,
    peers: tuple[PreparedDevice, ...],
    frame: TargetFrame,
    scientific: ScientificSettings,
    seed: LearningSeed,
    cache: BaselineCache,
) -> BaselineScores:
    settings = scientific.baselines.mcde
    supervised = cache.supervised.get(target.client)
    if supervised is None:
        supervised = cache.supervised[target.client] = fit_supervised(
            _peer_supervision(peers), settings, seed
        )
    detector = McdeDetector(
        supervised,
        fit_modes(
            select_local_training_rows(
                target.training, scientific.local_detectors.local_training_rows
            ),
            settings,
            seed,
        ),
        settings,
    )
    return BaselineScores(
        detector.evidence(frame.reference, frame.calibration),
        detector.evidence(frame.reference, frame.query),
    )


def _diot_scores(
    target: PreparedDevice,
    peers: tuple[PreparedDevice, ...],
    frame: TargetFrame,
    scientific: ScientificSettings,
    seed: LearningSeed,
    cache: BaselineCache,
) -> BaselineScores:
    settings = scientific.baselines.diot
    local = select_local_training_rows(
        target.training, scientific.local_detectors.local_training_rows
    )
    key = (target.client, scientific.local_detectors.local_training_rows)
    profile = cache.profiles.get(key)
    if profile is None:
        share = max(1, settings.symbol_fit_row_limit // max(1, len(peers)))
        pooled = np.concatenate([peer.training[:share] for peer in peers] + [local], axis=0)
        profile = cache.profiles[key] = fit_federated_profile(
            tuple(peer.training for peer in peers),
            fit_alphabet(pooled, settings, seed),
            settings,
            seed,
        )
    detector = fine_tune_detector(profile, local, settings, seed)
    query = np.concatenate(
        [detector.surprisal(frame.query[segment]) for segment in _query_segments(frame)]
    )
    return BaselineScores(
        np.concatenate(
            (detector.surprisal(frame.reference), detector.surprisal(frame.calibration))
        ),
        query,
    )


def _baseline_scores(
    method: ExternalBaseline,
    target: PreparedDevice,
    peers: tuple[PreparedDevice, ...],
    frame: TargetFrame,
    scientific: ScientificSettings,
    seed: LearningSeed,
    cache: BaselineCache,
) -> BaselineScores:
    match method:
        case ExternalBaseline.KITNET:
            return _kitnet_scores(target, frame, scientific, seed)
        case ExternalBaseline.MCDE:
            return _mcde_scores(target, peers, frame, scientific, seed, cache)
        case ExternalBaseline.DIOT:
            return _diot_scores(target, peers, frame, scientific, seed, cache)


def baseline_curves(
    target: PreparedDevice,
    peers: tuple[PreparedDevice, ...],
    frame: TargetFrame,
    scientific: ScientificSettings,
    seed: LearningSeed,
    cache: BaselineCache,
) -> tuple[OperatingCurve, ...]:
    rows = scientific.local_detectors.local_training_rows
    if rows > scientific.baselines.maximum_local_training_rows:
        return ()
    derived = derive_learning_seed(seed, target.client, SeedDomain.BASELINE_MODEL)
    curves: list[OperatingCurve] = []
    for method in scientific.baselines.methods:
        key = _ScoreKey(target.client, method, rows)
        scores = cache.scores.get(key)
        if scores is None:
            scores = cache.scores[key] = _baseline_scores(
                method, target, peers, frame, scientific, derived, cache
            )
        curves.append(
            score_threshold_curve(
                method.operating_method,
                frame,
                scores.calibration,
                scores.query,
                scientific,
            )
        )
    return tuple(curves)
