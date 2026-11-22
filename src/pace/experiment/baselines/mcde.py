from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingClassifier

from pace.config import McdeSettings
from pace.types import (
    FloatArray,
    FloatMatrix,
    LearningSeed,
    PeerRowLimit,
    RowIndexArray,
)


@dataclass(frozen=True, slots=True)
class PeerSupervision:
    benign: FloatMatrix
    attack: FloatMatrix


def _balanced_rows(
    rows: FloatMatrix, size: PeerRowLimit, generator: np.random.Generator
) -> FloatMatrix:
    chosen = generator.choice(rows.shape[0], size, replace=rows.shape[0] < size)
    return rows[np.sort(chosen)]


def fit_supervised(
    supervision: PeerSupervision, settings: McdeSettings, seed: LearningSeed
) -> HistGradientBoostingClassifier:
    if supervision.benign.shape[0] == 0 or supervision.attack.shape[0] == 0:
        raise ValueError("MCDE supervision needs benign and attack peer rows")
    generator = np.random.default_rng(seed)
    size = min(
        settings.peer_row_limit, max(supervision.benign.shape[0], supervision.attack.shape[0])
    )
    benign = _balanced_rows(supervision.benign, size, generator)
    attack = _balanced_rows(supervision.attack, size, generator)
    labels = np.concatenate((np.zeros(size), np.ones(size)))
    classifier = HistGradientBoostingClassifier(
        max_iter=settings.boosting_iterations,
        learning_rate=settings.boosting_learning_rate,
        random_state=seed,
        early_stopping=False,
    )
    classifier.fit(np.concatenate((benign, attack), axis=0), labels)
    return classifier


@dataclass(frozen=True, slots=True)
class BenignModes:
    centers: FloatMatrix
    scales: FloatMatrix

    def assign(self, features: FloatMatrix) -> tuple[RowIndexArray, FloatArray]:
        standardized = (features[:, np.newaxis, :] - self.centers[np.newaxis]) / self.scales[
            np.newaxis
        ]
        distances = np.sqrt(np.mean(np.square(standardized), axis=2))
        modes = np.argmin(distances, axis=1)
        return modes, np.asarray(distances[np.arange(features.shape[0]), modes], dtype=np.float64)


def _global_scale(local_training: FloatMatrix, settings: McdeSettings) -> FloatArray:
    spread = np.std(local_training, axis=0)
    informative = spread[spread > settings.negligible_spread]
    floor = (
        max(settings.scale_floor, settings.scale_floor_share * np.median(informative).item())
        if informative.size
        else settings.scale_floor
    )
    return np.asarray(np.maximum(spread, floor), dtype=np.float64)


def fit_modes(
    local_training: FloatMatrix, settings: McdeSettings, seed: LearningSeed
) -> BenignModes:
    count = min(settings.benign_modes, np.unique(local_training, axis=0).shape[0])
    clustering = KMeans(n_clusters=count, n_init=3, random_state=seed).fit(local_training)
    centers = np.asarray(clustering.cluster_centers_, dtype=np.float64)
    global_scale = _global_scale(local_training, settings)
    scales = np.empty_like(centers)
    for mode in range(count):
        members = local_training[clustering.labels_ == mode]
        scales[mode] = (
            np.maximum(np.std(members, axis=0), settings.scale_floor_share * global_scale)
            if members.shape[0] >= settings.minimum_mode_rows
            else global_scale
        )
    return BenignModes(centers, scales)


def upper_tail(reference: FloatArray, scores: FloatArray) -> FloatArray:
    ordered = np.sort(reference)
    at_least = ordered.size - np.searchsorted(ordered, scores, side="left")
    return np.asarray((1.0 + at_least) / (ordered.size + 1.0), dtype=np.float64)


@dataclass(frozen=True, slots=True)
class McdeDetector:
    supervised: HistGradientBoostingClassifier
    modes: BenignModes
    settings: McdeSettings

    def _components(self, features: FloatMatrix) -> tuple[FloatArray, RowIndexArray, FloatArray]:
        assigned, distance = self.modes.assign(features)
        return (
            np.asarray(self.supervised.decision_function(features), dtype=np.float64),
            assigned,
            distance,
        )

    def _mode_tail(
        self,
        reference: tuple[RowIndexArray, FloatArray],
        query: tuple[RowIndexArray, FloatArray],
    ) -> FloatArray:
        reference_modes, reference_distance = reference
        query_modes, query_distance = query
        tail = upper_tail(reference_distance, query_distance)
        for mode in range(self.modes.centers.shape[0]):
            members = reference_distance[reference_modes == mode]
            selected = query_modes == mode
            if members.size >= self.settings.sparse_reference_rows and selected.any():
                tail[selected] = upper_tail(members, query_distance[selected])
        return tail

    def evidence(self, reference: FloatMatrix, query: FloatMatrix) -> FloatArray:
        reference_supervised, reference_modes, reference_distance = self._components(reference)
        query_supervised, query_modes, query_distance = self._components(query)
        share = self.settings.supervised_budget_share
        supervised_tail = upper_tail(reference_supervised, query_supervised)
        mode_tail = self._mode_tail(
            (reference_modes, reference_distance), (query_modes, query_distance)
        )
        return np.asarray(
            -np.minimum(supervised_tail / share, mode_tail / (1.0 - share)), dtype=np.float64
        )
