from dataclasses import dataclass

import numpy as np
import torch
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.cluster import KMeans
from torch import Tensor, nn
from torch.nn.functional import cross_entropy, log_softmax
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from pace.config import DiotSettings
from pace.experiment.models import (
    TorchLoss,
    TorchOptimizer,
    adapt_library_result,
    build_seeded_module,
)
from pace.types import (
    FloatArray,
    FloatMatrix,
    IntMatrix,
    LearningSeed,
    LocalDetectorSeed,
    NumericGuard,
    SequenceLookback,
    SymbolCount,
    TrainingEpochCount,
)


@dataclass(frozen=True, slots=True)
class SymbolAlphabet:
    centers: FloatMatrix
    radii: FloatArray

    @property
    def known(self) -> SymbolCount:
        return self.centers.shape[0]

    @property
    def size(self) -> SymbolCount:
        return self.known + 1

    def _nearest(self, features: FloatMatrix) -> tuple[IntMatrix, FloatArray]:
        gaps = np.sum(np.square(features[:, np.newaxis, :] - self.centers[np.newaxis]), axis=2)
        nearest = np.asarray(np.argmin(gaps, axis=1), dtype=np.int64)
        return nearest, np.asarray(gaps[np.arange(features.shape[0]), nearest], dtype=np.float64)

    def symbols(self, features: FloatMatrix) -> IntMatrix:
        nearest, distance = self._nearest(features)
        unseen = distance > self.radii[nearest]
        return np.asarray(np.where(unseen, self.known, nearest), dtype=np.int64)

    def excess(self, features: FloatMatrix) -> FloatArray:
        nearest, distance = self._nearest(features)
        scale = np.maximum(self.radii[nearest], NumericGuard.NORMALISATION.guard)
        return np.asarray(np.log1p(np.maximum(distance - self.radii[nearest], 0.0) / scale))


def fit_alphabet(pooled: FloatMatrix, settings: DiotSettings, seed: LearningSeed) -> SymbolAlphabet:
    count = min(settings.symbol_count, np.unique(pooled, axis=0).shape[0])
    clustering = KMeans(n_clusters=count, n_init=3, random_state=seed).fit(pooled)
    centers = np.asarray(clustering.cluster_centers_, dtype=np.float64)
    gaps = np.sum(np.square(pooled - centers[clustering.labels_]), axis=1)
    radii = np.asarray(
        [
            np.quantile(gaps[clustering.labels_ == symbol], settings.unseen_quantile)
            if np.any(clustering.labels_ == symbol)
            else 0.0
            for symbol in range(count)
        ],
        dtype=np.float64,
    )
    return SymbolAlphabet(centers, radii)


class SymbolSequenceNetwork(nn.Module):
    def __init__(self, alphabet_size: SymbolCount, settings: DiotSettings) -> None:
        super().__init__()
        self.embedding = nn.Embedding(alphabet_size + 1, settings.hidden_width)
        self.recurrent = nn.GRU(
            settings.hidden_width,
            settings.hidden_width,
            num_layers=settings.hidden_layers,
            batch_first=True,
        )
        self.head = nn.Linear(settings.hidden_width, alphabet_size - 1)

    def forward(self, windows: Tensor) -> Tensor:
        output, _ = self.recurrent(self.embedding(windows))
        return self.head(output[:, -1, :])


def context_windows(symbols: IntMatrix, lookback: SequenceLookback, pad: SymbolCount) -> IntMatrix:
    padded = np.concatenate((np.full(lookback, pad, dtype=np.int64), symbols))
    return np.asarray(sliding_window_view(padded, lookback)[: symbols.size], dtype=np.int64)


def _train(
    network: SymbolSequenceNetwork,
    symbols: IntMatrix,
    alphabet_size: SymbolCount,
    settings: DiotSettings,
    epochs: TrainingEpochCount,
    generator: torch.Generator,
) -> None:
    contexts = context_windows(symbols, settings.lookback, alphabet_size)
    known = symbols < alphabet_size - 1
    windows = torch.tensor(contexts[known])
    targets = torch.tensor(symbols[known])
    optimizer = adapt_library_result(
        torch.optim.Adam(network.parameters(), lr=settings.learning_rate, weight_decay=0.0),
        torch.optim.Optimizer,
        TorchOptimizer,
    )
    network.train()
    for _ in range(epochs):
        order = torch.randperm(windows.shape[0], generator=generator)
        for start in range(0, windows.shape[0], settings.batch_size):
            rows = order[start : start + settings.batch_size]
            optimizer.zero_grad()
            loss = adapt_library_result(
                cross_entropy(network(windows[rows]), targets[rows]), Tensor, TorchLoss
            )
            loss.backward()
            optimizer.step()
    network.eval()


def _generator(seed: LearningSeed) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def _fresh_network(
    alphabet_size: SymbolCount, settings: DiotSettings, seed: LearningSeed
) -> SymbolSequenceNetwork:
    return build_seeded_module(
        lambda: SymbolSequenceNetwork(alphabet_size, settings), LocalDetectorSeed(seed)
    )


@dataclass(frozen=True, slots=True)
class FederatedProfile:
    alphabet: SymbolAlphabet
    parameters: Tensor


def fit_federated_profile(
    peer_streams: tuple[FloatMatrix, ...],
    alphabet: SymbolAlphabet,
    settings: DiotSettings,
    seed: LearningSeed,
) -> FederatedProfile:
    streams = tuple(
        alphabet.symbols(rows[: settings.peer_row_limit])
        for rows in peer_streams
        if rows.shape[0] > settings.lookback
    )
    if not streams:
        raise ValueError("Federated profile needs peer streams longer than the lookback")
    network = _fresh_network(alphabet.size, settings, seed)
    generator = _generator(seed)
    sizes = np.asarray([stream.size for stream in streams], dtype=np.float64)
    shares = sizes / np.sum(sizes)
    for _ in range(settings.federated_rounds):
        global_vector = parameters_to_vector(network.parameters()).detach().clone()
        merged = torch.zeros_like(global_vector)
        for stream, share in zip(streams, shares, strict=True):
            vector_to_parameters(global_vector.clone(), network.parameters())
            _train(network, stream, alphabet.size, settings, settings.local_epochs, generator)
            merged += share.item() * parameters_to_vector(network.parameters()).detach()
        vector_to_parameters(merged, network.parameters())
    return FederatedProfile(alphabet, parameters_to_vector(network.parameters()).detach().clone())


@dataclass(frozen=True, slots=True)
class DiotDetector:
    network: SymbolSequenceNetwork
    alphabet: SymbolAlphabet
    lookback: SequenceLookback

    def surprisal(self, features: FloatMatrix) -> FloatArray:
        symbols = self.alphabet.symbols(features)
        windows = torch.tensor(context_windows(symbols, self.lookback, self.alphabet.size))
        known = np.minimum(symbols, self.alphabet.known - 1)
        targets = torch.tensor(known)
        scores: list[FloatArray] = []
        with torch.no_grad():
            for start in range(0, windows.shape[0], 4096):
                logits = self.network(windows[start : start + 4096])
                chosen = log_softmax(logits, dim=1).gather(
                    1, targets[start : start + 4096].unsqueeze(1)
                )
                scores.append(np.asarray(-chosen.squeeze(1).numpy(), dtype=np.float64))
        surprisal = np.concatenate(scores)
        floor = -np.log(NumericGuard.NORMALISATION.guard)
        unseen = symbols == self.alphabet.known
        return np.where(unseen, floor + self.alphabet.excess(features), surprisal)


def fine_tune_detector(
    profile: FederatedProfile,
    local_training: FloatMatrix,
    settings: DiotSettings,
    seed: LearningSeed,
) -> DiotDetector:
    network = _fresh_network(profile.alphabet.size, settings, seed)
    vector_to_parameters(profile.parameters.clone(), network.parameters())
    symbols = profile.alphabet.symbols(local_training)
    _train(
        network,
        symbols,
        profile.alphabet.size,
        settings,
        settings.fine_tune_epochs,
        _generator(seed),
    )
    return DiotDetector(network, profile.alphabet, settings.lookback)
