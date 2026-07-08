"""Episode dataset abstractions (the "common episode dataset" of the exposé).

An Episode is one complete or interrupted navigation attempt with synchronized
observations, actions, rewards, and perturbation/failure metadata.

EpisodeDataset is the port through which trainers and evaluators access
episodes; storage backends (in-memory, on-disk, HDF5/Zarr, Isaac replicator
output) are interchangeable implementations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterator, Mapping, Sequence

from navbench.core.scenario import Scenario
from navbench.core.types import Action, Observation


@dataclass(frozen=True)
class Episode:
    """One recorded navigation attempt.

    Invariant: len(actions) == len(observations) - 1 == len(rewards)
    (the first observation is the reset observation).
    """

    episode_id: str
    scenario: Scenario
    observations: Sequence[Observation]
    actions: Sequence[Action]
    rewards: Sequence[float]
    success: bool
    seed: int
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def length(self) -> int:
        return len(self.actions)


class EpisodeDataset(ABC):
    """Port: read access to a collection of episodes.

    Splits ("train", "val", "test") follow the exposé's common-dataset
    strategy: nominal episodes for training, perturbed episodes reserved
    for validation/evaluation.
    """

    @abstractmethod
    def episode_ids(self, split: str = "train") -> Sequence[str]:
        """Return the ids of all episodes in a split."""

    @abstractmethod
    def load(self, episode_id: str) -> Episode:
        """Load a single episode by id."""

    def iter_episodes(self, split: str = "train") -> Iterator[Episode]:
        for episode_id in self.episode_ids(split):
            yield self.load(episode_id)

    def __len__(self) -> int:
        return sum(len(self.episode_ids(split)) for split in self.splits())

    @abstractmethod
    def splits(self) -> Sequence[str]:
        """Return available split names."""


class WritableEpisodeDataset(EpisodeDataset):
    """Port: datasets that also support appending recorded episodes."""

    @abstractmethod
    def add(self, episode: Episode, split: str = "train") -> None:
        """Append an episode to a split."""


class InMemoryEpisodeDataset(WritableEpisodeDataset):
    """Simple in-memory dataset used for tests and small pilot experiments.

    TODO(dataset): add an on-disk backend (e.g. per-episode directories with
    parquet/npz payloads, or HDF5/Zarr) once the Isaac logging pipeline
    produces real sensor data. The on-disk backend must implement the same
    EpisodeDataset port so trainers/evaluators stay unchanged.
    """

    def __init__(self) -> None:
        self._episodes: dict[str, Episode] = {}
        self._splits: dict[str, list[str]] = {}

    def add(self, episode: Episode, split: str = "train") -> None:
        if episode.episode_id in self._episodes:
            raise ValueError(f"duplicate episode id: {episode.episode_id!r}")
        self._episodes[episode.episode_id] = episode
        self._splits.setdefault(split, []).append(episode.episode_id)

    def episode_ids(self, split: str = "train") -> Sequence[str]:
        return tuple(self._splits.get(split, ()))

    def load(self, episode_id: str) -> Episode:
        try:
            return self._episodes[episode_id]
        except KeyError as exc:
            raise KeyError(f"unknown episode id: {episode_id!r}") from exc

    def splits(self) -> Sequence[str]:
        return tuple(self._splits.keys())