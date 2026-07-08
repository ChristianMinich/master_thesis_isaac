"""Trainer port: training / adaptation pipelines for learning-based approaches.

The exposé distinguishes training (RL policy, World Model), adaptation /
fine-tuning (VLA), and configuration/calibration (SLAM). All of these are
expressed through this single port; non-trainable approaches provide a
no-op trainer that only records configuration effort.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from navbench.core.agent import Agent
from navbench.core.dataset import EpisodeDataset
from navbench.core.simulator import SimulatorAdapter


@dataclass(frozen=True)
class TrainingResult:
    """Outcome of one training / adaptation run.

    Records the transparency data required by the exposé's model-training
    scope: steps, compute, data volume, and the checkpoint produced.
    """

    approach_name: str
    checkpoint_path: Path | None
    num_steps: int = 0
    num_episodes_used: int = 0
    wall_clock_seconds: float = 0.0
    final_metrics: Mapping[str, float] = field(default_factory=dict)
    hyperparameters: Mapping[str, Any] = field(default_factory=dict)


class Trainer(ABC):
    """Port: produces a trained/adapted Agent state.

    Dependencies are injected (dataset for offline learning, simulator for
    interactive RL) — the trainer never constructs its own simulator or
    dataset, which keeps backends swappable.
    """

    @abstractmethod
    def train(
        self,
        agent: Agent,
        dataset: EpisodeDataset,
        simulator: SimulatorAdapter | None,
        output_dir: Path,
        seed: int,
    ) -> TrainingResult:
        """Train or adapt ``agent``.

        - Offline approaches (WM, VLA fine-tuning) consume ``dataset``.
        - Interactive approaches (RL) additionally use ``simulator``.
        - SLAM-style approaches may only calibrate parameters.

        Implementations must write checkpoints and logs under ``output_dir``
        and be reproducible w.r.t. ``seed``.
        """


class NoOpTrainer(Trainer):
    """Trainer for approaches without a neural training phase (e.g. SLAM).

    Still returns a TrainingResult so effort accounting stays uniform.
    """

    def __init__(self, approach_name: str = "untrained") -> None:
        self._approach_name = approach_name

    def train(
        self,
        agent: Agent,
        dataset: EpisodeDataset,
        simulator: SimulatorAdapter | None,
        output_dir: Path,
        seed: int,
    ) -> TrainingResult:
        return TrainingResult(
            approach_name=self._approach_name,
            checkpoint_path=None,
            final_metrics={},
            hyperparameters={"note": "no training phase; configuration/calibration only"},
        )