"""Agent port: the single interface every navigation approach must implement.

An Agent maps Observations to Actions. It is deliberately minimal so that
radically different paradigms (SLAM pose monitor, RL policy, VLA module,
World Model planner, hybrids) can all sit behind it.

Design notes:
- Composition over inheritance: approaches should wrap their internals
  (SLAM system handle, policy network, planner) as members, not subclass
  deep hierarchies.
- Stateful episodic behavior is supported via ``reset``.
- Anything approach-specific that evaluators may need (pose estimates,
  deviation scores, predicted states) travels in ``Action.info``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from navbench.core.scenario import Scenario
from navbench.core.types import Action, Observation


class Agent(ABC):
    """Port: an episodic decision-maker for navigation."""

    @abstractmethod
    def reset(self, scenario: Scenario, seed: int) -> None:
        """Prepare for a new episode of the given scenario.

        Implementations must be deterministic w.r.t. ``seed`` so benchmark
        runs are reproducible.
        """

    @abstractmethod
    def act(self, observation: Observation) -> Action:
        """Produce the next action for the current observation."""

    def close(self) -> None:  # noqa: B027 - intentional no-op hook
        """Release resources (models, subprocesses). Optional override."""

    def load_checkpoint(self, path: Path) -> None:
        """Load trained weights/state. Optional override for trainable agents."""
        raise NotImplementedError(f"{type(self).__name__} does not support checkpoints")