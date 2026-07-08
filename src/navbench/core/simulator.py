"""SimulatorAdapter port: the only way core/plugins may talk to a simulator.

Concrete adapters (MockSimulatorAdapter, IsaacSimAdapter, IsaacLabAdapter)
live in ``navbench.sim``. Core logic depends only on this abstract port
(dependency inversion), so the whole benchmark runs without Isaac installed.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping

from navbench.core.scenario import Scenario
from navbench.core.types import Action, Observation, StepResult


class SimulatorAdapter(ABC):
    """Port: episodic simulation access for one scenario at a time.

    Lifecycle::

        adapter.load_scenario(scenario)
        obs = adapter.reset(seed=...)
        while True:
            result = adapter.step(action)
            if result.terminated or result.truncated:
                break
        adapter.close()

    Adapters are responsible for:
    - interpreting the declarative Scenario (scene, robot, sensors),
    - applying Perturbations at their onset steps,
    - computing task rewards and termination (goal reached, collision),
    - exposing ground-truth state in ``Observation`` where the scenario
      permits (simulation-only privilege used for evaluation).
    """

    @abstractmethod
    def load_scenario(self, scenario: Scenario) -> None:
        """Build / configure the simulated world for a scenario."""

    @abstractmethod
    def reset(self, seed: int) -> Observation:
        """Reset the environment deterministically and return the first observation."""

    @abstractmethod
    def step(self, action: Action) -> StepResult:
        """Advance the simulation by one control step."""

    @abstractmethod
    def close(self) -> None:
        """Shut down the simulation and free resources."""

    def capabilities(self) -> Mapping[str, Any]:
        """Optional: declare supported sensors / perturbation kinds.

        Runners may use this to fail fast when a scenario requires a
        modality the adapter cannot provide.
        """
        return {}