"""Scenario and perturbation definitions.

A Scenario is a declarative description of one navigation task variant:
layout, spawn/goal, instruction, sensor configuration, and perturbations.
Scenarios are data, not behavior — simulator adapters interpret them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class Perturbation:
    """A controlled disturbance injected into an episode.

    Examples (see exposé §4.4): obstacle displacement, lighting change,
    sensor noise, localization drift injection, goal/instruction mismatch.

    ``kind`` is an open string identifier interpreted by the simulator
    adapter; new perturbation types therefore do not require core changes.
    """

    kind: str
    onset_step: int = 0
    params: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Scenario:
    """One reproducible navigation task variant.

    ``scene`` names a scene asset / layout known to the simulator adapter
    (e.g. a USD stage path for Isaac Sim, or a mock grid layout id).

    ``robot`` names the robot embodiment (e.g. "unitree_go2").

    ``action_space`` / ``observation_space`` are declarative specs (dicts)
    that adapters and plugins use to configure themselves; keeping them as
    data avoids coupling core to any specific gym/Isaac space classes.
    """

    scenario_id: str
    scene: str
    robot: str = "unitree_go2"
    spawn_pose: tuple[float, float, float] = (0.0, 0.0, 0.0)
    goal_position: tuple[float, float, float] = (5.0, 5.0, 0.0)
    goal_radius: float = 0.5
    instruction: str | None = None
    max_steps: int = 500
    perturbations: Sequence[Perturbation] = field(default_factory=tuple)
    action_space: Mapping[str, Any] = field(
        default_factory=lambda: {"type": "continuous", "dim": 3, "low": -1.0, "high": 1.0}
    )
    observation_space: Mapping[str, Any] = field(
        default_factory=lambda: {"modalities": ["robot_state", "robot_pose", "goal_position"]}
    )
    tags: Sequence[str] = field(default_factory=tuple)  # e.g. ("nominal",) or ("perturbed",)

    @property
    def is_nominal(self) -> bool:
        return len(self.perturbations) == 0