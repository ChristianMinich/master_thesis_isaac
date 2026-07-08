"""MockSimulatorAdapter: deterministic kinematic point-robot world.

Purpose: let the *entire* benchmark pipeline (plugins, runner, evaluator,
tests, CI) execute on machines without Isaac Sim / Isaac Lab installed.

Model:
- The robot is a point in the XY plane with heading; Action.command is
  interpreted as [v_x, v_y, yaw_rate] (scenario default action space).
- Reward is negative goal distance delta plus a success bonus.
- Termination: goal reached within ``goal_radius``; truncation: max_steps.
- Perturbations supported: "actuator_dropout" (zero commands for a window),
  "goal_shift" (move the goal at onset), "sensor_noise" (increase noise).
  Unknown kinds are ignored with a log warning, mirroring how a real adapter
  would degrade gracefully.

Determinism: all randomness comes from a ``random.Random(seed)`` instance
created in ``reset``.
"""

from __future__ import annotations

import logging
import math
import random
from typing import Any, Mapping

from navbench.core.scenario import Scenario
from navbench.core.simulator import SimulatorAdapter
from navbench.core.types import Action, Observation, Pose, StepResult

logger = logging.getLogger(__name__)


class MockSimulatorAdapter(SimulatorAdapter):
    """Kinematic point-robot simulator implementing the SimulatorAdapter port."""

    def __init__(self, step_size: float = 0.1, noise_std: float = 0.0) -> None:
        self._step_size = step_size
        self._base_noise_std = noise_std
        self._scenario: Scenario | None = None
        self._rng: random.Random = random.Random(0)
        self._position: list[float] = [0.0, 0.0, 0.0]
        self._goal: tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._t: int = 0
        self._noise_std: float = noise_std
        self._prev_goal_dist: float = 0.0
        self._closed: bool = False

    # -- SimulatorAdapter port ---------------------------------------------
    def load_scenario(self, scenario: Scenario) -> None:
        self._scenario = scenario

    def reset(self, seed: int) -> Observation:
        if self._scenario is None:
            raise RuntimeError("load_scenario() must be called before reset()")
        self._rng = random.Random(seed)
        self._position = list(self._scenario.spawn_pose)
        self._goal = self._scenario.goal_position
        self._noise_std = self._base_noise_std
        self._t = 0
        self._prev_goal_dist = self._goal_distance()
        self._closed = False
        return self._make_observation()

    def step(self, action: Action) -> StepResult:
        if self._scenario is None:
            raise RuntimeError("load_scenario() must be called before step()")
        self._t += 1
        self._apply_perturbations()

        cmd = list(action.command)[:2] + [0.0, 0.0]  # v_x, v_y (yaw ignored kinematically)
        vx = max(-1.0, min(1.0, float(cmd[0])))
        vy = max(-1.0, min(1.0, float(cmd[1])))
        if self._actuators_disabled():
            vx, vy = 0.0, 0.0

        self._position[0] += vx * self._step_size + self._rng.gauss(0.0, self._noise_std)
        self._position[1] += vy * self._step_size + self._rng.gauss(0.0, self._noise_std)

        dist = self._goal_distance()
        reward = self._prev_goal_dist - dist  # progress reward
        self._prev_goal_dist = dist

        success = dist <= self._scenario.goal_radius
        if success:
            reward += 10.0
        truncated = self._t >= self._scenario.max_steps

        return StepResult(
            observation=self._make_observation(),
            reward=reward,
            terminated=success,
            truncated=truncated and not success,
            info={"success": success, "goal_distance": dist},
        )

    def close(self) -> None:
        self._closed = True

    def capabilities(self) -> Mapping[str, Any]:
        return {
            "sensors": ["robot_pose", "robot_state", "goal_position", "instruction"],
            "perturbations": ["actuator_dropout", "goal_shift", "sensor_noise"],
        }

    # -- internals ------------------------------------------------------------
    def _goal_distance(self) -> float:
        return math.dist(self._position[:2], self._goal[:2])

    def _actuators_disabled(self) -> bool:
        assert self._scenario is not None
        for p in self._scenario.perturbations:
            if p.kind == "actuator_dropout":
                duration = int(p.params.get("duration", 10))
                if p.onset_step <= self._t < p.onset_step + duration:
                    return True
        return False

    def _apply_perturbations(self) -> None:
        assert self._scenario is not None
        for p in self._scenario.perturbations:
            if self._t != p.onset_step:
                continue
            if p.kind == "goal_shift":
                dx = float(p.params.get("dx", 0.0))
                dy = float(p.params.get("dy", 0.0))
                self._goal = (self._goal[0] + dx, self._goal[1] + dy, self._goal[2])
                self._prev_goal_dist = self._goal_distance()
            elif p.kind == "sensor_noise":
                self._noise_std = float(p.params.get("std", 0.05))
            elif p.kind == "actuator_dropout":
                pass  # handled per-step in _actuators_disabled
            else:
                logger.warning(
                    "perturbation.unsupported", extra={"kind": p.kind, "adapter": "mock"}
                )

    def _make_observation(self) -> Observation:
        assert self._scenario is not None
        pose = Pose(position=(self._position[0], self._position[1], self._position[2]))
        dx = self._goal[0] - self._position[0]
        dy = self._goal[1] - self._position[1]
        return Observation(
            time_step=self._t,
            robot_pose=pose,
            robot_state=(self._position[0], self._position[1], dx, dy),
            goal_position=self._goal,
            instruction=self._scenario.instruction,
            extras={"goal_distance": self._goal_distance()},
        )