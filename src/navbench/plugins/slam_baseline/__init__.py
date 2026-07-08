"""SLAM / pose-monitoring baseline plugin.

Navigation: simple goal-seeking proportional controller (spatial baseline).
Spatial capability: a dead-reckoning pose estimator that integrates commanded
velocities and exposes its estimate via ``Action.info["estimated_pose"]`` so
the core ``TrajectoryError`` metric can compare it against ground truth.

TODO(slam): replace the dead-reckoning estimator with a real system, e.g.
ORB-SLAM3 (via pybind/ROS bridge) or a visual-odometry pipeline consuming
``Observation.rgb``/``Observation.depth``. Only this plugin changes; core,
dataset, evaluator, and simulator code stay untouched.
"""

from __future__ import annotations

from typing import Any, Mapping

from navbench.core.agent import Agent
from navbench.core.plugin import ApproachPlugin, register_approach
from navbench.core.scenario import Scenario
from navbench.core.trainer import NoOpTrainer, Trainer
from navbench.core.types import Action, Observation, Pose


class SlamBaselineAgent(Agent):
    """Goal-seeking controller + dead-reckoning pose monitor (composition)."""

    def __init__(self, gain: float = 1.0, step_size: float = 0.1) -> None:
        self._gain = gain
        self._step_size = step_size  # must match simulator control dt mapping
        self._est_position: list[float] = [0.0, 0.0, 0.0]

    def reset(self, scenario: Scenario, seed: int) -> None:
        self._est_position = list(scenario.spawn_pose)

    def act(self, observation: Observation) -> Action:
        goal = observation.goal_position or (0.0, 0.0, 0.0)
        # Controller uses the *estimated* pose (spatial-approach behavior).
        dx = goal[0] - self._est_position[0]
        dy = goal[1] - self._est_position[1]
        vx = max(-1.0, min(1.0, self._gain * dx))
        vy = max(-1.0, min(1.0, self._gain * dy))
        # Dead-reckoning integration (drifts because it ignores noise/slip).
        self._est_position[0] += vx * self._step_size
        self._est_position[1] += vy * self._step_size
        estimated = Pose(position=tuple(self._est_position))  # type: ignore[arg-type]
        return Action(command=(vx, vy, 0.0), info={"estimated_pose": estimated})


@register_approach()
class SlamBaselinePlugin(ApproachPlugin):
    @property
    def name(self) -> str:
        return "slam_baseline"

    def build_agent(self, config: Mapping[str, Any]) -> Agent:
        return SlamBaselineAgent(
            gain=float(config.get("gain", 1.0)),
            step_size=float(config.get("step_size", 0.1)),
        )

    def build_trainer(self, config: Mapping[str, Any]) -> Trainer:
        # SLAM has no neural training phase: configuration/calibration only.
        return NoOpTrainer(approach_name=self.name)

    def describe(self) -> Mapping[str, Any]:
        return {
            "paradigm": "SLAM / spatial pose monitoring",
            "reference": "Campos et al. 2021 (ORB-SLAM3)",
            "modalities": ["robot_state", "goal_position"],
            "trainable": False,
        }