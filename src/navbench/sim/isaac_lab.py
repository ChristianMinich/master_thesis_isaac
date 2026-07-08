"""IsaacLabAdapter: skeleton adapter for NVIDIA Isaac Lab.

Isaac Lab provides GPU-accelerated, vectorized robot-learning environments.
This adapter exposes ONE environment instance through the episodic
SimulatorAdapter port; large-scale vectorized RL training may additionally use
Isaac Lab natively inside the RL plugin's Trainer (still behind this module).

TODO(isaaclab): wiring skeleton — connect real Isaac Lab APIs at the marked
points. Do not add Isaac Lab imports at module level.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping

from navbench.core.scenario import Scenario
from navbench.core.simulator import SimulatorAdapter
from navbench.core.types import Action, Observation, StepResult

logger = logging.getLogger(__name__)


class IsaacLabAdapter(SimulatorAdapter):
    """Adapter for Isaac Lab manager-based / direct RL environments.

    Config keys (``simulator:`` section):
        type: isaac_lab
        task: Isaac Lab task id (e.g. a Go2 navigation task registered
              in the Isaac Lab task registry)
        num_envs: int (this adapter uses env index 0 for episodic rollout)
        device: "cuda:0" | "cpu"
    """

    def __init__(self, config: Mapping[str, Any]) -> None:
        self._config = dict(config)
        self._scenario: Scenario | None = None
        self._env: Any = None
        self._t = 0

    def load_scenario(self, scenario: Scenario) -> None:
        self._scenario = scenario
        if self._env is None:
            # TODO(isaaclab): launch the app and build the environment:
            #   from isaaclab.app import AppLauncher
            #   app_launcher = AppLauncher(headless=True)
            #   import gymnasium as gym
            #   import isaaclab_tasks  # registers tasks
            #   self._env = gym.make(self._config["task"], num_envs=..., ...)
            # Map Scenario fields (scene, spawn/goal, perturbations) onto the
            # task's env_cfg before construction (scene randomization events,
            # command ranges, termination terms).
            raise NotImplementedError(
                "IsaacLabAdapter is a skeleton. Install Isaac Lab and complete the "
                "TODO(isaaclab) markers in navbench/sim/isaac_lab.py."
            )

    def reset(self, seed: int) -> Observation:
        if self._scenario is None:
            raise RuntimeError("load_scenario() must be called before reset()")
        self._t = 0
        # TODO(isaaclab): obs, info = self._env.reset(seed=seed)
        # Convert the tensor observation dict of env 0 into a navbench
        # Observation (robot_state vector, ground-truth pose from sim state).
        raise NotImplementedError("TODO(isaaclab): implement reset")

    def step(self, action: Action) -> StepResult:
        if self._scenario is None:
            raise RuntimeError("load_scenario() must be called before step()")
        self._t += 1
        # TODO(isaaclab): convert Action.command to the env's action tensor,
        # step, then translate (obs, reward, terminated, truncated, info) of
        # env 0 into a StepResult. Apply due Perturbations via env events.
        raise NotImplementedError("TODO(isaaclab): implement step")

    def close(self) -> None:
        # TODO(isaaclab): self._env.close(); shut down the app launcher.
        self._env = None

    def capabilities(self) -> Mapping[str, Any]:
        return {
            "sensors": ["robot_state", "robot_pose", "goal_position"],
            "perturbations": ["sensor_noise", "goal_shift", "layout_change"],
            "vectorized_training": True,
        }