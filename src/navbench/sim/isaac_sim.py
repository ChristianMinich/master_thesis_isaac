"""IsaacSimAdapter: skeleton adapter for NVIDIA Isaac Sim.

Keeps ALL Isaac Sim specifics behind the SimulatorAdapter port. This module
imports Isaac APIs lazily so the repository works on machines without Isaac.

TODO(isaac): This is a wiring skeleton. Connect the real Isaac Sim APIs at the
marked points. Do not add Isaac imports at module level.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping

from navbench.core.scenario import Scenario
from navbench.core.simulator import SimulatorAdapter
from navbench.core.types import Action, Observation, StepResult

logger = logging.getLogger(__name__)


class IsaacSimAdapter(SimulatorAdapter):
    """Adapter for Isaac Sim (photorealistic scenes, synthetic sensors).

    Config keys (``simulator:`` section):
        type: isaac_sim
        headless: bool
        physics_dt: float
        rendering_dt: float
        scene_root: path prefix for USD stages referenced by Scenario.scene
    """

    def __init__(self, config: Mapping[str, Any]) -> None:
        self._config = dict(config)
        self._scenario: Scenario | None = None
        self._app: Any = None
        self._world: Any = None
        self._robot: Any = None
        self._t = 0

    def load_scenario(self, scenario: Scenario) -> None:
        self._scenario = scenario
        if self._app is None:
            # TODO(isaac): start SimulationApp exactly once per process:
            #   from isaacsim import SimulationApp
            #   self._app = SimulationApp({"headless": self._config.get("headless", True)})
            # Then import omni.isaac.core and create the World:
            #   from omni.isaac.core import World
            #   self._world = World(physics_dt=..., rendering_dt=...)
            raise NotImplementedError(
                "IsaacSimAdapter is a skeleton. Install Isaac Sim and complete the "
                "TODO(isaac) markers in navbench/sim/isaac_sim.py."
            )
        # TODO(isaac): open the USD stage for `scenario.scene`, spawn the robot
        # asset for `scenario.robot` (e.g. Unitree Go2 USD) at scenario.spawn_pose,
        # place goal marker / obstacles, and configure sensors (RGB-D camera,
        # ground-truth pose) according to scenario.observation_space.

    def reset(self, seed: int) -> Observation:
        if self._scenario is None:
            raise RuntimeError("load_scenario() must be called before reset()")
        self._t = 0
        # TODO(isaac): seed domain randomization (omni.replicator) with `seed`,
        # call self._world.reset(), read initial sensor frames, and convert them
        # into a navbench Observation (rgb, depth, robot_pose, robot_state, ...).
        raise NotImplementedError("TODO(isaac): implement reset")

    def step(self, action: Action) -> StepResult:
        if self._scenario is None:
            raise RuntimeError("load_scenario() must be called before step()")
        self._t += 1
        # TODO(isaac): map Action.command [v_x, v_y, yaw_rate] onto the robot's
        # controller (e.g. Go2 velocity command interface), step the world,
        # apply due Perturbations (obstacle moves, lighting, sensor noise),
        # compute reward/termination (goal distance, collisions from contact
        # sensors), and return a StepResult with the new Observation.
        raise NotImplementedError("TODO(isaac): implement step")

    def close(self) -> None:
        # TODO(isaac): self._world.stop(); self._app.close()
        self._app = None
        self._world = None

    def capabilities(self) -> Mapping[str, Any]:
        return {
            "sensors": ["rgb", "depth", "robot_pose", "robot_state", "goal_position"],
            "perturbations": [
                "obstacle_move",
                "lighting_change",
                "sensor_noise",
                "goal_shift",
                "layout_change",
            ],
        }