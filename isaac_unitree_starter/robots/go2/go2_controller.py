"""Thin wrapper around the official Go2FlatTerrainPolicy controller.

Keeps initialization/reset state-machine logic in one place so callers'
physics callbacks stay simple.
"""

from __future__ import annotations

from isaacsim.robot.policy.examples.robots import Go2FlatTerrainPolicy

DEFAULT_PRIM_PATH = "/World/Go2"
DEFAULT_SPAWN_POSITION = [0.0, 0.0, 0.5]
# Relative to the robot's own prim_path -- part of the Mujoco Menagerie Go2 asset's
# internal USD structure, not something a caller should need to know about.
BASE_LINK_RELATIVE_PATH = "Geometry/base"


class Go2Controller:
    """Owns the Go2 articulation + locomotion policy and its ready/init state."""

    def __init__(self, prim_path: str = DEFAULT_PRIM_PATH, position=None) -> None:
        self._prim_path = prim_path
        self._position = position or DEFAULT_SPAWN_POSITION
        self.policy: Go2FlatTerrainPolicy | None = None
        self._ready = False

    def spawn(self) -> None:
        """Create the Go2 articulation + load its locomotion policy on the stage."""
        self.policy = Go2FlatTerrainPolicy(
            prim_path=self._prim_path,
            position=self._position,
        )

    def on_physics_step(self, dt: float, command) -> None:
        """Advance the robot by one physics step.

        Initializes the articulation on the first valid physics tensor step (mirrors the
        official interactive Go2 example), then forwards the command through the policy.
        """
        if self.policy is None:
            return

        if not self.policy.robot.is_physics_tensor_entity_valid():
            self._ready = False

        if self._ready:
            self.policy.forward(dt, command)
        else:
            self._ready = True
            self.policy.initialize()
            self.policy.post_reset()

    @property
    def ready(self) -> bool:
        return self._ready

    @property
    def base_link_path(self) -> str:
        """World-space prim path of the robot's base link (parent of all mounted sensors)."""
        return f"{self._prim_path}/{BASE_LINK_RELATIVE_PATH}"

    def get_last_action(self):
        """Return the policy's most recently computed 12-dim joint action tensor, or None."""
        if self.policy is None:
            return None
        return self.policy._current_action
