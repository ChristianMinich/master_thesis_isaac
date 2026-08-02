"""Autonomous random-walk command generator: a keyboard-free way to drive the Go2.

Picks a new random forward+yaw command every few seconds and holds it, so the
robot wanders in smooth, varied arcs rather than snapping direction every frame.
Two safety overrides:

- Near an outer wall (only if the scene reports bounds -- see `bounds_half_extent`):
  steers toward the room's center (a proportional heading controller, not a
  fixed turn direction -- a fixed-direction turn can trap the robot spinning in
  a tight circle right at the wall if it happens to be facing the wrong way).
- Stuck anywhere else (furniture, an obstacle, a corner the wall check can't
  see): if net displacement over the last few seconds is near zero, stop and
  turn in place for a bit, then resume. This is a general catch-all that
  doesn't need to know where every obstacle in the scene is, and is the only
  safety net at all in an unbounded scene (bounds_half_extent=None).
"""

from __future__ import annotations

import math
import random

import isaacsim.core.experimental.utils.transform as transform_utils
from isaacsim.core.deprecation_manager import import_module

FORWARD_SPEED_RANGE = (0.5, 1.0)  # m/s, always forward (no reversing)
# Kept small relative to forward speed so the robot travels in broad sweeping arcs
# across the scene rather than spinning in tight circles in one spot.
YAW_SPEED_RANGE = (-0.25, 0.25)  # rad/s
MIN_HOLD_S = 3.0
MAX_HOLD_S = 7.0
WALL_MARGIN = 0.8  # meters from a wall at which avoidance kicks in
AVOID_YAW_SPEED = 1.0  # rad/s while turning away from a wall
AVOID_FORWARD_SPEED = 0.3  # m/s while turning away from a wall

# General reactive fallback for anything the wall-proximity check can't see (furniture,
# obstacles, corners, or an unbounded scene): if the robot hasn't actually moved in this
# many seconds, it's wedged against something -- stop and turn in place, then resume.
STUCK_WINDOW_S = 3.0
STUCK_DISTANCE_THRESHOLD = 0.15  # meters of net displacement over the window
RECOVERY_DURATION_S = 2.0
RECOVERY_YAW_SPEED = 1.2  # rad/s, turn-in-place while recovering


def _wrap_to_pi(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


class RandomWalkController:
    """Generates a continuously-varying [v_x, v_y, w_z] command with wall + stuck avoidance."""

    def __init__(
        self,
        command_device: str,
        go2_controller,
        bounds_half_extent: tuple[float, float] | None = None,
        seed: int | None = None,
    ) -> None:
        self._device = command_device
        self._go2 = go2_controller
        self._bounds_half_extent = bounds_half_extent
        self._rng = random.Random(seed)
        self._time_until_change = 0.0
        self._command = [0.0, 0.0, 0.0]
        self._elapsed = 0.0
        self._position_history: list[tuple[float, float, float]] = []
        self._recovery_time_left = 0.0
        self._recovery_yaw = 0.0
        self._pick_new_command()

    def _pick_new_command(self) -> None:
        v_x = self._rng.uniform(*FORWARD_SPEED_RANGE)
        w_z = self._rng.uniform(*YAW_SPEED_RANGE)
        self._command = [v_x, 0.0, w_z]
        self._time_until_change = self._rng.uniform(MIN_HOLD_S, MAX_HOLD_S)

    def _current_pose_2d(self) -> tuple[float, float, float]:
        """Return (x, y, yaw) of the robot base in world frame."""
        position, orientation = self._go2.policy.robot.get_world_poses()
        pos = position.numpy()[0]
        rot = transform_utils.quaternion_to_rotation_matrix(orientation).numpy()[0]
        yaw = math.atan2(rot[1, 0], rot[0, 0])
        return float(pos[0]), float(pos[1]), yaw

    def _is_stuck(self, x: float, y: float) -> bool:
        """True if net displacement over the trailing window is below threshold."""
        if not self._position_history:
            return False
        oldest_t, oldest_x, oldest_y = self._position_history[0]
        if self._elapsed - oldest_t < STUCK_WINDOW_S:
            return False
        return math.hypot(x - oldest_x, y - oldest_y) < STUCK_DISTANCE_THRESHOLD

    def _near_wall(self, x: float, y: float) -> bool:
        if self._bounds_half_extent is None:
            return False
        half_x, half_y = self._bounds_half_extent
        margin = WALL_MARGIN
        return x > half_x - margin or x < -half_x + margin or y > half_y - margin or y < -half_y + margin

    def update(self, dt: float):
        """Advance the internal timer by dt and return the current command tensor."""
        torch = import_module("torch")

        self._elapsed += dt
        self._time_until_change -= dt
        if self._time_until_change <= 0.0:
            self._pick_new_command()

        x, y, yaw = self._current_pose_2d()

        if self._recovery_time_left > 0.0:
            self._recovery_time_left -= dt
            command = [0.0, 0.0, self._recovery_yaw]
        elif self._is_stuck(x, y):
            self._recovery_time_left = RECOVERY_DURATION_S
            self._recovery_yaw = self._rng.choice([-1.0, 1.0]) * RECOVERY_YAW_SPEED
            self._position_history.clear()
            self._time_until_change = 0.0  # force a fresh random command once recovery ends
            command = [0.0, 0.0, self._recovery_yaw]
        elif self._near_wall(x, y):
            # Proportional heading controller toward the scene's center: turn rate is
            # driven by the actual heading error, so it converges and stops (rather
            # than a fixed turn direction, which can trap the robot circling in place
            # right at the wall if it happens to already be facing the "wrong" way).
            target_yaw = math.atan2(-y, -x)
            yaw_error = _wrap_to_pi(target_yaw - yaw)
            w_z = max(-AVOID_YAW_SPEED, min(AVOID_YAW_SPEED, 2.0 * yaw_error))
            command = [AVOID_FORWARD_SPEED, 0.0, w_z]
            self._time_until_change = 0.0  # force a fresh random command once clear of the wall
        else:
            command = self._command

        self._position_history.append((self._elapsed, x, y))
        self._position_history = [p for p in self._position_history if self._elapsed - p[0] <= STUCK_WINDOW_S]

        return torch.tensor(command, device=self._device)
