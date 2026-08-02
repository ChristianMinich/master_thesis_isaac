"""Shared high-level action interface.

Specification reference: section 3 ("Shared high-level action interface") and
section 8.7 ("Actions and policy state").

Every compared approach — expert planner, OneVLA, PPO, V-JEPA 2-AC, LeWM,
human teleoperation — emits exactly this 5-D action at 5 Hz. The frozen Go2
locomotion controller is the only consumer. No approach may bypass it, which is
what makes the comparison in spec section 14 fair.

Canonical field order (used by every ``5D`` array in the dataset)::

    [cmd_vx_mps, cmd_vy_mps, cmd_yaw_rate_rps, inspect_trigger, terminate_trigger]

Recommended ranges (spec section 3):

======================  ===========================
Field                   Range
======================  ===========================
``cmd_vx_mps``          ``[-0.8, 0.8]`` m/s
``cmd_vy_mps``          ``[-0.4, 0.4]`` m/s
``cmd_yaw_rate_rps``    ``[-1.0, 1.0]`` rad/s
``inspect_trigger``     ``[0, 1]`` probability or binary
``terminate_trigger``   ``{0, 1}`` binary
======================  ===========================

The continuous form is the canonical stored action because it can be quantized
later, while discrete actions cannot reconstruct the original velocity command
(spec section 3).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence

import numpy as np

ACTION_DIM = 5
"""Dimensionality of the shared high-level action vector."""

ACTION_FIELDS: tuple[str, ...] = (
    "cmd_vx_mps",
    "cmd_vy_mps",
    "cmd_yaw_rate_rps",
    "inspect_trigger",
    "terminate_trigger",
)
"""Canonical column order for all ``5D`` action arrays and parquet tables."""


@dataclass(frozen=True)
class ActionLimits:
    """Velocity and trigger limits shared by every approach (spec section 14.3).

    The same limits are applied to every policy, so no approach can win by
    commanding a faster base than another.
    """

    vx_min: float = -0.8
    vx_max: float = 0.8
    vy_min: float = -0.4
    vy_max: float = 0.4
    yaw_rate_min: float = -1.0
    yaw_rate_max: float = 1.0

    def to_dict(self) -> dict[str, float]:
        """Return a JSON-serializable mapping for ``action_definition.json``."""
        return {
            "cmd_vx_mps_min": self.vx_min,
            "cmd_vx_mps_max": self.vx_max,
            "cmd_vy_mps_min": self.vy_min,
            "cmd_vy_mps_max": self.vy_max,
            "cmd_yaw_rate_rps_min": self.yaw_rate_min,
            "cmd_yaw_rate_rps_max": self.yaw_rate_max,
        }


class DiscreteAction(Enum):
    """Discrete action set for the discrete-action comparison (spec section 3).

    Derived from the continuous command; never stored as the canonical action.
    """

    FORWARD = "FORWARD"
    TURN_LEFT = "TURN_LEFT"
    TURN_RIGHT = "TURN_RIGHT"
    STOP_AND_INSPECT = "STOP_AND_INSPECT"
    TERMINATE = "TERMINATE"


class SafetyOverrideReason(Enum):
    """Why a LiDAR safety shield modified the requested action (spec 8.7)."""

    NONE = "none"
    COLLISION = "collision"
    INSTABILITY = "instability"
    VELOCITY_LIMIT = "velocity_limit"


@dataclass(frozen=True)
class HighLevelAction:
    """One high-level command emitted at the 5 Hz decision rate.

    Attributes:
        cmd_vx_mps: Forward/backward base velocity command.
        cmd_vy_mps: Lateral base velocity command.
        cmd_yaw_rate_rps: Yaw angular velocity command.
        inspect_trigger: Request for a stable inspection capture, in ``[0, 1]``.
        terminate_trigger: Declaration that the task is finished, in ``{0, 1}``.
    """

    cmd_vx_mps: float = 0.0
    cmd_vy_mps: float = 0.0
    cmd_yaw_rate_rps: float = 0.0
    inspect_trigger: float = 0.0
    terminate_trigger: float = 0.0

    @staticmethod
    def zero() -> HighLevelAction:
        """Return the neutral action (stand still, no trigger)."""
        return HighLevelAction()

    @staticmethod
    def from_array(values: Sequence[float]) -> HighLevelAction:
        """Build an action from a ``5D`` sequence in :data:`ACTION_FIELDS` order."""
        if len(values) != ACTION_DIM:
            raise ValueError(f"action must have {ACTION_DIM} entries, got {len(values)}")
        return HighLevelAction(
            float(values[0]),
            float(values[1]),
            float(values[2]),
            float(values[3]),
            float(values[4]),
        )

    def as_array(self) -> np.ndarray:
        """Return the ``5D`` float32 array in :data:`ACTION_FIELDS` order."""
        return np.array(
            [
                self.cmd_vx_mps,
                self.cmd_vy_mps,
                self.cmd_yaw_rate_rps,
                self.inspect_trigger,
                self.terminate_trigger,
            ],
            dtype=np.float32,
        )

    def to_dict(self) -> dict[str, float]:
        """Return a mapping keyed by :data:`ACTION_FIELDS`."""
        return dict(zip(ACTION_FIELDS, (float(v) for v in self.as_array())))

    @property
    def wants_inspection(self) -> bool:
        """Whether the inspection trigger crosses the 0.5 decision threshold."""
        return self.inspect_trigger >= 0.5

    @property
    def wants_termination(self) -> bool:
        """Whether the terminate trigger crosses the 0.5 decision threshold."""
        return self.terminate_trigger >= 0.5

    def clipped(self, limits: ActionLimits) -> HighLevelAction:
        """Return this action clipped into ``limits`` with triggers in ``[0, 1]``."""
        return HighLevelAction(
            float(np.clip(self.cmd_vx_mps, limits.vx_min, limits.vx_max)),
            float(np.clip(self.cmd_vy_mps, limits.vy_min, limits.vy_max)),
            float(np.clip(self.cmd_yaw_rate_rps, limits.yaw_rate_min, limits.yaw_rate_max)),
            float(np.clip(self.inspect_trigger, 0.0, 1.0)),
            float(np.clip(self.terminate_trigger, 0.0, 1.0)),
        )

    def scaled(self, factor: float) -> HighLevelAction:
        """Return the action with velocities scaled, triggers untouched.

        Used by the safety shield to slow the base down without changing the
        policy's semantic intent.
        """
        return HighLevelAction(
            self.cmd_vx_mps * factor,
            self.cmd_vy_mps * factor,
            self.cmd_yaw_rate_rps * factor,
            self.inspect_trigger,
            self.terminate_trigger,
        )

    def to_discrete(self, forward_threshold: float = 0.1) -> DiscreteAction:
        """Quantize to the discrete action set (spec section 3).

        Priority follows task semantics: termination outranks inspection, which
        outranks locomotion.
        """
        if self.wants_termination:
            return DiscreteAction.TERMINATE
        if self.wants_inspection:
            return DiscreteAction.STOP_AND_INSPECT
        if abs(self.cmd_yaw_rate_rps) > abs(self.cmd_vx_mps):
            return (
                DiscreteAction.TURN_LEFT
                if self.cmd_yaw_rate_rps > 0.0
                else DiscreteAction.TURN_RIGHT
            )
        if self.cmd_vx_mps > forward_threshold:
            return DiscreteAction.FORWARD
        return DiscreteAction.STOP_AND_INSPECT


@dataclass(frozen=True)
class ActionRecord:
    """A fully audited decision step (spec section 8.7).

    Stores the raw policy output *and* the command that actually reached the
    locomotion layer, so that safety interventions remain visible in the data.
    """

    timestamp_ns: int
    decision_index: int
    frame_index: int
    requested: HighLevelAction
    executed: HighLevelAction
    previous: HighLevelAction
    source: str
    safety_override: bool = False
    safety_override_reason: SafetyOverrideReason = SafetyOverrideReason.NONE
    action_log_probability: float | None = None
    value_estimate: float | None = None