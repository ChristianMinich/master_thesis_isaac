"""Fundamental value types shared across the benchmark.

These are deliberately simple, framework-free dataclasses so that core logic,
plugins, and adapters can exchange data without depending on numpy/torch/Isaac.
Adapters may convert to/from richer array types at the boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class Pose:
    """Robot or camera pose: position (x, y, z) and orientation quaternion (w, x, y, z)."""

    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    orientation: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)


@dataclass(frozen=True)
class Observation:
    """A single synchronized observation delivered to an Agent.

    Modality fields are optional so that approach classes only consume what
    they need (SLAM: rgb/depth + pose GT; VLA: rgb + instruction; RL: state
    vector; WM: sequences of everything).

    ``extras`` carries adapter- or scenario-specific payloads (e.g. lidar,
    contact sensors, semantic masks) without changing this interface.
    """

    time_step: int
    rgb: Any | None = None  # H x W x 3 array-like; adapter-defined concrete type.
    depth: Any | None = None  # H x W array-like.
    robot_pose: Pose | None = None  # Ground-truth pose (simulation-only privilege).
    robot_state: Sequence[float] | None = None  # Proprioceptive / task state vector.
    goal_position: tuple[float, float, float] | None = None
    instruction: str | None = None  # Language / symbolic goal description.
    extras: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Action:
    """A robot command produced by an Agent.

    ``command`` is a flat continuous control vector whose semantics are defined
    by the scenario's action space (e.g. [v_x, v_y, yaw_rate] for a mobile
    base / quadruped high-level controller).

    ``info`` carries approach-specific diagnostics (e.g. predicted pose from a
    SLAM agent, value estimates from RL, predicted next-state error from a WM)
    that evaluators may consume for capability-specific metrics.
    """

    command: Sequence[float]
    info: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StepResult:
    """Result of advancing the simulation by one control step."""

    observation: Observation
    reward: float = 0.0
    terminated: bool = False  # Episode ended by task logic (success / collision).
    truncated: bool = False  # Episode ended by time limit.
    info: Mapping[str, Any] = field(default_factory=dict)