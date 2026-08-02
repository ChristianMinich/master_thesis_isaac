"""Framework-free geometric and state value types.

Specification reference: sections 8.2 (calibration transforms), 8.6
(proprioception), 8.8 (target/viewpoint poses).

Conventions used consistently across the whole pipeline:

* World frame is right-handed, ``z`` up, matching USD's default stage up-axis.
* Quaternions are stored as ``(x, y, z, w)`` in every recorded file, matching
  the spec field name ``imu_orientation_xyzw``. In-memory helpers also use
  ``xyzw`` so no silent reordering can happen.
* ``7D`` pose in the spec means ``[x, y, z, qx, qy, qz, qw]``.
* A ``4x4`` transform ``T_a_b`` maps a point expressed in frame ``b`` into
  frame ``a``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

Vec3 = tuple[float, float, float]
QuatXYZW = tuple[float, float, float, float]


def normalize_quat(q: Sequence[float]) -> QuatXYZW:
    """Return the unit quaternion of ``q`` given as ``(x, y, z, w)``."""
    arr = np.asarray(q, dtype=np.float64)
    norm = float(np.linalg.norm(arr))
    if norm < 1e-12:
        return (0.0, 0.0, 0.0, 1.0)
    arr = arr / norm
    return (float(arr[0]), float(arr[1]), float(arr[2]), float(arr[3]))


def quat_from_yaw(yaw: float) -> QuatXYZW:
    """Return the quaternion of a rotation about ``+z`` by ``yaw`` radians."""
    half = 0.5 * yaw
    return (0.0, 0.0, math.sin(half), math.cos(half))


def yaw_from_quat(q: Sequence[float]) -> float:
    """Return the yaw (rotation about ``+z``) of a quaternion ``(x, y, z, w)``."""
    x, y, z, w = normalize_quat(q)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def quat_from_rpy(roll: float, pitch: float, yaw: float) -> QuatXYZW:
    """Return the quaternion of intrinsic Z-Y-X Euler angles."""
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    return normalize_quat(
        (
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy,
        )
    )


def quat_to_matrix(q: Sequence[float]) -> np.ndarray:
    """Return the ``3x3`` rotation matrix of a quaternion ``(x, y, z, w)``."""
    x, y, z, w = normalize_quat(q)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def matrix_to_quat(matrix: np.ndarray) -> QuatXYZW:
    """Return the quaternion ``(x, y, z, w)`` of a ``3x3`` rotation matrix."""
    m = np.asarray(matrix, dtype=np.float64)[:3, :3]
    trace = float(np.trace(m))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    return normalize_quat((x, y, z, w))


def quat_multiply(a: Sequence[float], b: Sequence[float]) -> QuatXYZW:
    """Return the Hamilton product ``a * b`` of two ``(x, y, z, w)`` quaternions."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return normalize_quat(
        (
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        )
    )


def quat_inverse(q: Sequence[float]) -> QuatXYZW:
    """Return the inverse (conjugate of the unit quaternion) of ``q``."""
    x, y, z, w = normalize_quat(q)
    return (-x, -y, -z, w)


def wrap_angle(angle: float) -> float:
    """Wrap an angle to ``(-pi, pi]``."""
    wrapped = math.fmod(angle + math.pi, 2.0 * math.pi)
    if wrapped <= 0.0:
        wrapped += 2.0 * math.pi
    return wrapped - math.pi


@dataclass(frozen=True)
class Pose:
    """A rigid pose: position in metres plus orientation quaternion ``xyzw``."""

    position: Vec3
    orientation: QuatXYZW = (0.0, 0.0, 0.0, 1.0)

    @staticmethod
    def from_xy_yaw(x: float, y: float, yaw: float, z: float = 0.0) -> Pose:
        """Build a planar pose, the common case for base and viewpoint poses."""
        return Pose((float(x), float(y), float(z)), quat_from_yaw(yaw))

    @staticmethod
    def from_array(values: Sequence[float]) -> Pose:
        """Build a pose from the spec's ``7D`` layout ``[x,y,z,qx,qy,qz,qw]``."""
        if len(values) != 7:
            raise ValueError(f"7D pose expected, got {len(values)} values")
        return Pose(
            (float(values[0]), float(values[1]), float(values[2])),
            normalize_quat(values[3:7]),
        )

    @staticmethod
    def from_matrix(matrix: np.ndarray) -> Pose:
        """Build a pose from a ``4x4`` homogeneous transform."""
        m = np.asarray(matrix, dtype=np.float64)
        return Pose(
            (float(m[0, 3]), float(m[1, 3]), float(m[2, 3])),
            matrix_to_quat(m[:3, :3]),
        )

    @property
    def x(self) -> float:
        """World ``x`` in metres."""
        return self.position[0]

    @property
    def y(self) -> float:
        """World ``y`` in metres."""
        return self.position[1]

    @property
    def z(self) -> float:
        """World ``z`` in metres."""
        return self.position[2]

    @property
    def yaw(self) -> float:
        """Heading about ``+z`` in radians."""
        return yaw_from_quat(self.orientation)

    def as_array(self) -> np.ndarray:
        """Return the spec's ``7D`` representation ``[x,y,z,qx,qy,qz,qw]``."""
        return np.array([*self.position, *self.orientation], dtype=np.float64)

    def as_matrix(self) -> np.ndarray:
        """Return the ``4x4`` homogeneous transform of this pose."""
        m = np.eye(4, dtype=np.float64)
        m[:3, :3] = quat_to_matrix(self.orientation)
        m[:3, 3] = np.asarray(self.position, dtype=np.float64)
        return m

    def inverse(self) -> Pose:
        """Return the inverse transform."""
        rot_t = quat_to_matrix(self.orientation).T
        pos = -rot_t @ np.asarray(self.position, dtype=np.float64)
        return Pose((float(pos[0]), float(pos[1]), float(pos[2])), matrix_to_quat(rot_t))

    def compose(self, other: Pose) -> Pose:
        """Return ``self * other`` (apply ``other`` in this pose's frame)."""
        rot = quat_to_matrix(self.orientation)
        pos = np.asarray(self.position, dtype=np.float64) + rot @ np.asarray(
            other.position, dtype=np.float64
        )
        return Pose(
            (float(pos[0]), float(pos[1]), float(pos[2])),
            quat_multiply(self.orientation, other.orientation),
        )

    def transform_point(self, point: Sequence[float]) -> Vec3:
        """Map a point from this pose's local frame into the parent frame."""
        rot = quat_to_matrix(self.orientation)
        out = np.asarray(self.position, dtype=np.float64) + rot @ np.asarray(
            point, dtype=np.float64
        )
        return (float(out[0]), float(out[1]), float(out[2]))

    def distance_to(self, other: Pose) -> float:
        """Euclidean distance between two pose origins."""
        a = np.asarray(self.position, dtype=np.float64)
        b = np.asarray(other.position, dtype=np.float64)
        return float(np.linalg.norm(a - b))


@dataclass(frozen=True)
class AABB:
    """Axis-aligned bounding box in world coordinates."""

    minimum: Vec3
    maximum: Vec3

    @staticmethod
    def from_points(points: Iterable[Sequence[float]]) -> AABB:
        """Return the tight AABB enclosing ``points``."""
        arr = np.asarray(list(points), dtype=np.float64)
        if arr.size == 0:
            raise ValueError("cannot build an AABB from zero points")
        lo = arr.min(axis=0)
        hi = arr.max(axis=0)
        return AABB(tuple(float(v) for v in lo), tuple(float(v) for v in hi))  # type: ignore[arg-type]

    @property
    def center(self) -> Vec3:
        """Box centre."""
        return tuple(  # type: ignore[return-value]
            0.5 * (lo + hi) for lo, hi in zip(self.minimum, self.maximum)
        )

    @property
    def extent(self) -> Vec3:
        """Box side lengths."""
        return tuple(  # type: ignore[return-value]
            hi - lo for lo, hi in zip(self.minimum, self.maximum)
        )

    def as_list(self) -> list[list[float]]:
        """Return ``[[min...],[max...]]`` for JSON serialization."""
        return [list(self.minimum), list(self.maximum)]

    def contains(self, point: Sequence[float]) -> bool:
        """Return whether ``point`` lies inside the box."""
        return all(lo <= v <= hi for lo, v, hi in zip(self.minimum, point, self.maximum))

    def expanded(self, margin: float) -> AABB:
        """Return the box grown by ``margin`` in every direction."""
        return AABB(
            tuple(v - margin for v in self.minimum),  # type: ignore[arg-type]
            tuple(v + margin for v in self.maximum),  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class Twist:
    """Body-frame velocity: linear m/s and angular rad/s."""

    linear: Vec3 = (0.0, 0.0, 0.0)
    angular: Vec3 = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class RobotState:
    """Full Go2 state at one control tick.

    Mirrors spec section 8.6. ``base_pose`` is privileged ground truth
    (``base_pose_world_gt``); observation-side consumers receive only the
    proprioceptive fields.
    """

    timestamp_ns: int
    base_pose: Pose
    base_linear_velocity: Vec3
    base_angular_velocity: Vec3
    projected_gravity: Vec3
    joint_position: np.ndarray
    joint_velocity: np.ndarray
    joint_effort: np.ndarray
    joint_position_target: np.ndarray
    foot_contact: np.ndarray
    foot_contact_force: np.ndarray
    stability_margin: float = 0.0