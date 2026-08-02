"""Sensor calibration and sensor models.

Specification reference: section 8.2 (``camera_intrinsics``, ``camera_distortion``,
``T_base_camera``, ``T_base_lidar``, ``T_base_imu``, ``sensor_time_offset_ns``,
``exposure_time_s``) and sections 8.3/8.4 (sensor configuration fields).

All extrinsics are expressed as ``T_base_sensor``: a 4x4 transform mapping a
point from the sensor frame into the Go2 base frame. Camera frames follow the
optical convention ``+x`` right, ``+y`` down, ``+z`` forward.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from navbench.domain.types import Pose, quat_from_rpy


def _camera_optical_from_base(pitch_rad: float = 0.0, yaw_rad: float = 0.0) -> np.ndarray:
    """Return the rotation of an optical camera frame relative to the base frame.

    The base frame is ``+x`` forward, ``+y`` left, ``+z`` up. The optical frame
    is ``+z`` forward, ``+x`` right, ``+y`` down. ``pitch_rad`` tilts the camera
    downwards for positive values; ``yaw_rad`` rotates it to the left.
    """
    # Columns are the optical axes expressed in the base frame.
    base_from_optical = np.array(
        [
            [0.0, 0.0, 1.0],  # optical +z -> base +x (forward)
            [-1.0, 0.0, 0.0],  # optical +x -> base -y (right)
            [0.0, -1.0, 0.0],  # optical +y -> base -z (down)
        ],
        dtype=np.float64,
    )
    cp, sp = math.cos(pitch_rad), math.sin(pitch_rad)
    pitch = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=np.float64)
    cy, sy = math.cos(yaw_rad), math.sin(yaw_rad)
    yaw = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return yaw @ pitch @ base_from_optical


@dataclass
class CameraCalibration:
    """Pinhole camera intrinsics, distortion and extrinsics (spec 8.2/8.3)."""

    name: str
    width: int
    height: int
    horizontal_fov_deg: float
    mount_xyz: tuple[float, float, float]
    mount_pitch_deg: float = 0.0
    mount_yaw_deg: float = 0.0
    near_clip_m: float = 0.05
    far_clip_m: float = 40.0
    distortion_model: str = "plumb_bob"
    distortion_coeffs: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0)
    exposure_time_s: float = 0.008
    time_offset_ns: int = 0
    rolling_shutter_params: dict[str, Any] | None = None

    @property
    def fx(self) -> float:
        """Focal length in pixels along ``x``, derived from the horizontal FOV."""
        return 0.5 * self.width / math.tan(0.5 * math.radians(self.horizontal_fov_deg))

    @property
    def fy(self) -> float:
        """Focal length in pixels along ``y`` (square pixels)."""
        return self.fx

    @property
    def cx(self) -> float:
        """Principal point ``x`` in pixels."""
        return 0.5 * self.width

    @property
    def cy(self) -> float:
        """Principal point ``y`` in pixels."""
        return 0.5 * self.height

    @property
    def vertical_fov_deg(self) -> float:
        """Vertical field of view implied by the intrinsics."""
        return math.degrees(2.0 * math.atan(0.5 * self.height / self.fy))

    def intrinsics_matrix(self) -> np.ndarray:
        """Return the ``3x3`` intrinsics matrix ``K``."""
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    def T_base_sensor(self) -> np.ndarray:
        """Return the ``4x4`` extrinsic transform ``T_base_camera``."""
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, :3] = _camera_optical_from_base(
            math.radians(self.mount_pitch_deg), math.radians(self.mount_yaw_deg)
        )
        matrix[:3, 3] = np.asarray(self.mount_xyz, dtype=np.float64)
        return matrix

    def pose_in_base(self) -> Pose:
        """Return the camera pose in the base frame."""
        return Pose.from_matrix(self.T_base_sensor())

    def to_dict(self) -> dict[str, Any]:
        """Return the ``calibration.json`` camera block."""
        return {
            "name": self.name,
            "width": self.width,
            "height": self.height,
            "model": "pinhole",
            "fx": self.fx,
            "fy": self.fy,
            "cx": self.cx,
            "cy": self.cy,
            "camera_intrinsics": self.intrinsics_matrix().tolist(),
            "camera_distortion": {
                "model": self.distortion_model,
                "coefficients": list(self.distortion_coeffs),
            },
            "horizontal_fov_deg": self.horizontal_fov_deg,
            "vertical_fov_deg": self.vertical_fov_deg,
            "near_clip_m": self.near_clip_m,
            "far_clip_m": self.far_clip_m,
            "T_base_camera": self.T_base_sensor().tolist(),
            "exposure_time_s": self.exposure_time_s,
            "sensor_time_offset_ns": self.time_offset_ns,
            "rolling_shutter_params": self.rolling_shutter_params,
        }


@dataclass
class LidarCalibration:
    """3D LiDAR configuration and extrinsics (spec 8.2/8.4)."""

    name: str = "lidar_top"
    channels: int = 16
    points_per_channel: int = 1024
    vfov_deg: tuple[float, float] = (-15.0, 15.0)
    hfov_deg: tuple[float, float] = (-180.0, 180.0)
    min_range_m: float = 0.2
    max_range_m: float = 50.0
    mount_xyz: tuple[float, float, float] = (0.10, 0.0, 0.22)
    mount_roll_deg: float = 0.0
    mount_pitch_deg: float = 0.0
    mount_yaw_deg: float = 0.0
    time_offset_ns: int = 0
    scan_duration_s: float = 0.1

    @property
    def range_image_shape(self) -> tuple[int, int]:
        """Shape ``(R, A)`` of ``lidar_range_image_m`` (spec 8.4)."""
        return (self.channels, self.points_per_channel)

    def elevation_angles_rad(self) -> np.ndarray:
        """Return the per-ring elevation angles in radians."""
        return np.linspace(
            math.radians(self.vfov_deg[0]),
            math.radians(self.vfov_deg[1]),
            self.channels,
            dtype=np.float64,
        )

    def azimuth_angles_rad(self) -> np.ndarray:
        """Return the per-column azimuth angles in radians (endpoint excluded)."""
        return np.linspace(
            math.radians(self.hfov_deg[0]),
            math.radians(self.hfov_deg[1]),
            self.points_per_channel,
            endpoint=False,
            dtype=np.float64,
        )

    def T_base_sensor(self) -> np.ndarray:
        """Return the ``4x4`` extrinsic transform ``T_base_lidar``."""
        matrix = np.eye(4, dtype=np.float64)
        quat = quat_from_rpy(
            math.radians(self.mount_roll_deg),
            math.radians(self.mount_pitch_deg),
            math.radians(self.mount_yaw_deg),
        )
        matrix[:3, :3] = Pose((0.0, 0.0, 0.0), quat).as_matrix()[:3, :3]
        matrix[:3, 3] = np.asarray(self.mount_xyz, dtype=np.float64)
        return matrix

    def pose_in_base(self) -> Pose:
        """Return the LiDAR pose in the base frame."""
        return Pose.from_matrix(self.T_base_sensor())

    def to_dict(self) -> dict[str, Any]:
        """Return the ``calibration.json`` LiDAR block."""
        return {
            "name": self.name,
            "channels": self.channels,
            "points_per_channel": self.points_per_channel,
            "range_image_shape": list(self.range_image_shape),
            "vfov_deg": list(self.vfov_deg),
            "hfov_deg": list(self.hfov_deg),
            "lidar_min_range_m": self.min_range_m,
            "lidar_max_range_m": self.max_range_m,
            "scan_duration_s": self.scan_duration_s,
            "T_base_lidar": self.T_base_sensor().tolist(),
            "sensor_time_offset_ns": self.time_offset_ns,
        }


@dataclass
class ImuCalibration:
    """IMU mounting and noise model (spec 8.2/8.5)."""

    name: str = "imu_base"
    mount_xyz: tuple[float, float, float] = (0.0, 0.0, 0.02)
    accel_noise_density: float = 0.02
    gyro_noise_density: float = 0.002
    accel_bias_sigma: float = 0.05
    gyro_bias_sigma: float = 0.005
    time_offset_ns: int = 0

    def T_base_sensor(self) -> np.ndarray:
        """Return the ``4x4`` extrinsic transform ``T_base_imu``."""
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, 3] = np.asarray(self.mount_xyz, dtype=np.float64)
        return matrix

    def to_dict(self) -> dict[str, Any]:
        """Return the ``calibration.json`` IMU block."""
        return {
            "name": self.name,
            "T_base_imu": self.T_base_sensor().tolist(),
            "accel_noise_density": self.accel_noise_density,
            "gyro_noise_density": self.gyro_noise_density,
            "accel_bias_sigma": self.accel_bias_sigma,
            "gyro_bias_sigma": self.gyro_bias_sigma,
            "sensor_time_offset_ns": self.time_offset_ns,
        }


@dataclass
class SensorRig:
    """The complete Go2 sensor rig written to ``calibration.json``."""

    cameras: dict[str, CameraCalibration] = field(default_factory=dict)
    lidar: LidarCalibration = field(default_factory=LidarCalibration)
    imu: ImuCalibration = field(default_factory=ImuCalibration)
    base_height_m: float = 0.32

    @staticmethod
    def go2_default(
        width: int = 640,
        height: int = 480,
        multi_view: bool = True,
    ) -> SensorRig:
        """Return the default Go2 rig: front RGB-D, optional side views, LiDAR, IMU.

        Mount positions follow the Go2 geometry: head camera at ``x=0.28 m``,
        ``z=0.10 m`` above the base origin, LiDAR on top at ``z=0.22 m``.
        """
        cameras = {
            "rgb_front": CameraCalibration(
                name="rgb_front",
                width=width,
                height=height,
                horizontal_fov_deg=87.0,
                mount_xyz=(0.28, 0.0, 0.10),
                mount_pitch_deg=8.0,
            )
        }
        if multi_view:
            cameras["rgb_left"] = CameraCalibration(
                name="rgb_left",
                width=width,
                height=height,
                horizontal_fov_deg=87.0,
                mount_xyz=(0.18, 0.06, 0.10),
                mount_pitch_deg=8.0,
                mount_yaw_deg=55.0,
            )
            cameras["rgb_right"] = CameraCalibration(
                name="rgb_right",
                width=width,
                height=height,
                horizontal_fov_deg=87.0,
                mount_xyz=(0.18, -0.06, 0.10),
                mount_pitch_deg=8.0,
                mount_yaw_deg=-55.0,
            )
        return SensorRig(cameras=cameras)

    @property
    def front_camera(self) -> CameraCalibration:
        """The main perception camera (``rgb_front``)."""
        return self.cameras["rgb_front"]

    def to_dict(self) -> dict[str, Any]:
        """Return the ``calibration.json`` payload."""
        return {
            "frame_conventions": {
                "world_up_axis": "z",
                "quaternion_order": "xyzw",
                "base_frame": "go2_base",
                "camera_frame": "optical_x_right_y_down_z_forward",
                "extrinsics_meaning": "T_base_sensor maps sensor-frame points into the base frame",
            },
            "base_height_m": self.base_height_m,
            "cameras": {name: cam.to_dict() for name, cam in sorted(self.cameras.items())},
            "lidar": self.lidar.to_dict(),
            "imu": self.imu.to_dict(),
        }