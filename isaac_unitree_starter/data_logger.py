"""Timestamped-session recorder: RGB (PNG), depth (NPY), LiDAR (NPZ), and a
per-step telemetry CSV covering joints, base state, IMU, foot contacts,
command, policy action, and sensor/robot transforms.

Generic: doesn't know it's logging a Go2 specifically -- dof names, foot names,
transform prim paths, and any extra session metadata are all supplied by the
caller (see robots/go2/go2_robot.py), so this logger could record any robot
whose controller/sensors expose the same read_* shapes.

Recording is off by default; toggled on/off via KeyboardController's record key
(see main.py), which calls DataLogger.toggle().
"""

from __future__ import annotations

import csv
import json
import os
from datetime import datetime

import cv2
import numpy as np
import omni.usd
from pxr import Usd, UsdGeom

DEFAULT_RECORDINGS_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "recordings")


def _world_transform(stage, prim_path: str):
    """Return (position [x,y,z], orientation [w,x,y,z]) for a prim's world transform."""
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return [0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]
    matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    translation = matrix.ExtractTranslation()
    quat = matrix.ExtractRotationQuat()
    imaginary = quat.GetImaginary()
    return (
        [translation[0], translation[1], translation[2]],
        [quat.GetReal(), imaginary[0], imaginary[1], imaginary[2]],
    )


class DataLogger:
    """Owns a single timestamped recording session at a time."""

    def __init__(
        self,
        dof_names: list[str],
        foot_names: list[str],
        transform_prim_paths: dict[str, str],
        session_root: str = DEFAULT_RECORDINGS_ROOT,
        extra_metadata: dict | None = None,
    ) -> None:
        self._dof_names = list(dof_names)
        self._foot_names = list(foot_names)
        self._transform_prim_paths = dict(transform_prim_paths)
        self._session_root = session_root
        self._extra_metadata = dict(extra_metadata) if extra_metadata else {}
        self.is_recording = False

        self._session_dir = None
        self._rgb_dir = None
        self._depth_dir = None
        self._lidar_dir = None
        self._csv_file = None
        self._csv_writer = None
        self._step_in_session = 0

    def toggle(self) -> None:
        """Start a new session if idle, or close the current one if recording."""
        if self.is_recording:
            self.stop()
        else:
            self.start()

    def start(self) -> None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._session_dir = os.path.join(self._session_root, f"session_{timestamp}")
        self._rgb_dir = os.path.join(self._session_dir, "rgb")
        self._depth_dir = os.path.join(self._session_dir, "depth")
        self._lidar_dir = os.path.join(self._session_dir, "lidar")
        for directory in (self._session_dir, self._rgb_dir, self._depth_dir, self._lidar_dir):
            os.makedirs(directory, exist_ok=True)

        metadata = {
            "started_at": timestamp,
            "dof_names": self._dof_names,
            "foot_names": self._foot_names,
            "transform_prim_paths": self._transform_prim_paths,
            **self._extra_metadata,
        }
        with open(os.path.join(self._session_dir, "session_metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        self._csv_file = open(os.path.join(self._session_dir, "telemetry.csv"), "w", newline="")
        self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=self._build_fieldnames())
        self._csv_writer.writeheader()

        self._step_in_session = 0
        self.is_recording = True
        print(f">>> [logger] recording started: {self._session_dir}", flush=True)

    def stop(self) -> None:
        if self._csv_file is not None:
            self._csv_file.close()
            self._csv_file = None
        self._csv_writer = None
        self.is_recording = False
        print(">>> [logger] recording stopped", flush=True)

    def _build_fieldnames(self) -> list[str]:
        names = ["step", "sim_time", "physics_step", "rgb_file", "depth_file", "lidar_file"]
        names += ["cmd_vx", "cmd_vy", "cmd_wz"]
        names += [f"action_{i}" for i in range(12)]
        names += [
            "base_pos_x", "base_pos_y", "base_pos_z",
            "base_quat_w", "base_quat_x", "base_quat_y", "base_quat_z",
            "base_linvel_x", "base_linvel_y", "base_linvel_z",
            "base_angvel_x", "base_angvel_y", "base_angvel_z",
        ]
        for name in self._dof_names:
            names.append(f"joint_pos_{name}")
        for name in self._dof_names:
            names.append(f"joint_vel_{name}")
        for name in self._dof_names:
            names.append(f"joint_effort_{name}")
        names += [
            "imu_linacc_x", "imu_linacc_y", "imu_linacc_z",
            "imu_angvel_x", "imu_angvel_y", "imu_angvel_z",
            "imu_quat_w", "imu_quat_x", "imu_quat_y", "imu_quat_z",
        ]
        for foot in self._foot_names:
            names += [f"foot_{foot}_in_contact", f"foot_{foot}_force"]
        for sensor_name in self._transform_prim_paths:
            names += [
                f"{sensor_name}_tf_pos_x", f"{sensor_name}_tf_pos_y", f"{sensor_name}_tf_pos_z",
                f"{sensor_name}_tf_quat_w", f"{sensor_name}_tf_quat_x",
                f"{sensor_name}_tf_quat_y", f"{sensor_name}_tf_quat_z",
            ]
        return names

    def log_step(self, *, step: int, sim_time: float, physics_step: int, go2, sensors, command, action) -> None:
        """Record one frame of sensor + robot telemetry. No-op if not currently recording."""
        if not self.is_recording:
            return

        camera_reading = sensors.read_camera()
        lidar_reading = sensors.read_lidar()
        imu_reading = sensors.read_imu()
        contact_readings = sensors.read_foot_contacts()

        rgb_file = depth_file = lidar_file = ""
        if camera_reading["rgb"] is not None:
            rgb_file = f"rgb_{self._step_in_session:06d}.png"
            rgb_bgr = cv2.cvtColor(camera_reading["rgb"], cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(self._rgb_dir, rgb_file), rgb_bgr)
        if camera_reading["depth"] is not None:
            depth_file = f"depth_{self._step_in_session:06d}.npy"
            np.save(os.path.join(self._depth_dir, depth_file), camera_reading["depth"])
        if lidar_reading["num_points"] > 0:
            lidar_file = f"lidar_{self._step_in_session:06d}.npz"
            np.savez(
                os.path.join(self._lidar_dir, lidar_file),
                points=lidar_reading["points"],
                range=lidar_reading["range"],
                intensity=lidar_reading["intensity"],
            )

        base_pos_wp, base_quat_wp = go2.policy.robot.get_world_poses()
        base_linvel_wp, base_angvel_wp = go2.policy.robot.get_velocities()
        joint_pos = go2.policy.robot.get_dof_positions().numpy()[0]
        joint_vel = go2.policy.robot.get_dof_velocities().numpy()[0]
        joint_effort = go2.policy.robot.get_dof_efforts().numpy()[0]

        base_pos = base_pos_wp.numpy()[0]
        base_quat = base_quat_wp.numpy()[0]
        base_linvel = base_linvel_wp.numpy()[0]
        base_angvel = base_angvel_wp.numpy()[0]

        command_np = command.detach().cpu().numpy()
        action_np = action.detach().cpu().numpy() if action is not None else np.zeros(12)

        row = {
            "step": step,
            "sim_time": sim_time,
            "physics_step": physics_step,
            "rgb_file": rgb_file,
            "depth_file": depth_file,
            "lidar_file": lidar_file,
            "cmd_vx": float(command_np[0]),
            "cmd_vy": float(command_np[1]),
            "cmd_wz": float(command_np[2]),
        }
        for i in range(12):
            row[f"action_{i}"] = float(action_np[i])

        row.update(
            {
                "base_pos_x": float(base_pos[0]), "base_pos_y": float(base_pos[1]), "base_pos_z": float(base_pos[2]),
                "base_quat_w": float(base_quat[0]), "base_quat_x": float(base_quat[1]),
                "base_quat_y": float(base_quat[2]), "base_quat_z": float(base_quat[3]),
                "base_linvel_x": float(base_linvel[0]), "base_linvel_y": float(base_linvel[1]),
                "base_linvel_z": float(base_linvel[2]),
                "base_angvel_x": float(base_angvel[0]), "base_angvel_y": float(base_angvel[1]),
                "base_angvel_z": float(base_angvel[2]),
            }
        )

        for i, name in enumerate(self._dof_names):
            row[f"joint_pos_{name}"] = float(joint_pos[i])
            row[f"joint_vel_{name}"] = float(joint_vel[i])
            row[f"joint_effort_{name}"] = float(joint_effort[i])

        lin_acc = imu_reading["linear_acceleration"]
        ang_vel = imu_reading["angular_velocity"]
        imu_quat = imu_reading["orientation"]
        row.update(
            {
                "imu_linacc_x": float(lin_acc[0]), "imu_linacc_y": float(lin_acc[1]), "imu_linacc_z": float(lin_acc[2]),
                "imu_angvel_x": float(ang_vel[0]), "imu_angvel_y": float(ang_vel[1]), "imu_angvel_z": float(ang_vel[2]),
                "imu_quat_w": float(imu_quat[0]), "imu_quat_x": float(imu_quat[1]),
                "imu_quat_y": float(imu_quat[2]), "imu_quat_z": float(imu_quat[3]),
            }
        )

        for foot, reading in contact_readings.items():
            row[f"foot_{foot}_in_contact"] = bool(reading["in_contact"])
            row[f"foot_{foot}_force"] = float(reading["force"])

        stage = omni.usd.get_context().get_stage()
        for sensor_name, prim_path in self._transform_prim_paths.items():
            tf_pos, tf_quat = _world_transform(stage, prim_path)
            row[f"{sensor_name}_tf_pos_x"] = tf_pos[0]
            row[f"{sensor_name}_tf_pos_y"] = tf_pos[1]
            row[f"{sensor_name}_tf_pos_z"] = tf_pos[2]
            row[f"{sensor_name}_tf_quat_w"] = tf_quat[0]
            row[f"{sensor_name}_tf_quat_x"] = tf_quat[1]
            row[f"{sensor_name}_tf_quat_y"] = tf_quat[2]
            row[f"{sensor_name}_tf_quat_z"] = tf_quat[3]

        self._csv_writer.writerow(row)
        self._csv_file.flush()
        self._step_in_session += 1
