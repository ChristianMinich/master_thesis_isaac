"""Sensor suite for the Go2: RGB+depth camera, LiDAR, IMU, and per-foot contact sensors.

Every sensor is created as a child prim of the robot's base link, so all of them
move rigidly with the robot without any manual per-frame transform bookkeeping.
Self-contained: all Go2-specific mount offsets/asset tunables live here rather
than in a shared app_config, so this whole robots/go2/ package is one portable unit.
"""

from __future__ import annotations

import numpy as np
from isaacsim.core.experimental.utils.app import enable_extension
from isaacsim.sensors.experimental.physics import IMU, Contact, ContactSensor, IMUSensor
from isaacsim.sensors.experimental.rtx import CameraSensor, Lidar, LidarSensor, RtxCamera, parse_generic_model_output_data

from robots.go2.go2_controller import BASE_LINK_RELATIVE_PATH, DEFAULT_PRIM_PATH

enable_extension("isaacsim.sensors.rtx.nodes")

DEFAULT_BASE_LINK_PATH = f"{DEFAULT_PRIM_PATH}/{BASE_LINK_RELATIVE_PATH}"

# Sensor mount sub-paths, relative to the base link. Defined once here and reused by
# transform_prim_paths() below, so the logging/telemetry path can never drift out of
# sync with where the sensors actually are.
CAMERA_SUBPATH = "rgb_depth_camera"
LIDAR_SUBPATH = "lidar"
IMU_SUBPATH = "imu_sensor"

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
# RtxCamera's aperture defaults to ~2.1x1.6mm (not the standard USD 20.955x15.2908mm
# film back) unless set explicitly -- combined with a 24mm focal length that was an
# unusable ~5 degree FOV (a telephoto lens pointed at whatever's directly ahead, which
# is why the feed showed mostly wall). Explicit standard aperture + a shorter focal
# length gives a normal wide-ish robot nav camera FOV (~80 x 64 degrees).
CAMERA_HORIZONTAL_APERTURE = 20.955  # mm, standard USD/Maya film back
CAMERA_VERTICAL_APERTURE = 15.2908  # mm, standard USD/Maya film back
CAMERA_FOCAL_LENGTH = 12.0
CAMERA_CLIPPING_RANGE = (0.05, 50.0)
# Mounted near the nose, roughly body-height. Forward-facing mount quaternion
# (wxyz): maps camera forward (-Z_cam) -> parent's +X, image-up (+Y_cam) ->
# parent's +Z. Derived analytically from the camera's local axis convention,
# verified by rotating the camera's local forward/up vectors and checking
# they land on +X/+Z (see isaac_unitree_starter dev notes).
CAMERA_LOCAL_POSITION = (0.32, 0.0, 0.06)
CAMERA_LOCAL_ORIENTATION_WXYZ = (-0.5, -0.5, 0.5, 0.5)

LIDAR_CONFIG_NAME = "Example_Rotary"
LIDAR_LOCAL_POSITION = (0.0, 0.0, 0.15)

IMU_LOCAL_POSITION = (0.0, 0.0, 0.0)

# Local offsets/radius of the Go2's foot collision spheres, read directly off
# the loaded USD asset (see isaac_unitree_starter dev notes) -- all four feet
# share the same offset in their respective calf-link local frame.
FOOT_LINK_RELATIVE_PATHS = {
    "FL": "FL_hip/FL_thigh/FL_calf",
    "FR": "FR_hip/FR_thigh/FR_calf",
    "RL": "RL_hip/RL_thigh/RL_calf",
    "RR": "RR_hip/RR_thigh/RR_calf",
}
FOOT_LOCAL_OFFSET = (-0.002, 0.0, -0.213)
FOOT_CONTACT_SENSOR_RADIUS = 0.03
FOOT_CONTACT_MIN_THRESHOLD = 1.0
FOOT_CONTACT_MAX_THRESHOLD = 100000.0


def transform_prim_paths(base_link_path: str) -> dict[str, str]:
    """Prim paths worth logging a world transform for, keyed by a short label.

    Reused by DataLogger so its telemetry columns can never point at the wrong
    prim -- these are the exact sub-paths SensorManager mounts sensors at.
    """
    return {
        "base": base_link_path,
        "camera": f"{base_link_path}/{CAMERA_SUBPATH}",
        "lidar": f"{base_link_path}/{LIDAR_SUBPATH}",
        "imu": f"{base_link_path}/{IMU_SUBPATH}",
    }


class SensorManager:
    """Owns and reads the Go2's RGB/depth camera, LiDAR, IMU, and foot contact sensors."""

    def __init__(self, base_link_path: str = DEFAULT_BASE_LINK_PATH) -> None:
        self._base_link_path = base_link_path

        self.camera = RtxCamera(
            f"{base_link_path}/{CAMERA_SUBPATH}",
            tick_rate=30.0,
            translations=np.array(CAMERA_LOCAL_POSITION),
            orientations=np.array(CAMERA_LOCAL_ORIENTATION_WXYZ),
        )
        self.camera.camera.set_apertures(
            horizontal_apertures=[CAMERA_HORIZONTAL_APERTURE],
            vertical_apertures=[CAMERA_VERTICAL_APERTURE],
        )
        self.camera.camera.set_focal_lengths(CAMERA_FOCAL_LENGTH)
        self.camera.camera.set_clipping_ranges(*CAMERA_CLIPPING_RANGE)
        self.camera_sensor = CameraSensor(
            self.camera,
            resolution=(CAMERA_HEIGHT, CAMERA_WIDTH),
            annotators=["rgb", "distance_to_image_plane"],
        )

        self.lidar = Lidar.create(
            f"{base_link_path}/{LIDAR_SUBPATH}",
            config=LIDAR_CONFIG_NAME,
            translations=np.array(LIDAR_LOCAL_POSITION),
        )
        # No "draw-point-cloud" debug-draw writer here: that draws into the shared
        # Hydra scene, so it shows up in EVERY render product looking at the scene --
        # including our own RGB camera sensor, not just the main viewport. Live LiDAR
        # visualization instead goes through live_viewer.py, which rasterizes points
        # into their own separate 2D window with no connection to any 3D camera.
        self.lidar_sensor = LidarSensor(self.lidar, annotators=["generic-model-output"])

        self.imu_sensor = IMUSensor(
            IMU.create(
                f"{base_link_path}/{IMU_SUBPATH}",
                translations=np.array([list(IMU_LOCAL_POSITION)]),
            )
        )

        # NOTE: with PHYSICS_DEVICE="cuda" (the official Go2 example's setting, kept here to
        # match it exactly), PhysX's direct-GPU pipeline does not populate legacy contact-report
        # callbacks, so these sensors are wired correctly but will read force=0/in_contact=False
        # on every step. Confirmed by a side-by-side run with physics on CPU, where the same
        # sensors report real per-foot forces (~30-50N, matching the robot's weight). Kept anyway
        # so the recorded data has the field (honestly zero) rather than omitting it.
        self.foot_contact_sensors: dict[str, ContactSensor] = {}
        for foot_name, relative_path in FOOT_LINK_RELATIVE_PATHS.items():
            sensor_path = f"{base_link_path}/{relative_path}/{foot_name}_contact_sensor"
            self.foot_contact_sensors[foot_name] = ContactSensor(
                Contact.create(
                    sensor_path,
                    min_threshold=FOOT_CONTACT_MIN_THRESHOLD,
                    max_threshold=FOOT_CONTACT_MAX_THRESHOLD,
                    radius=FOOT_CONTACT_SENSOR_RADIUS,
                    translations=np.array([list(FOOT_LOCAL_OFFSET)]),
                )
            )

    def read_camera(self) -> dict:
        """Return {'rgb': HxWx4 uint8 array or None, 'depth': HxW float32 array or None}."""
        rgb_data, _ = self.camera_sensor.get_data("rgb")
        depth_data, _ = self.camera_sensor.get_data("distance_to_image_plane")
        return {
            "rgb": rgb_data.numpy() if rgb_data is not None else None,
            "depth": depth_data.numpy() if depth_data is not None else None,
        }

    def read_lidar(self) -> dict:
        """Return the current LiDAR scan as Cartesian points + range + intensity."""
        raw, _ = self.lidar_sensor.get_data("generic-model-output")
        gmo = parse_generic_model_output_data(raw)
        n = int(gmo.numElements)
        if n == 0:
            return {
                "points": np.zeros((0, 3), dtype=np.float32),
                "range": np.zeros((0,), dtype=np.float32),
                "intensity": np.zeros((0,), dtype=np.float32),
                "num_points": 0,
            }
        x = np.asarray(gmo.x[:n], dtype=np.float32)
        y = np.asarray(gmo.y[:n], dtype=np.float32)
        z = np.asarray(gmo.z[:n], dtype=np.float32)
        points = np.stack([x, y, z], axis=-1)
        range_m = np.sqrt(x**2 + y**2 + z**2).astype(np.float32)
        scalar = np.asarray(gmo.scalar, dtype=np.float32) if gmo.scalar is not None else None
        intensity = scalar[:n] if scalar is not None and len(scalar) >= n else np.zeros((n,), dtype=np.float32)
        return {"points": points, "range": range_m, "intensity": intensity, "num_points": n}

    def read_imu(self) -> dict:
        """Return {'linear_acceleration', 'angular_velocity', 'orientation' [w,x,y,z], 'time', 'physics_step'}."""
        return self.imu_sensor.get_data()

    def read_foot_contacts(self) -> dict:
        """Return {foot_name: {'in_contact': bool, 'force': float}} for each of the four feet."""
        return {name: sensor.get_data() for name, sensor in self.foot_contact_sensors.items()}

    def read_all(self) -> dict:
        """Convenience accessor: gather camera, LiDAR, IMU, and foot-contact readings in one call."""
        return {
            "camera": self.read_camera(),
            "lidar": self.read_lidar(),
            "imu": self.read_imu(),
            "foot_contacts": self.read_foot_contacts(),
        }
