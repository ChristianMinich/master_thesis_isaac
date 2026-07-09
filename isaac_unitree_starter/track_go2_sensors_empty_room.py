"""Track Go2 sensor data (actuators, camera, LiDAR) in an empty room.

Builds a minimal empty room (floor + four walls, no obstacles), imports the
Unitree Go2 from the NVIDIA Omniverse content server:

    https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.0/Isaac/Robots/Unitree/Go2/go2.usd

and records the following streams every simulation step:

    Actuators (articulation)
    - joint_positions       (n_dof,)   float32  rad
    - joint_velocities      (n_dof,)   float32  rad/s
    - joint_efforts         (n_dof,)   float32  N*m (measured, if supported)
    Camera (front-facing, follows the robot base)
    - rgb                   (H, W, 3)  uint8
    - depth                 (H, W)     float32  distance to image plane (m)
    LiDAR (planar 360 deg scan, follows the robot base)
    - lidar_ranges          (N_RAYS,)  float32  ranges (m)
    - lidar_angles          (N_RAYS,)  float32  ray azimuths (rad, robot frame)
    Base state
    - base_position         (3,)       float32  world frame
    - base_orientation      (4,)       float32  quaternion (w, x, y, z)
    - sim_time              ()         float32

The whole recording is saved as one compressed .npz (arrays stacked over time)
plus a metadata.json describing shapes, sensor configuration, and joint names.

Sensors that cannot be created on a given Isaac Sim install (APIs are version
dependent) degrade gracefully to zero-filled arrays with a printed warning,
so the output format stays fixed.

Run with the Isaac Sim python interpreter, e.g.:
    ./python.sh isaac_unitree_starter/track_go2_sensors_empty_room.py --headless
"""

import argparse

# -- Launch Isaac Sim first (must happen before any other omni/isaac imports) --
parser = argparse.ArgumentParser(description="Track Go2 actuator/camera/LiDAR data in an empty room.")
parser.add_argument("--headless", action="store_true", help="Run without the GUI.")
parser.add_argument("--steps", type=int, default=300, help="Number of recorded simulation steps.")
parser.add_argument("--warmup-steps", type=int, default=60, help="Sim steps before recording starts.")
parser.add_argument("--output-dir", type=str, default="data/go2_sensor_tracking", help="Output directory.")
parser.add_argument("--print-every", type=int, default=25, help="Print a status line every N steps.")
args_cli = parser.parse_args()

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": args_cli.headless})

import json
import math
import os

import numpy as np
import omni.usd
from pxr import Gf, UsdGeom, UsdLux

from isaacsim.core.api import World
from isaacsim.core.api.objects import FixedCuboid
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.sensors.camera import Camera

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PHYSICS_DT = 1.0 / 200.0
RENDER_DT = 1.0 / 60.0

# Empty room (floor + walls only, no obstacles)
ROOM_SIZE = 10.0
WALL_HEIGHT = 1.0
WALL_THICKNESS = 0.2

# Go2 asset streamed from the Omniverse content server
GO2_USD_PATH = (
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com"
    "/Assets/Isaac/6.0/Isaac/Robots/Unitree/Go2/go2.usd"
)
GO2_PRIM_PATH = "/World/Go2"
GO2_SPAWN_POSITION = (0.0, 0.0, 0.45)

# Sensor rig: kinematic Xform carrying camera + LiDAR; re-posed to the robot
# base every step so the sensors follow the robot.
RIG_PRIM_PATH = "/World/SensorRig"
RIG_HEIGHT = 0.45

# Camera
CAMERA_PRIM_PATH = f"{RIG_PRIM_PATH}/FrontCamera"
CAMERA_OFFSET = (0.3, 0.0, 0.1)     # forward of the base, slightly above (m)
CAMERA_RESOLUTION = (320, 240)      # (width, height)

# LiDAR (planar 360 deg scan)
LIDAR_PRIM_PATH = f"{RIG_PRIM_PATH}/Lidar"
LIDAR_OFFSET = (0.1, 0.0, 0.15)
LIDAR_NUM_RAYS = 360
LIDAR_MAX_RANGE = 20.0
LIDAR_MIN_RANGE = 0.1

# Fallback DOF count if the articulation cannot be wrapped (Go2 has 12 joints)
NUM_DOF_FALLBACK = 12

NPZ_KEYS = [
    "joint_positions", "joint_velocities", "joint_efforts",
    "rgb", "depth",
    "lidar_ranges", "lidar_angles",
    "base_position", "base_orientation",
    "sim_time",
]


# ---------------------------------------------------------------------------
# Scene construction
# ---------------------------------------------------------------------------
def add_lighting(stage):
    dome = UsdLux.DomeLight.Define(stage, "/World/Lights/DomeLight")
    dome.CreateIntensityAttr(1000.0)

    distant = UsdLux.DistantLight.Define(stage, "/World/Lights/DistantLight")
    distant.CreateIntensityAttr(3000.0)
    UsdGeom.XformCommonAPI(distant.GetPrim()).SetRotate(Gf.Vec3f(-45.0, 0.0, 0.0))


def add_room_walls(world: World):
    """Empty room: just four walls around the ground plane."""
    half = ROOM_SIZE / 2.0
    wall_specs = [
        {"name": "wall_north", "pos": (0.0, half), "scale": (ROOM_SIZE + WALL_THICKNESS, WALL_THICKNESS, WALL_HEIGHT)},
        {"name": "wall_south", "pos": (0.0, -half), "scale": (ROOM_SIZE + WALL_THICKNESS, WALL_THICKNESS, WALL_HEIGHT)},
        {"name": "wall_east", "pos": (half, 0.0), "scale": (WALL_THICKNESS, ROOM_SIZE + WALL_THICKNESS, WALL_HEIGHT)},
        {"name": "wall_west", "pos": (-half, 0.0), "scale": (WALL_THICKNESS, ROOM_SIZE + WALL_THICKNESS, WALL_HEIGHT)},
    ]
    for spec in wall_specs:
        x, y = spec["pos"]
        sx, sy, sz = spec["scale"]
        world.scene.add(
            FixedCuboid(
                prim_path=f"/World/Room/{spec['name']}",
                name=spec["name"],
                position=np.array([x, y, sz / 2.0]),
                scale=np.array([sx, sy, sz]),
                color=np.array([0.5, 0.5, 0.5]),
            )
        )


def add_go2(world: World):
    """Reference the Go2 asset and wrap it as an articulation for actuator data."""
    add_reference_to_stage(usd_path=GO2_USD_PATH, prim_path=GO2_PRIM_PATH)
    prim = world.stage.GetPrimAtPath(GO2_PRIM_PATH)
    if not prim.IsValid():
        raise RuntimeError(f"Go2 was not created at {GO2_PRIM_PATH}")
    UsdGeom.XformCommonAPI(prim).SetTranslate(Gf.Vec3d(*GO2_SPAWN_POSITION))
    print(f"[INFO] Loaded Go2 from: {GO2_USD_PATH}")

    try:
        from isaacsim.core.prims import SingleArticulation

        robot = SingleArticulation(prim_path=GO2_PRIM_PATH, name="go2")
        world.scene.add(robot)
        return robot
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] Could not wrap Go2 as articulation ({exc}). Actuator channels will be zeros.")
        return None


# ---------------------------------------------------------------------------
# Sensor rig
# ---------------------------------------------------------------------------
def add_sensor_rig(stage):
    """Create the kinematic Xform that carries camera + LiDAR."""
    rig = UsdGeom.Xform.Define(stage, RIG_PRIM_PATH)
    api = UsdGeom.XformCommonAPI(rig.GetPrim())
    api.SetTranslate(Gf.Vec3d(GO2_SPAWN_POSITION[0], GO2_SPAWN_POSITION[1], RIG_HEIGHT))
    return rig


def set_rig_pose(rig_prim, x: float, y: float, z: float, yaw: float):
    api = UsdGeom.XformCommonAPI(rig_prim)
    api.SetTranslate(Gf.Vec3d(x, y, z))
    api.SetRotate(Gf.Vec3f(0.0, 0.0, math.degrees(yaw)))


def add_camera() -> Camera:
    camera = Camera(
        prim_path=CAMERA_PRIM_PATH,
        translation=np.array(CAMERA_OFFSET),
        resolution=CAMERA_RESOLUTION,
    )
    return camera


def add_lidar(world: World):
    """Create a planar rotating LiDAR. Returns None if unavailable on this install."""
    try:
        from isaacsim.sensors.physx import RotatingLidarPhysX

        lidar = RotatingLidarPhysX(
            prim_path=LIDAR_PRIM_PATH,
            name="go2_lidar",
            translation=np.array(LIDAR_OFFSET),
            rotation_frequency=0.0,  # 0 -> full scan every step
            fov=(360.0, 0.0),        # planar scan
            resolution=(360.0 / LIDAR_NUM_RAYS, 1.0),
            valid_range=(LIDAR_MIN_RANGE, LIDAR_MAX_RANGE),
        )
        world.scene.add(lidar)
        lidar.add_depth_data_to_frame()
        print("[INFO] LiDAR sensor created.")
        return lidar
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] Could not create LiDAR ({exc}). lidar_ranges will be zeros.")
        return None


# ---------------------------------------------------------------------------
# Sensor readout helpers (fixed output shapes, graceful fallbacks)
# ---------------------------------------------------------------------------
def read_actuators(robot):
    """Joint positions / velocities / (measured) efforts for all DOFs."""
    if robot is not None:
        try:
            q = np.asarray(robot.get_joint_positions(), dtype=np.float32).reshape(-1)
            dq = np.asarray(robot.get_joint_velocities(), dtype=np.float32).reshape(-1)
            tau = None
            try:
                tau = robot.get_measured_joint_efforts()
            except Exception:  # noqa: BLE001
                try:
                    tau = robot.get_applied_joint_efforts()
                except Exception:  # noqa: BLE001
                    tau = None
            if tau is None:
                tau = np.zeros_like(q)
            tau = np.asarray(tau, dtype=np.float32).reshape(-1)
            return q, dq, tau
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] Actuator read failed ({exc}).")
    zeros = np.zeros(NUM_DOF_FALLBACK, dtype=np.float32)
    return zeros.copy(), zeros.copy(), zeros.copy()


def read_camera(camera: Camera):
    rgb = None
    depth = None
    try:
        rgba = camera.get_rgba()
        if rgba is not None and np.asarray(rgba).size > 0:
            rgb = np.asarray(rgba)[..., :3].astype(np.uint8)
        frame = camera.get_current_frame()
        d = frame.get("distance_to_image_plane")
        if d is not None:
            depth = np.asarray(d, dtype=np.float32)
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] Camera read failed ({exc}).")
    h, w = CAMERA_RESOLUTION[1], CAMERA_RESOLUTION[0]
    if rgb is None:
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
    if depth is None:
        depth = np.zeros((h, w), dtype=np.float32)
    return rgb, depth


def read_lidar(lidar) -> np.ndarray:
    ranges = np.zeros(LIDAR_NUM_RAYS, dtype=np.float32)
    if lidar is None:
        return ranges
    try:
        frame = lidar.get_current_frame()
        depth = frame.get("depth")
        if depth is not None:
            flat = np.asarray(depth, dtype=np.float32).reshape(-1)
            n = min(flat.shape[0], LIDAR_NUM_RAYS)
            ranges[:n] = flat[:n]
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] LiDAR read failed ({exc}).")
    return ranges


def read_base_pose(robot):
    position = np.array(GO2_SPAWN_POSITION, dtype=np.float32)
    orientation = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)  # (w, x, y, z)
    if robot is not None:
        try:
            pos, quat = robot.get_world_pose()
            position = np.asarray(pos, dtype=np.float32).reshape(3)
            orientation = np.asarray(quat, dtype=np.float32).reshape(4)
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] Base pose read failed ({exc}).")
    return position, orientation


def yaw_from_quat(quat_wxyz: np.ndarray) -> float:
    w, x, y, z = (float(v) for v in quat_wxyz)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


# ---------------------------------------------------------------------------
# Data writing
# ---------------------------------------------------------------------------
def save_recording(buffers: dict, output_dir: str) -> str:
    data = {key: np.stack(vals, axis=0) for key, vals in buffers.items()}
    out_path = os.path.join(output_dir, "sensor_log.npz")
    np.savez_compressed(out_path, **data)
    print(f"[INFO] Saved recording: {data['rgb'].shape[0]} steps -> {out_path}")
    return out_path


def write_metadata(output_dir: str, joint_names, buffers: dict) -> None:
    metadata = {
        "go2_usd_path": GO2_USD_PATH,
        "robot_spawn_position": list(GO2_SPAWN_POSITION),
        "room": {
            "size": ROOM_SIZE,
            "wall_height": WALL_HEIGHT,
            "wall_thickness": WALL_THICKNESS,
            "obstacles": [],
        },
        "sensors": {
            "camera": {"resolution": list(CAMERA_RESOLUTION), "offset": list(CAMERA_OFFSET)},
            "lidar": {
                "num_rays": LIDAR_NUM_RAYS,
                "max_range": LIDAR_MAX_RANGE,
                "min_range": LIDAR_MIN_RANGE,
                "offset": list(LIDAR_OFFSET),
            },
            "actuators": {"joint_names": joint_names},
        },
        "physics_dt": PHYSICS_DT,
        "render_dt": RENDER_DT,
        "steps": args_cli.steps,
        "warmup_steps": args_cli.warmup_steps,
        "npz_keys": NPZ_KEYS,
        "shapes": {key: list(np.stack(vals, axis=0).shape) for key, vals in buffers.items()},
    }
    out_path = os.path.join(output_dir, "metadata.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"[INFO] Wrote metadata -> {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    output_dir = args_cli.output_dir
    os.makedirs(output_dir, exist_ok=True)

    world = World(physics_dt=PHYSICS_DT, rendering_dt=RENDER_DT, stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()

    stage = omni.usd.get_context().get_stage()
    add_lighting(stage)
    add_room_walls(world)
    robot = add_go2(world)

    rig = add_sensor_rig(stage)
    rig_prim = rig.GetPrim()
    camera = add_camera()
    lidar = add_lidar(world)

    world.reset()
    camera.initialize()
    camera.add_distance_to_image_plane_to_frame()

    joint_names = []
    if robot is not None:
        try:
            joint_names = list(robot.dof_names)
            print(f"[INFO] Go2 articulation ready: {len(joint_names)} DOFs -> {joint_names}")
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] Could not read joint names ({exc}).")

    lidar_angles = np.linspace(-math.pi, math.pi, LIDAR_NUM_RAYS, endpoint=False).astype(np.float32)

    # let physics settle before recording
    print(f"[INFO] Warming up for {args_cli.warmup_steps} steps ...")
    for _ in range(args_cli.warmup_steps):
        world.step(render=True)

    buffers = {key: [] for key in NPZ_KEYS}
    print(f"[INFO] Recording {args_cli.steps} steps ...")

    for i in range(args_cli.steps):
        # keep the sensor rig glued to the robot base
        base_pos, base_quat = read_base_pose(robot)
        set_rig_pose(
            rig_prim,
            float(base_pos[0]),
            float(base_pos[1]),
            float(base_pos[2]) + (RIG_HEIGHT - GO2_SPAWN_POSITION[2]),
            yaw_from_quat(base_quat),
        )

        world.step(render=True)

        q, dq, tau = read_actuators(robot)
        rgb, depth = read_camera(camera)
        ranges = read_lidar(lidar)

        buffers["joint_positions"].append(q)
        buffers["joint_velocities"].append(dq)
        buffers["joint_efforts"].append(tau)
        buffers["rgb"].append(rgb)
        buffers["depth"].append(depth)
        buffers["lidar_ranges"].append(ranges)
        buffers["lidar_angles"].append(lidar_angles)
        buffers["base_position"].append(base_pos)
        buffers["base_orientation"].append(base_quat)
        buffers["sim_time"].append(np.float32(world.current_time))

        if args_cli.print_every > 0 and i % args_cli.print_every == 0:
            valid = ranges[(ranges > LIDAR_MIN_RANGE) & (ranges < LIDAR_MAX_RANGE)]
            min_range = float(valid.min()) if valid.size else float("nan")
            print(
                f"[INFO] step {i:4d} | t={float(world.current_time):6.2f}s"
                f" | base z={float(base_pos[2]):.3f} m"
                f" | |q|={float(np.linalg.norm(q)):.3f}"
                f" | |dq|={float(np.linalg.norm(dq)):.3f}"
                f" | rgb mean={float(rgb.mean()):.1f}"
                f" | lidar min={min_range:.2f} m"
            )

    save_recording(buffers, output_dir)
    write_metadata(output_dir, joint_names, buffers)
    print(f"[INFO] Done. Sensor data written to {output_dir}/")

    simulation_app.close()


if __name__ == "__main__":
    main()