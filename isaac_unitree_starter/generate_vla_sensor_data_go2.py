"""Generate multi-sensor synthetic VLA training data for the Go2 in the room scene.

Builds the room-with-obstacles scene (same layout as create_room_scene.py) and
performs a scripted kinematic traversal from the robot spawn to the green goal
marker. Every recorded step captures the full sensor suite a VLA policy for
the Go2 would consume, plus the action and language annotation it would be
trained to predict:

    Vision
    - rgb                 (H, W, 3)  uint8    forward-facing camera
    - depth               (H, W)     float32  distance to image plane (m)
    LiDAR
    - lidar_ranges        (N_RAYS,)  float32  planar 360 deg scan (m)
    - lidar_angles        (N_RAYS,)  float32  ray azimuths (rad, robot frame)
    IMU
    - imu_lin_acc         (3,)       float32  linear acceleration (m/s^2)
    - imu_ang_vel         (3,)       float32  angular velocity (rad/s)
    - imu_orientation     (4,)       float32  quaternion (w, x, y, z)
    Proprioception
    - joint_positions     (n_dof,)   float32
    - joint_velocities    (n_dof,)   float32
    State / goal
    - base_position       (3,)       float32  world frame
    - base_yaw            ()         float32  heading (rad)
    - goal_position       (3,)       float32  world frame
    - goal_vector_body    (3,)       float32  goal in robot (body) frame
    - distance_to_goal    ()         float32
    Action (VLA target: base velocity command)
    - action              (3,)       float32  (v_x, v_y, w_z) body frame
    - sim_time            ()         float32

Each episode is saved as one compressed .npz file (arrays stacked over time),
and a dataset_metadata.json stores the language instruction, scene layout,
sensor configuration, and key list. Episodes randomize the trajectory with a
seeded RNG so the dataset has some diversity.

Sensors that cannot be created on a given Isaac Sim install (APIs are version
dependent) degrade gracefully to zero-filled arrays with a printed warning,
so the output format stays fixed.

Run with the Isaac Sim python interpreter, e.g.:
    ./python.sh isaac_unitree_starter/generate_vla_sensor_data_go2.py --headless
"""

import argparse

# -- Launch Isaac Sim first (must happen before any other omni/isaac imports) --
parser = argparse.ArgumentParser(description="Generate multi-sensor VLA data for the Go2 room scene.")
parser.add_argument("--headless", action="store_true", help="Run without the GUI.")
parser.add_argument("--num-episodes", type=int, default=3, help="Number of episodes to record.")
parser.add_argument("--steps-per-episode", type=int, default=200, help="Recorded steps per episode.")
parser.add_argument("--output-dir", type=str, default="data/vla_sensors_go2", help="Output directory.")
parser.add_argument("--seed", type=int, default=42, help="RNG seed for trajectory randomization.")
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
from isaacsim.core.api.objects import FixedCuboid, VisualCylinder
from isaacsim.core.utils.rotations import euler_angles_to_quat
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.sensors.camera import Camera
from isaacsim.storage.native import get_assets_root_path

# ---------------------------------------------------------------------------
# Inline configuration (scene layout mirrors create_room_scene.py)
# ---------------------------------------------------------------------------
PHYSICS_DT = 1.0 / 200.0
RENDER_DT = 1.0 / 60.0

# Room layout
ROOM_SIZE = 10.0
WALL_HEIGHT = 1.0
WALL_THICKNESS = 0.2

# Go2 USD asset. Adjust if your Isaac Sim / Nucleus version stores it elsewhere.
GO2_RELATIVE_PATH = "/Isaac/Robots/Unitree/Go2/go2.usd"  # <-- placeholder, version dependent
GO2_PRIM_PATH = "/World/Go2"
GO2_SPAWN_POSITION = (-3.5, -3.5, 0.45)

# Goal marker (visual only, no collision)
GOAL_POSITION = (3.5, 3.5)
GOAL_RADIUS = 0.3
GOAL_HEIGHT = 0.05

# Static box obstacles inside the room (position xy, size).
OBSTACLE_SPECS = [
    {"name": "block_1", "pos": (0.0, 0.0), "size": 0.6},
    {"name": "block_2", "pos": (-1.5, 1.5), "size": 0.5},
    {"name": "block_3", "pos": (1.5, -1.0), "size": 0.4},
    {"name": "block_4", "pos": (2.0, 1.5), "size": 0.5},
    {"name": "block_5", "pos": (-1.0, -2.0), "size": 0.4},
    {"name": "block_6", "pos": (0.5, 2.5), "size": 0.3},
]

# Sensor rig (kinematic mount carrying camera / lidar / imu, moved along the
# trajectory together with the Go2 root)
RIG_PRIM_PATH = "/World/SensorRig"
RIG_HEIGHT = 0.45              # base height above ground (Go2 standing height)

# Camera
CAMERA_PRIM_PATH = f"{RIG_PRIM_PATH}/FrontCamera"
CAMERA_OFFSET = (0.3, 0.0, 0.1)     # forward of the base, slightly above (m)
CAMERA_RESOLUTION = (320, 240)      # (width, height)

# LiDAR (planar 360 deg scan, Go2-carried L1-style simplified to 2D)
LIDAR_PRIM_PATH = f"{RIG_PRIM_PATH}/Lidar"
LIDAR_OFFSET = (0.1, 0.0, 0.15)
LIDAR_NUM_RAYS = 360
LIDAR_MAX_RANGE = 20.0
LIDAR_MIN_RANGE = 0.1

# IMU
IMU_PRIM_PATH = f"{RIG_PRIM_PATH}/Imu"
IMU_OFFSET = (0.0, 0.0, 0.0)

# Language annotation for this navigation task (single instruction dataset;
# extend to templated instructions later)
LANGUAGE_INSTRUCTION = "Walk to the green goal marker while avoiding the boxes."

# Trajectory: nominal waypoints (x, y) from spawn to goal, skirting obstacles.
# Per-episode jitter is added to the interior waypoints for dataset diversity.
NOMINAL_WAYPOINTS = [
    (-3.5, -3.5),
    (-2.5, -0.5),
    (-0.8, 0.8),
    (1.0, 2.2),
    (3.5, 3.5),
]
WAYPOINT_JITTER = 0.4          # max abs jitter (m) applied to interior waypoints

WARMUP_STEPS = 30              # sim/render steps before recording each episode


# ---------------------------------------------------------------------------
# Scene construction
# ---------------------------------------------------------------------------
def add_lighting(stage):
    dome = UsdLux.DomeLight.Define(stage, "/World/Lights/DomeLight")
    dome.CreateIntensityAttr(1000.0)

    distant = UsdLux.DistantLight.Define(stage, "/World/Lights/DistantLight")
    distant.CreateIntensityAttr(3000.0)
    distant.CreateAngleAttr(0.5)
    UsdGeom.XformCommonAPI(distant.GetPrim()).SetRotate(Gf.Vec3f(-45.0, 0.0, 0.0))


def add_room_walls(world: World):
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


def add_obstacles(world: World):
    for spec in OBSTACLE_SPECS:
        x, y = spec["pos"]
        s = spec["size"]
        world.scene.add(
            FixedCuboid(
                prim_path=f"/World/Obstacles/{spec['name']}",
                name=spec["name"],
                position=np.array([x, y, s / 2.0]),
                scale=np.array([s, s, s]),
                color=np.array([0.8, 0.3, 0.1]),
            )
        )


def add_goal(world: World):
    x, y = GOAL_POSITION
    world.scene.add(
        VisualCylinder(
            prim_path="/World/Goal",
            name="goal_marker",
            position=np.array([x, y, GOAL_HEIGHT / 2.0]),
            radius=GOAL_RADIUS,
            height=GOAL_HEIGHT,
            color=np.array([0.1, 0.9, 0.1]),
        )
    )


def add_go2(world: World):
    """Reference the Go2 asset and wrap it as an articulation for proprioception."""
    assets_root = get_assets_root_path()
    if assets_root is None:
        print("[WARN] Could not resolve assets root path. Skipping Go2 load.")
        return None

    go2_usd_path = assets_root + GO2_RELATIVE_PATH
    try:
        add_reference_to_stage(usd_path=go2_usd_path, prim_path=GO2_PRIM_PATH)
        prim = world.stage.GetPrimAtPath(GO2_PRIM_PATH)
        UsdGeom.XformCommonAPI(prim).SetTranslate(Gf.Vec3d(*GO2_SPAWN_POSITION))
        print(f"[INFO] Loaded Go2 from: {go2_usd_path}")
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] Failed to load Go2 asset ({exc}). Continuing without robot.")
        return None

    try:
        from isaacsim.core.prims import SingleArticulation

        robot = SingleArticulation(prim_path=GO2_PRIM_PATH, name="go2")
        world.scene.add(robot)
        return robot
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] Could not wrap Go2 as articulation ({exc}). Proprioception will be zeros.")
        return None


# ---------------------------------------------------------------------------
# Sensor rig
# ---------------------------------------------------------------------------
def add_sensor_rig(stage):
    """Create the kinematic Xform that carries camera / lidar / imu."""
    rig = UsdGeom.Xform.Define(stage, RIG_PRIM_PATH)
    api = UsdGeom.XformCommonAPI(rig.GetPrim())
    api.SetTranslate(Gf.Vec3d(NOMINAL_WAYPOINTS[0][0], NOMINAL_WAYPOINTS[0][1], RIG_HEIGHT))
    return rig


def set_rig_pose(rig_prim, x: float, y: float, yaw: float):
    api = UsdGeom.XformCommonAPI(rig_prim)
    api.SetTranslate(Gf.Vec3d(x, y, RIG_HEIGHT))
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


def add_imu():
    """Create an IMU sensor. Returns None if unavailable on this install."""
    try:
        from isaacsim.sensors.physics import IMUSensor

        imu = IMUSensor(
            prim_path=IMU_PRIM_PATH,
            name="go2_imu",
            translation=np.array(IMU_OFFSET),
        )
        print("[INFO] IMU sensor created.")
        return imu
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] Could not create IMU ({exc}). IMU channels will be zeros.")
        return None


# ---------------------------------------------------------------------------
# Sensor readout helpers (fixed output shapes, graceful fallbacks)
# ---------------------------------------------------------------------------
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


def read_imu(imu):
    lin_acc = np.zeros(3, dtype=np.float32)
    ang_vel = np.zeros(3, dtype=np.float32)
    orientation = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    if imu is None:
        return lin_acc, ang_vel, orientation
    try:
        frame = imu.get_current_frame()
        if frame.get("lin_acc") is not None:
            lin_acc = np.asarray(frame["lin_acc"], dtype=np.float32).reshape(3)
        if frame.get("ang_vel") is not None:
            ang_vel = np.asarray(frame["ang_vel"], dtype=np.float32).reshape(3)
        if frame.get("orientation") is not None:
            orientation = np.asarray(frame["orientation"], dtype=np.float32).reshape(4)
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] IMU read failed ({exc}).")
    return lin_acc, ang_vel, orientation


def read_proprioception(robot, num_dof_fallback: int = 12):
    if robot is not None:
        try:
            q = np.asarray(robot.get_joint_positions(), dtype=np.float32).reshape(-1)
            dq = np.asarray(robot.get_joint_velocities(), dtype=np.float32).reshape(-1)
            return q, dq
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] Proprioception read failed ({exc}).")
    zeros = np.zeros(num_dof_fallback, dtype=np.float32)
    return zeros.copy(), zeros.copy()


# ---------------------------------------------------------------------------
# Trajectory helpers
# ---------------------------------------------------------------------------
def sample_episode_waypoints(rng: np.random.Generator) -> list:
    """Jitter interior waypoints for per-episode diversity (endpoints fixed)."""
    waypoints = [list(NOMINAL_WAYPOINTS[0])]
    for x, y in NOMINAL_WAYPOINTS[1:-1]:
        jx, jy = rng.uniform(-WAYPOINT_JITTER, WAYPOINT_JITTER, size=2)
        waypoints.append([x + jx, y + jy])
    waypoints.append(list(NOMINAL_WAYPOINTS[-1]))
    return waypoints


def build_trajectory(waypoints: list, num_frames: int) -> np.ndarray:
    """Resample the waypoint polyline into `num_frames` evenly spaced (x, y, yaw)."""
    pts = np.array(waypoints, dtype=np.float64)
    seg_vecs = np.diff(pts, axis=0)
    seg_lens = np.linalg.norm(seg_vecs, axis=1)
    cum_lens = np.concatenate([[0.0], np.cumsum(seg_lens)])
    total_len = cum_lens[-1]

    samples = np.linspace(0.0, total_len, num_frames)
    poses = np.zeros((num_frames, 3), dtype=np.float64)  # x, y, yaw

    for i, s in enumerate(samples):
        seg_idx = min(np.searchsorted(cum_lens, s, side="right") - 1, len(seg_lens) - 1)
        seg_t = 0.0 if seg_lens[seg_idx] == 0.0 else (s - cum_lens[seg_idx]) / seg_lens[seg_idx]
        xy = pts[seg_idx] + seg_t * seg_vecs[seg_idx]
        yaw = math.atan2(seg_vecs[seg_idx][1], seg_vecs[seg_idx][0])
        poses[i] = (xy[0], xy[1], yaw)

    return poses


def compute_actions(trajectory: np.ndarray, dt: float) -> np.ndarray:
    """Body-frame velocity command (v_x, v_y, w_z) between consecutive poses.

    This is the action a navigation VLA typically predicts (high-level base
    velocity command, with locomotion handled by a low-level controller).
    """
    num = trajectory.shape[0]
    actions = np.zeros((num, 3), dtype=np.float32)
    for i in range(num - 1):
        x, y, yaw = trajectory[i]
        nx, ny, nyaw = trajectory[i + 1]
        dx_w, dy_w = (nx - x) / dt, (ny - y) / dt
        # rotate world-frame velocity into the body frame
        c, s = math.cos(-yaw), math.sin(-yaw)
        vx = c * dx_w - s * dy_w
        vy = s * dx_w + c * dy_w
        dyaw = math.atan2(math.sin(nyaw - yaw), math.cos(nyaw - yaw)) / dt
        actions[i] = (vx, vy, dyaw)
    # last step: hold zero command (arrived at goal)
    return actions


# ---------------------------------------------------------------------------
# Data writing
# ---------------------------------------------------------------------------
NPZ_KEYS = [
    "rgb", "depth",
    "lidar_ranges", "lidar_angles",
    "imu_lin_acc", "imu_ang_vel", "imu_orientation",
    "joint_positions", "joint_velocities",
    "base_position", "base_yaw",
    "goal_position", "goal_vector_body", "distance_to_goal",
    "action", "sim_time",
]


def save_episode(buffers: dict, episode: int, output_dir: str) -> None:
    data = {key: np.stack(vals, axis=0) for key, vals in buffers.items()}
    out_path = os.path.join(output_dir, f"episode_{episode:04d}.npz")
    np.savez_compressed(out_path, **data)
    print(f"[INFO] Saved episode {episode}: {data['rgb'].shape[0]} steps -> {out_path}")


def write_dataset_metadata(output_dir: str, num_episodes: int, steps_per_episode: int) -> None:
    metadata = {
        "language_instruction": LANGUAGE_INSTRUCTION,
        "action_space": {"type": "base_velocity_command", "dims": ["v_x", "v_y", "w_z"]},
        "room_size": ROOM_SIZE,
        "wall_height": WALL_HEIGHT,
        "wall_thickness": WALL_THICKNESS,
        "obstacles": OBSTACLE_SPECS,
        "goal_position": [GOAL_POSITION[0], GOAL_POSITION[1], GOAL_HEIGHT / 2.0],
        "goal_radius": GOAL_RADIUS,
        "robot_spawn_position": list(GO2_SPAWN_POSITION),
        "sensors": {
            "camera": {"resolution": list(CAMERA_RESOLUTION), "offset": list(CAMERA_OFFSET)},
            "lidar": {
                "num_rays": LIDAR_NUM_RAYS,
                "max_range": LIDAR_MAX_RANGE,
                "min_range": LIDAR_MIN_RANGE,
                "offset": list(LIDAR_OFFSET),
            },
            "imu": {"offset": list(IMU_OFFSET)},
        },
        "nominal_waypoints": NOMINAL_WAYPOINTS,
        "waypoint_jitter": WAYPOINT_JITTER,
        "num_episodes": num_episodes,
        "steps_per_episode": steps_per_episode,
        "physics_dt": PHYSICS_DT,
        "render_dt": RENDER_DT,
        "seed": args_cli.seed,
        "npz_keys": NPZ_KEYS,
    }
    out_path = os.path.join(output_dir, "dataset_metadata.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"[INFO] Wrote dataset metadata -> {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    output_dir = args_cli.output_dir
    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.default_rng(args_cli.seed)

    world = World(physics_dt=PHYSICS_DT, rendering_dt=RENDER_DT, stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()

    stage = omni.usd.get_context().get_stage()
    add_lighting(stage)
    add_room_walls(world)
    add_obstacles(world)
    add_goal(world)
    robot = add_go2(world)

    rig = add_sensor_rig(stage)
    rig_prim = rig.GetPrim()
    camera = add_camera()
    lidar = add_lidar(world)
    imu = add_imu()

    world.reset()
    camera.initialize()
    camera.add_distance_to_image_plane_to_frame()

    goal_xyz = np.array([GOAL_POSITION[0], GOAL_POSITION[1], GOAL_HEIGHT / 2.0], dtype=np.float32)
    lidar_angles = np.linspace(-math.pi, math.pi, LIDAR_NUM_RAYS, endpoint=False).astype(np.float32)
    # duration of one recorded step (used to convert pose deltas into velocity commands)
    step_dt = RENDER_DT

    for episode in range(args_cli.num_episodes):
        waypoints = sample_episode_waypoints(rng)
        trajectory = build_trajectory(waypoints, args_cli.steps_per_episode)
        actions = compute_actions(trajectory, step_dt)

        # move rig (and robot) to the start, then warm up physics + renderer
        x0, y0, yaw0 = trajectory[0]
        set_rig_pose(rig_prim, x0, y0, yaw0)
        if robot is not None:
            try:
                robot.set_world_pose(
                    position=np.array([x0, y0, GO2_SPAWN_POSITION[2]]),
                    orientation=euler_angles_to_quat(np.array([0.0, 0.0, yaw0])),
                )
            except Exception as exc:  # noqa: BLE001
                print(f"[WARN] Could not set robot pose ({exc}).")
        for _ in range(WARMUP_STEPS):
            world.step(render=True)

        buffers = {key: [] for key in NPZ_KEYS}
        print(f"[INFO] Episode {episode}: recording {args_cli.steps_per_episode} steps ...")

        for i, (x, y, yaw) in enumerate(trajectory):
            # kinematic traversal: teleport rig + robot root along the trajectory
            set_rig_pose(rig_prim, x, y, yaw)
            if robot is not None:
                try:
                    robot.set_world_pose(
                        position=np.array([x, y, GO2_SPAWN_POSITION[2]]),
                        orientation=euler_angles_to_quat(np.array([0.0, 0.0, yaw])),
                    )
                except Exception:  # noqa: BLE001
                    pass

            world.step(render=True)

            rgb, depth = read_camera(camera)
            ranges = read_lidar(lidar)
            lin_acc, ang_vel, orientation = read_imu(imu)
            q, dq = read_proprioception(robot)

            base_pos = np.array([x, y, RIG_HEIGHT], dtype=np.float32)
            goal_vec_world = goal_xyz - base_pos
            c, s = math.cos(-yaw), math.sin(-yaw)
            goal_vec_body = np.array(
                [
                    c * goal_vec_world[0] - s * goal_vec_world[1],
                    s * goal_vec_world[0] + c * goal_vec_world[1],
                    goal_vec_world[2],
                ],
                dtype=np.float32,
            )

            buffers["rgb"].append(rgb)
            buffers["depth"].append(depth)
            buffers["lidar_ranges"].append(ranges)
            buffers["lidar_angles"].append(lidar_angles)
            buffers["imu_lin_acc"].append(lin_acc)
            buffers["imu_ang_vel"].append(ang_vel)
            buffers["imu_orientation"].append(orientation)
            buffers["joint_positions"].append(q)
            buffers["joint_velocities"].append(dq)
            buffers["base_position"].append(base_pos)
            buffers["base_yaw"].append(np.float32(yaw))
            buffers["goal_position"].append(goal_xyz)
            buffers["goal_vector_body"].append(goal_vec_body)
            buffers["distance_to_goal"].append(np.float32(np.linalg.norm(goal_vec_world)))
            buffers["action"].append(actions[i])
            buffers["sim_time"].append(np.float32(world.current_time))

        save_episode(buffers, episode, output_dir)

    write_dataset_metadata(output_dir, args_cli.num_episodes, args_cli.steps_per_episode)
    print(f"[INFO] Done. Wrote {args_cli.num_episodes} episodes into {output_dir}/")

    simulation_app.close()


if __name__ == "__main__":
    main()