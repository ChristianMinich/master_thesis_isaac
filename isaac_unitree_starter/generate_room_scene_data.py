"""Run a simple simulation in the room scene and save synthetic data.

Rebuilds the room-with-obstacles scene (same layout as create_room_scene.py:
four walls, static box obstacles, a green goal marker, and a Unitree Go2),
then simulates an ego camera traversing from the robot spawn toward the goal
at robot height, recording per-frame:

    - rgb                  (H, W, 3) uint8
    - depth                (H, W)    float32  (distance to image plane, meters)
    - camera_position      (3,)      float32  world-frame xyz
    - camera_yaw           ()        float32  heading around +Z (radians)
    - goal_position        (3,)      float32  world-frame xyz
    - goal_vector          (3,)      float32  goal - camera (world frame)
    - distance_to_goal     ()        float32
    - sim_time             ()        float32

Data is written as compressed .npz chunk files plus a scene_metadata.json
describing the room, obstacle layout, spawn, and goal, so the dataset is
self-contained for downstream training (world models, VLA, SLAM eval, ...).

Run with the Isaac Sim python interpreter, e.g.:
    ./python.sh isaac_unitree_starter/generate_room_scene_data.py --headless
"""

import argparse

# -- Launch Isaac Sim first (must happen before any other omni/isaac imports) --
parser = argparse.ArgumentParser(description="Generate synthetic data from the Go2 room scene.")
parser.add_argument("--headless", action="store_true", help="Run without the GUI.")
parser.add_argument("--num-frames", type=int, default=300, help="Number of frames to record.")
parser.add_argument("--output-dir", type=str, default="data/room_scene_go2", help="Output directory.")
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
ROOM_SIZE = 10.0          # side length of the square room (meters)
WALL_HEIGHT = 1.0
WALL_THICKNESS = 0.2

# Go2 USD asset. Adjust if your Isaac Sim / Nucleus version stores it elsewhere.
GO2_RELATIVE_PATH = "/Isaac/Robots/Unitree/Go2/go2.usd"  # <-- placeholder, version dependent
GO2_PRIM_PATH = "/World/Go2"
GO2_SPAWN_POSITION = (-3.5, -3.5, 0.45)  # near one corner of the room

# Goal marker (visual only, no collision)
GOAL_POSITION = (3.5, 3.5)   # opposite corner from the robot spawn
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

# Ego camera (simulates the robot's forward-facing view during traversal)
CAMERA_PRIM_PATH = "/World/EgoCamera"
CAMERA_HEIGHT = 0.5        # meters above ground (roughly Go2 head height)
CAMERA_RESOLUTION = (320, 240)  # (width, height)

# Ego trajectory: waypoints (x, y) from spawn to goal, skirting the obstacles.
TRAJECTORY_WAYPOINTS = [
    (-3.5, -3.5),
    (-2.5, -0.5),
    (-0.8, 0.8),
    (1.0, 2.2),
    (3.5, 3.5),
]

# Data saving
CHUNK_SIZE = 100           # frames per saved .npz file
WARMUP_STEPS = 30          # sim/render steps before recording (settle physics & renderer)


# ---------------------------------------------------------------------------
# Scene construction
# ---------------------------------------------------------------------------
def add_lighting(stage):
    """Add a dome light and a distant light for basic illumination."""
    dome = UsdLux.DomeLight.Define(stage, "/World/Lights/DomeLight")
    dome.CreateIntensityAttr(1000.0)

    distant = UsdLux.DistantLight.Define(stage, "/World/Lights/DistantLight")
    distant.CreateIntensityAttr(3000.0)
    distant.CreateAngleAttr(0.5)
    UsdGeom.XformCommonAPI(distant.GetPrim()).SetRotate(Gf.Vec3f(-45.0, 0.0, 0.0))


def add_room_walls(world: World):
    """Enclose the room with four fixed walls."""
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
    """Create simple static blocks inside the room."""
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
    """Add a flat green cylinder as a visual-only goal marker (no collision)."""
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
    """Reference the Unitree Go2 asset into the stage if it is available."""
    assets_root = get_assets_root_path()
    if assets_root is None:
        print("[WARN] Could not resolve assets root path. Skipping Go2 load.")
        return

    go2_usd_path = assets_root + GO2_RELATIVE_PATH
    try:
        add_reference_to_stage(usd_path=go2_usd_path, prim_path=GO2_PRIM_PATH)
        prim = world.stage.GetPrimAtPath(GO2_PRIM_PATH)
        xform_api = UsdGeom.XformCommonAPI(prim)
        xform_api.SetTranslate(Gf.Vec3d(*GO2_SPAWN_POSITION))
        print(f"[INFO] Loaded Go2 from: {go2_usd_path}")
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] Failed to load Go2 asset ({exc}). Continuing without robot.")


def add_ego_camera() -> Camera:
    """Create the ego camera used to render synthetic observations."""
    camera = Camera(
        prim_path=CAMERA_PRIM_PATH,
        position=np.array([*TRAJECTORY_WAYPOINTS[0], CAMERA_HEIGHT]),
        resolution=CAMERA_RESOLUTION,
    )
    return camera


# ---------------------------------------------------------------------------
# Trajectory helpers
# ---------------------------------------------------------------------------
def build_trajectory(num_frames: int) -> np.ndarray:
    """Resample the waypoint polyline into `num_frames` evenly spaced (x, y, yaw)."""
    waypoints = np.array(TRAJECTORY_WAYPOINTS, dtype=np.float64)
    seg_vecs = np.diff(waypoints, axis=0)
    seg_lens = np.linalg.norm(seg_vecs, axis=1)
    cum_lens = np.concatenate([[0.0], np.cumsum(seg_lens)])
    total_len = cum_lens[-1]

    samples = np.linspace(0.0, total_len, num_frames)
    poses = np.zeros((num_frames, 3), dtype=np.float64)  # x, y, yaw

    for i, s in enumerate(samples):
        seg_idx = min(np.searchsorted(cum_lens, s, side="right") - 1, len(seg_lens) - 1)
        seg_t = 0.0 if seg_lens[seg_idx] == 0.0 else (s - cum_lens[seg_idx]) / seg_lens[seg_idx]
        xy = waypoints[seg_idx] + seg_t * seg_vecs[seg_idx]
        yaw = math.atan2(seg_vecs[seg_idx][1], seg_vecs[seg_idx][0])
        poses[i] = (xy[0], xy[1], yaw)

    return poses


# ---------------------------------------------------------------------------
# Data writing
# ---------------------------------------------------------------------------
def flush_chunk(buffers: dict, chunk_index: int, output_dir: str) -> None:
    data = {key: np.stack(vals, axis=0) for key, vals in buffers.items()}
    out_path = os.path.join(output_dir, f"frames_{chunk_index:04d}.npz")
    np.savez_compressed(out_path, **data)
    print(f"[INFO] Saved chunk {chunk_index}: {data['rgb'].shape[0]} frames -> {out_path}")


def write_scene_metadata(output_dir: str, num_frames: int) -> None:
    metadata = {
        "room_size": ROOM_SIZE,
        "wall_height": WALL_HEIGHT,
        "wall_thickness": WALL_THICKNESS,
        "obstacles": OBSTACLE_SPECS,
        "goal_position": [GOAL_POSITION[0], GOAL_POSITION[1], GOAL_HEIGHT / 2.0],
        "goal_radius": GOAL_RADIUS,
        "robot_spawn_position": list(GO2_SPAWN_POSITION),
        "camera_height": CAMERA_HEIGHT,
        "camera_resolution": list(CAMERA_RESOLUTION),
        "trajectory_waypoints": TRAJECTORY_WAYPOINTS,
        "num_frames": num_frames,
        "physics_dt": PHYSICS_DT,
        "render_dt": RENDER_DT,
        "npz_keys": [
            "rgb",
            "depth",
            "camera_position",
            "camera_yaw",
            "goal_position",
            "goal_vector",
            "distance_to_goal",
            "sim_time",
        ],
    }
    out_path = os.path.join(output_dir, "scene_metadata.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"[INFO] Wrote scene metadata -> {out_path}")


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
    add_obstacles(world)
    add_goal(world)
    add_go2(world)
    camera = add_ego_camera()

    world.reset()
    camera.initialize()
    camera.add_distance_to_image_plane_to_frame()  # enables depth in get_current_frame()

    # let physics settle and the renderer warm up (first frames can be black)
    for _ in range(WARMUP_STEPS):
        world.step(render=True)

    trajectory = build_trajectory(args_cli.num_frames)
    goal_xyz = np.array([GOAL_POSITION[0], GOAL_POSITION[1], GOAL_HEIGHT / 2.0], dtype=np.float32)

    buffers = {
        "rgb": [],
        "depth": [],
        "camera_position": [],
        "camera_yaw": [],
        "goal_position": [],
        "goal_vector": [],
        "distance_to_goal": [],
        "sim_time": [],
    }
    chunk_index = 0
    recorded = 0

    print(f"[INFO] Recording {args_cli.num_frames} frames into {output_dir}/ ...")

    for x, y, yaw in trajectory:
        position = np.array([x, y, CAMERA_HEIGHT])
        orientation = euler_angles_to_quat(np.array([0.0, 0.0, yaw]))
        camera.set_world_pose(position=position, orientation=orientation)

        world.step(render=True)

        frame = camera.get_current_frame()
        rgba = camera.get_rgba()
        if rgba is None or rgba.size == 0:
            print("[WARN] Empty camera frame, skipping.")
            continue
        rgb = np.asarray(rgba)[..., :3].astype(np.uint8)
        depth = frame.get("distance_to_image_plane")
        if depth is None:
            depth = np.zeros(rgb.shape[:2], dtype=np.float32)
        depth = np.asarray(depth, dtype=np.float32)

        cam_pos = position.astype(np.float32)
        goal_vec = goal_xyz - cam_pos

        buffers["rgb"].append(rgb)
        buffers["depth"].append(depth)
        buffers["camera_position"].append(cam_pos)
        buffers["camera_yaw"].append(np.float32(yaw))
        buffers["goal_position"].append(goal_xyz)
        buffers["goal_vector"].append(goal_vec.astype(np.float32))
        buffers["distance_to_goal"].append(np.float32(np.linalg.norm(goal_vec)))
        buffers["sim_time"].append(np.float32(world.current_time))
        recorded += 1

        if len(buffers["rgb"]) >= CHUNK_SIZE:
            flush_chunk(buffers, chunk_index, output_dir)
            buffers = {key: [] for key in buffers}
            chunk_index += 1

    # flush remainder
    if buffers["rgb"]:
        flush_chunk(buffers, chunk_index, output_dir)

    write_scene_metadata(output_dir, recorded)
    print(f"[INFO] Done. Recorded {recorded} frames into {output_dir}/")

    simulation_app.close()


if __name__ == "__main__":
    main()