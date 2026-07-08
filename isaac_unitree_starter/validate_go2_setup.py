"""
Staged validation of Isaac Sim and Isaac Lab availability for the Unitree Go2 use case.

Expected project structure:

    master_thesis_isaac/
    ├── assets/
    │   └── robots/
    │       └── go2.usd
    └── isaac_unitree_starter/
        └── validate_go2_setup.py

Runs a sequence of checks and prints a PASS/FAIL summary:

    [1] simulation_app     Isaac Sim launches
    [2] core_imports       Isaac Sim / USD / NumPy imports work
    [3] isaac_lab          Isaac Lab is importable, informational only
    [4] local_go2_asset    Local Go2 USD file exists
    [5] scene_build        Room with walls, obstacles, and goal builds
    [6] go2_load           Go2 USD loads as an articulation
    [7] physics_settle     Robot settles under physics
    [8] movement           Robot can be placed through waypoints in the room

Run with Isaac Sim's Python interpreter, for example:

    cd /path/to/master_thesis_isaac
    /path/to/isaac-sim/python.sh isaac_unitree_starter/validate_go2_setup.py --headless

Or, if your local wrapper is called isaacsim.sh:

    cd /path/to/master_thesis_isaac
    ./isaacsim.sh isaac_unitree_starter/validate_go2_setup.py --headless
"""

import argparse
import math
import sys
import traceback
from pathlib import Path


# -----------------------------------------------------------------------------
# CLI arguments
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Validate Isaac Sim/Lab setup for the local Unitree Go2 use case."
)
parser.add_argument(
    "--headless",
    action="store_true",
    help="Run without the GUI.",
)
parser.add_argument(
    "--steps-per-waypoint",
    type=int,
    default=60,
    help="Simulation steps between waypoint placements.",
)
args_cli = parser.parse_args()


# -----------------------------------------------------------------------------
# Result tracking
# -----------------------------------------------------------------------------
RESULTS = []  # list of (name, passed, required, detail)


def record(name: str, passed: bool, required: bool = True, detail: str = ""):
    """Store and print a validation result."""
    RESULTS.append((name, passed, required, detail))
    status = "PASS" if passed else ("FAIL" if required else "WARN")
    print(f"[{status}] {name}" + (f" -- {detail}" if detail else ""))


def summarize_and_exit():
    """Print summary, close Isaac Sim, and exit with appropriate code."""
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)

    failed_required = 0

    for name, passed, required, detail in RESULTS:
        status = "PASS" if passed else ("FAIL" if required else "WARN")
        print(f"  [{status:4}] {name}" + (f" -- {detail}" if detail else ""))

        if required and not passed:
            failed_required += 1

    print("=" * 60)

    if failed_required == 0:
        print("All required checks passed. Your setup supports the local Go2 room use case.")
    else:
        print(f"{failed_required} required check(s) FAILED. Fix these before proceeding.")

    print("=" * 60)

    try:
        simulation_app.close()
    except Exception:
        pass

    sys.exit(0 if failed_required == 0 else 1)


# -----------------------------------------------------------------------------
# [1] Launch Isaac Sim
# -----------------------------------------------------------------------------
try:
    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": args_cli.headless})
    record("simulation_app", True, detail="Isaac Sim launched")
except Exception as exc:
    print(f"[FAIL] simulation_app -- could not launch Isaac Sim: {exc}")
    traceback.print_exc()
    sys.exit(1)


# -----------------------------------------------------------------------------
# [2] Core imports
# -----------------------------------------------------------------------------
try:
    import numpy as np
    import omni.usd
    from pxr import Gf, UsdGeom, UsdLux

    from isaacsim.core.api import World
    from isaacsim.core.api.objects import FixedCuboid, VisualCylinder
    from isaacsim.core.utils.rotations import euler_angles_to_quat
    from isaacsim.core.utils.stage import add_reference_to_stage

    record("core_imports", True)
except Exception as exc:
    record("core_imports", False, detail=str(exc))
    summarize_and_exit()


# -----------------------------------------------------------------------------
# [3] Isaac Lab availability
# -----------------------------------------------------------------------------
try:
    import importlib

    importlib.import_module("isaaclab")
    importlib.import_module("isaaclab_assets")

    record(
        "isaac_lab",
        True,
        required=False,
        detail="isaaclab + isaaclab_assets importable",
    )
except Exception as exc:
    record(
        "isaac_lab",
        False,
        required=False,
        detail=(
            f"not importable here ({exc}); "
            "this is okay if you run Isaac Lab scripts separately via ./isaaclab.sh -p"
        ),
    )


# -----------------------------------------------------------------------------
# Project paths
# -----------------------------------------------------------------------------
# This file is expected to be here:
#   master_thesis_isaac/isaac_unitree_starter/validate_go2_setup.py
#
# Therefore:
#   Path(__file__).resolve().parents[0] = isaac_unitree_starter/
#   Path(__file__).resolve().parents[1] = master_thesis_isaac/
PROJECT_ROOT = Path(__file__).resolve().parents[1]

GO2_USD_PATH = PROJECT_ROOT / "assets" / "robots" / "go2.usd"


# -----------------------------------------------------------------------------
# Inline configuration
# -----------------------------------------------------------------------------
PHYSICS_DT = 1.0 / 200.0
RENDER_DT = 1.0 / 60.0

ROOM_SIZE = 10.0
WALL_HEIGHT = 1.0
WALL_THICKNESS = 0.2

GO2_PRIM_PATH = "/World/Go2"
GO2_SPAWN_POSITION = (-3.5, -3.5, 0.45)

GOAL_POSITION = (3.5, 3.5)
GOAL_RADIUS = 0.3
GOAL_HEIGHT = 0.05

OBSTACLE_SPECS = [
    {"name": "block_1", "pos": (0.0, 0.0), "size": 0.6},
    {"name": "block_2", "pos": (-1.5, 1.5), "size": 0.5},
    {"name": "block_3", "pos": (1.5, -1.0), "size": 0.4},
    {"name": "block_4", "pos": (2.0, 1.5), "size": 0.5},
    {"name": "block_5", "pos": (-1.0, -2.0), "size": 0.4},
    {"name": "block_6", "pos": (0.5, 2.5), "size": 0.3},
]

MOVE_WAYPOINTS = [
    (-3.5, -3.5),
    (-2.5, -0.5),
    (-0.8, 0.8),
    (1.0, 2.2),
    (3.5, 3.5),
]

SETTLE_STEPS = 120
POSE_TOLERANCE = 0.5
MIN_BASE_HEIGHT = 0.05


# -----------------------------------------------------------------------------
# [4] Local Go2 asset path
# -----------------------------------------------------------------------------
try:
    if GO2_USD_PATH.exists():
        record("local_go2_asset", True, detail=str(GO2_USD_PATH))
    else:
        record(
            "local_go2_asset",
            False,
            detail=f"Go2 USD not found at: {GO2_USD_PATH}",
        )
except Exception as exc:
    record("local_go2_asset", False, detail=str(exc))


# -----------------------------------------------------------------------------
# [5] Build the room scene
# -----------------------------------------------------------------------------
world = None

try:
    world = World(
        physics_dt=PHYSICS_DT,
        rendering_dt=RENDER_DT,
        stage_units_in_meters=1.0,
    )

    world.scene.add_default_ground_plane()

    stage = omni.usd.get_context().get_stage()

    # Lighting
    dome = UsdLux.DomeLight.Define(stage, "/World/Lights/DomeLight")
    dome.CreateIntensityAttr(1000.0)

    distant = UsdLux.DistantLight.Define(stage, "/World/Lights/DistantLight")
    distant.CreateIntensityAttr(3000.0)
    UsdGeom.XformCommonAPI(distant.GetPrim()).SetRotate(Gf.Vec3f(-45.0, 0.0, 0.0))

    # Walls
    half = ROOM_SIZE / 2.0

    wall_specs = [
        {
            "name": "wall_north",
            "pos": (0.0, half),
            "scale": (ROOM_SIZE + WALL_THICKNESS, WALL_THICKNESS, WALL_HEIGHT),
        },
        {
            "name": "wall_south",
            "pos": (0.0, -half),
            "scale": (ROOM_SIZE + WALL_THICKNESS, WALL_THICKNESS, WALL_HEIGHT),
        },
        {
            "name": "wall_east",
            "pos": (half, 0.0),
            "scale": (WALL_THICKNESS, ROOM_SIZE + WALL_THICKNESS, WALL_HEIGHT),
        },
        {
            "name": "wall_west",
            "pos": (-half, 0.0),
            "scale": (WALL_THICKNESS, ROOM_SIZE + WALL_THICKNESS, WALL_HEIGHT),
        },
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

    # Obstacles
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

    # Goal marker
    world.scene.add(
        VisualCylinder(
            prim_path="/World/Goal",
            name="goal_marker",
            position=np.array(
                [
                    GOAL_POSITION[0],
                    GOAL_POSITION[1],
                    GOAL_HEIGHT / 2.0,
                ]
            ),
            radius=GOAL_RADIUS,
            height=GOAL_HEIGHT,
            color=np.array([0.1, 0.9, 0.1]),
        )
    )

    record(
        "scene_build",
        True,
        detail=f"{len(wall_specs)} walls, {len(OBSTACLE_SPECS)} obstacles, goal marker",
    )

except Exception as exc:
    record("scene_build", False, detail=str(exc))
    summarize_and_exit()


# -----------------------------------------------------------------------------
# [6] Load local Go2 and wrap as articulation
# -----------------------------------------------------------------------------
robot = None

try:
    if not GO2_USD_PATH.exists():
        raise FileNotFoundError(f"Go2 USD not found: {GO2_USD_PATH}")

    go2_usd_path = str(GO2_USD_PATH)

    add_reference_to_stage(
        usd_path=go2_usd_path,
        prim_path=GO2_PRIM_PATH,
    )

    prim = world.stage.GetPrimAtPath(GO2_PRIM_PATH)

    if not prim.IsValid():
        raise RuntimeError(
            f"Prim at {GO2_PRIM_PATH} is invalid after referencing {go2_usd_path}"
        )

    UsdGeom.XformCommonAPI(prim).SetTranslate(Gf.Vec3d(*GO2_SPAWN_POSITION))

    from isaacsim.core.prims import SingleArticulation

    robot = SingleArticulation(
        prim_path=GO2_PRIM_PATH,
        name="go2",
    )

    world.scene.add(robot)

    world.reset()

    num_dof = robot.num_dof

    if num_dof is None or num_dof <= 0:
        raise RuntimeError(f"Articulation has no DOFs. num_dof={num_dof}")

    record(
        "go2_load",
        True,
        detail=f"{go2_usd_path} loaded with {num_dof} DOFs",
    )

except Exception as exc:
    record("go2_load", False, detail=str(exc))

    try:
        if world.physics_sim_view is None:
            world.reset()
    except Exception:
        pass


# -----------------------------------------------------------------------------
# [7] Physics settling
# -----------------------------------------------------------------------------
try:
    if robot is None:
        raise RuntimeError("Robot unavailable because go2_load failed.")

    for _ in range(SETTLE_STEPS):
        world.step(render=not args_cli.headless)

    pos, _ = robot.get_world_pose()
    base_height = float(pos[2])

    if not math.isfinite(base_height):
        raise RuntimeError(f"Non-finite base height: {base_height}")

    if base_height < MIN_BASE_HEIGHT:
        raise RuntimeError(
            f"Robot base sank to z={base_height:.3f} m. "
            "It may have fallen through the floor or collapsed."
        )

    record(
        "physics_settle",
        True,
        detail=f"base height after settling: {base_height:.3f} m",
    )

except Exception as exc:
    record("physics_settle", False, detail=str(exc))


# -----------------------------------------------------------------------------
# [8] Movement validation
# -----------------------------------------------------------------------------
try:
    if robot is None:
        raise RuntimeError("Robot unavailable because go2_load failed.")

    reached = 0

    for wx, wy in MOVE_WAYPOINTS:
        cur, _ = robot.get_world_pose()

        yaw = math.atan2(
            wy - float(cur[1]),
            wx - float(cur[0]),
        )

        # This is not real locomotion yet.
        # It places the robot at waypoints and steps physics.
        # Real Go2 walking will require a controller or trained policy later.
        robot.set_world_pose(
            position=np.array(
                [
                    wx,
                    wy,
                    GO2_SPAWN_POSITION[2],
                ]
            ),
            orientation=euler_angles_to_quat(
                np.array(
                    [
                        0.0,
                        0.0,
                        yaw,
                    ]
                )
            ),
        )

        for _ in range(args_cli.steps_per_waypoint):
            world.step(render=not args_cli.headless)

        pos, _ = robot.get_world_pose()

        err = math.hypot(
            float(pos[0]) - wx,
            float(pos[1]) - wy,
        )

        height_ok = float(pos[2]) >= MIN_BASE_HEIGHT and math.isfinite(float(pos[2]))

        if err <= POSE_TOLERANCE and height_ok:
            reached += 1
            print(
                f"[INFO] waypoint ({wx:+.1f}, {wy:+.1f}) reached "
                f"(xy err {err:.2f} m, z {float(pos[2]):.2f} m)"
            )
        else:
            print(
                f"[WARN] waypoint ({wx:+.1f}, {wy:+.1f}) off "
                f"(xy err {err:.2f} m, z {float(pos[2]):.2f} m)"
            )

    if reached == len(MOVE_WAYPOINTS):
        record(
            "movement",
            True,
            detail=f"visited all {reached}/{len(MOVE_WAYPOINTS)} waypoints in the room",
        )
    else:
        record(
            "movement",
            False,
            detail=f"only {reached}/{len(MOVE_WAYPOINTS)} waypoints validated",
        )

except Exception as exc:
    record("movement", False, detail=str(exc))


summarize_and_exit()