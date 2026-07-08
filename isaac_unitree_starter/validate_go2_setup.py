"""Staged validation of Isaac Sim (and Isaac Lab availability) for the Go2 use case.

Runs a sequence of checks and prints a PASS/FAIL summary at the end, so you can
quickly verify that your installation supports the thesis workflow before
building on top of it:

    [1] simulation_app   Isaac Sim launches (SimulationApp)
    [2] core_imports     isaacsim.core / pxr / numpy import correctly
    [3] isaac_lab        isaaclab + isaaclab_assets are importable (optional)
    [4] assets_root      Nucleus assets root path resolves
    [5] scene_build      Room with walls + obstacles + goal builds
    [6] go2_load         Go2 USD loads and wraps as an articulation (DOFs found)
    [7] physics_settle   Robot settles under physics without falling through
    [8] movement         Robot moves around the room through waypoints,
                         pose updates are reflected in simulation

Exit code is 0 if all required checks pass, 1 otherwise. The Isaac Lab check
is informational (does not fail the run) since this script runs under the
plain Isaac Sim interpreter.

Run with the Isaac Sim python interpreter, e.g.:
    ./python.sh isaac_unitree_starter/validate_go2_setup.py --headless
"""

import argparse
import sys
import traceback

parser = argparse.ArgumentParser(description="Validate Isaac Sim/Lab setup for the Go2 use case.")
parser.add_argument("--headless", action="store_true", help="Run without the GUI.")
parser.add_argument("--steps-per-waypoint", type=int, default=60, help="Sim steps between waypoints.")
args_cli = parser.parse_args()

# ---------------------------------------------------------------------------
# Result tracking
# ---------------------------------------------------------------------------
RESULTS = []  # list of (name, passed, required, detail)


def record(name: str, passed: bool, required: bool = True, detail: str = ""):
    RESULTS.append((name, passed, required, detail))
    status = "PASS" if passed else ("FAIL" if required else "WARN")
    print(f"[{status}] {name}" + (f" -- {detail}" if detail else ""))


def summarize_and_exit():
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
        print("All required checks passed. Your setup supports the Go2 room use case.")
    else:
        print(f"{failed_required} required check(s) FAILED. Fix these before proceeding.")
    print("=" * 60)
    try:
        simulation_app.close()
    except Exception:  # noqa: BLE001
        pass
    sys.exit(0 if failed_required == 0 else 1)


# ---------------------------------------------------------------------------
# [1] Launch Isaac Sim
# ---------------------------------------------------------------------------
try:
    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": args_cli.headless})
    record("simulation_app", True, detail="Isaac Sim launched")
except Exception as exc:  # noqa: BLE001
    print(f"[FAIL] simulation_app -- could not launch Isaac Sim: {exc}")
    traceback.print_exc()
    sys.exit(1)

# ---------------------------------------------------------------------------
# [2] Core imports
# ---------------------------------------------------------------------------
try:
    import math

    import numpy as np
    import omni.usd
    from pxr import Gf, UsdGeom, UsdLux

    from isaacsim.core.api import World
    from isaacsim.core.api.objects import FixedCuboid, VisualCylinder
    from isaacsim.core.utils.rotations import euler_angles_to_quat
    from isaacsim.core.utils.stage import add_reference_to_stage
    from isaacsim.storage.native import get_assets_root_path

    record("core_imports", True)
except Exception as exc:  # noqa: BLE001
    record("core_imports", False, detail=str(exc))
    summarize_and_exit()

# ---------------------------------------------------------------------------
# [3] Isaac Lab availability (informational only under Isaac Sim python)
# ---------------------------------------------------------------------------
try:
    import importlib

    importlib.import_module("isaaclab")
    importlib.import_module("isaaclab_assets")
    record("isaac_lab", True, required=False, detail="isaaclab + isaaclab_assets importable")
except Exception as exc:  # noqa: BLE001
    record(
        "isaac_lab",
        False,
        required=False,
        detail=f"not importable here ({exc}); run Isaac Lab scripts via ./isaaclab.sh -p",
    )

# ---------------------------------------------------------------------------
# Inline configuration (room layout mirrors create_room_scene.py)
# ---------------------------------------------------------------------------
PHYSICS_DT = 1.0 / 200.0
RENDER_DT = 1.0 / 60.0

ROOM_SIZE = 10.0
WALL_HEIGHT = 1.0
WALL_THICKNESS = 0.2

GO2_RELATIVE_PATH = "/Isaac/Robots/Unitree/Go2/go2.usd"  # <-- placeholder, version dependent
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

# Movement validation: waypoints (x, y) around the room, skirting obstacles.
MOVE_WAYPOINTS = [
    (-3.5, -3.5),
    (-2.5, -0.5),
    (-0.8, 0.8),
    (1.0, 2.2),
    (3.5, 3.5),
]

SETTLE_STEPS = 120        # physics steps to let the robot settle
POSE_TOLERANCE = 0.5      # max allowed xy error (m) after moving to a waypoint
MIN_BASE_HEIGHT = 0.05    # robot fell through the floor / collapsed below this


# ---------------------------------------------------------------------------
# [4] Assets root
# ---------------------------------------------------------------------------
assets_root = None
try:
    assets_root = get_assets_root_path()
    if assets_root is None:
        record("assets_root", False, detail="get_assets_root_path() returned None (Nucleus not reachable?)")
    else:
        record("assets_root", True, detail=assets_root)
except Exception as exc:  # noqa: BLE001
    record("assets_root", False, detail=str(exc))

# ---------------------------------------------------------------------------
# [5] Build the room scene
# ---------------------------------------------------------------------------
world = None
try:
    world = World(physics_dt=PHYSICS_DT, rendering_dt=RENDER_DT, stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()

    stage = omni.usd.get_context().get_stage()

    # lighting
    dome = UsdLux.DomeLight.Define(stage, "/World/Lights/DomeLight")
    dome.CreateIntensityAttr(1000.0)
    distant = UsdLux.DistantLight.Define(stage, "/World/Lights/DistantLight")
    distant.CreateIntensityAttr(3000.0)
    UsdGeom.XformCommonAPI(distant.GetPrim()).SetRotate(Gf.Vec3f(-45.0, 0.0, 0.0))

    # walls
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

    # obstacles
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

    # goal marker
    world.scene.add(
        VisualCylinder(
            prim_path="/World/Goal",
            name="goal_marker",
            position=np.array([GOAL_POSITION[0], GOAL_POSITION[1], GOAL_HEIGHT / 2.0]),
            radius=GOAL_RADIUS,
            height=GOAL_HEIGHT,
            color=np.array([0.1, 0.9, 0.1]),
        )
    )
    record("scene_build", True, detail=f"{len(wall_specs)} walls, {len(OBSTACLE_SPECS)} obstacles, goal marker")
except Exception as exc:  # noqa: BLE001
    record("scene_build", False, detail=str(exc))
    summarize_and_exit()

# ---------------------------------------------------------------------------
# [6] Load Go2 and wrap as articulation
# ---------------------------------------------------------------------------
robot = None
try:
    if assets_root is None:
        raise RuntimeError("assets root unavailable; cannot load Go2 USD")

    go2_usd_path = assets_root + GO2_RELATIVE_PATH
    add_reference_to_stage(usd_path=go2_usd_path, prim_path=GO2_PRIM_PATH)
    prim = world.stage.GetPrimAtPath(GO2_PRIM_PATH)
    if not prim.IsValid():
        raise RuntimeError(f"prim at {GO2_PRIM_PATH} is invalid after referencing {go2_usd_path}")
    UsdGeom.XformCommonAPI(prim).SetTranslate(Gf.Vec3d(*GO2_SPAWN_POSITION))

    from isaacsim.core.prims import SingleArticulation

    robot = SingleArticulation(prim_path=GO2_PRIM_PATH, name="go2")
    world.scene.add(robot)

    world.reset()

    num_dof = robot.num_dof
    if num_dof is None or num_dof <= 0:
        raise RuntimeError(f"articulation has no DOFs (num_dof={num_dof})")
    record("go2_load", True, detail=f"{go2_usd_path} ({num_dof} DOFs)")
except Exception as exc:  # noqa: BLE001
    record("go2_load", False, detail=str(exc))
    # try to at least init the world so the remaining physics checks can run
    try:
        if world.physics_sim_view is None:
            world.reset()
    except Exception:  # noqa: BLE001
        pass

# ---------------------------------------------------------------------------
# [7] Physics settling: step and confirm the robot does not fall through
# ---------------------------------------------------------------------------
try:
    if robot is None:
        raise RuntimeError("robot unavailable (go2_load failed)")

    for _ in range(SETTLE_STEPS):
        world.step(render=not args_cli.headless)

    pos, _ = robot.get_world_pose()
    base_height = float(pos[2])
    if not math.isfinite(base_height):
        raise RuntimeError(f"non-finite base height: {base_height}")
    if base_height < MIN_BASE_HEIGHT:
        raise RuntimeError(f"robot base sank to z={base_height:.3f} m (fell through floor / collapsed)")
    record("physics_settle", True, detail=f"base height after settling: {base_height:.3f} m")
except Exception as exc:  # noqa: BLE001
    record("physics_settle", False, detail=str(exc))

# ---------------------------------------------------------------------------
# [8] Movement: drive the robot through waypoints around the obstacles
# ---------------------------------------------------------------------------
try:
    if robot is None:
        raise RuntimeError("robot unavailable (go2_load failed)")

    reached = 0
    for wx, wy in MOVE_WAYPOINTS:
        # heading toward the waypoint
        cur, _ = robot.get_world_pose()
        yaw = math.atan2(wy - float(cur[1]), wx - float(cur[0]))

        # basic movement validation: set the base pose to the waypoint and let
        # physics run so contacts/joints respond. (Locomotion policies come later;
        # here we only validate that the robot can be placed and simulated
        # anywhere in the room without exploding or falling through.)
        robot.set_world_pose(
            position=np.array([wx, wy, GO2_SPAWN_POSITION[2]]),
            orientation=euler_angles_to_quat(np.array([0.0, 0.0, yaw])),
        )
        for _ in range(args_cli.steps_per_waypoint):
            world.step(render=not args_cli.headless)

        pos, _ = robot.get_world_pose()
        err = math.hypot(float(pos[0]) - wx, float(pos[1]) - wy)
        height_ok = float(pos[2]) >= MIN_BASE_HEIGHT and math.isfinite(float(pos[2]))
        if err <= POSE_TOLERANCE and height_ok:
            reached += 1
            print(f"[INFO] waypoint ({wx:+.1f}, {wy:+.1f}) reached (xy err {err:.2f} m, z {float(pos[2]):.2f} m)")
        else:
            print(f"[WARN] waypoint ({wx:+.1f}, {wy:+.1f}) off (xy err {err:.2f} m, z {float(pos[2]):.2f} m)")

    if reached == len(MOVE_WAYPOINTS):
        record("movement", True, detail=f"visited all {reached}/{len(MOVE_WAYPOINTS)} waypoints in the room")
    else:
        record("movement", False, detail=f"only {reached}/{len(MOVE_WAYPOINTS)} waypoints validated")
except Exception as exc:  # noqa: BLE001
    record("movement", False, detail=str(exc))

summarize_and_exit()