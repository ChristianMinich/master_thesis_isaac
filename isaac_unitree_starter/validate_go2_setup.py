"""
Simple Isaac Sim scene with Unitree Go2 + room obstacles.

Expected project structure:

    master_thesis_isaac/
    ├── assets/
    │   └── robots/
    │       └── go2.usd
    └── isaac_unitree_starter/
        └── simple_go2_scene.py

Run:

    cd /path/to/master_thesis_isaac
    /path/to/isaac-sim/python.sh isaac_unitree_starter/simple_go2_scene.py

Or headless:

    /path/to/isaac-sim/python.sh isaac_unitree_starter/simple_go2_scene.py --headless
"""

import argparse
from pathlib import Path

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true", help="Run without GUI.")
parser.add_argument("--steps", type=int, default=1000, help="Number of simulation steps.")
args = parser.parse_args()


# -----------------------------------------------------------------------------
# Start Isaac Sim
# -----------------------------------------------------------------------------
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": args.headless})


# -----------------------------------------------------------------------------
# Imports after SimulationApp
# -----------------------------------------------------------------------------
import numpy as np
import omni.usd
from pxr import Gf, UsdGeom, UsdLux

from isaacsim.core.api import World
from isaacsim.core.api.objects import FixedCuboid, VisualCylinder
from isaacsim.core.utils.stage import add_reference_to_stage


# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

GO2_USD_PATH = PROJECT_ROOT / "assets" / "robots" / "go2.usd"

if not GO2_USD_PATH.exists():
    raise FileNotFoundError(f"Go2 USD not found here: {GO2_USD_PATH}")


# -----------------------------------------------------------------------------
# Scene configuration
# -----------------------------------------------------------------------------
PHYSICS_DT = 1.0 / 200.0
RENDER_DT = 1.0 / 60.0

ROOM_SIZE = 10.0
WALL_HEIGHT = 1.0
WALL_THICKNESS = 0.2

GO2_PRIM_PATH = "/World/Go2"
GO2_POSITION = np.array([-3.5, -3.5, 0.45])

GOAL_POSITION = np.array([3.5, 3.5, 0.025])

OBSTACLE_SPECS = [
    {"name": "block_1", "pos": (0.0, 0.0), "size": 0.6},
    {"name": "block_2", "pos": (-1.5, 1.5), "size": 0.5},
    {"name": "block_3", "pos": (1.5, -1.0), "size": 0.4},
    {"name": "block_4", "pos": (2.0, 1.5), "size": 0.5},
    {"name": "block_5", "pos": (-1.0, -2.0), "size": 0.4},
    {"name": "block_6", "pos": (0.5, 2.5), "size": 0.3},
]


# -----------------------------------------------------------------------------
# Create world
# -----------------------------------------------------------------------------
world = World(
    physics_dt=PHYSICS_DT,
    rendering_dt=RENDER_DT,
    stage_units_in_meters=1.0,
)

world.scene.add_default_ground_plane()

stage = omni.usd.get_context().get_stage()


# -----------------------------------------------------------------------------
# Lighting
# -----------------------------------------------------------------------------
dome = UsdLux.DomeLight.Define(stage, "/World/Lights/DomeLight")
dome.CreateIntensityAttr(1000.0)

distant = UsdLux.DistantLight.Define(stage, "/World/Lights/DistantLight")
distant.CreateIntensityAttr(3000.0)
UsdGeom.XformCommonAPI(distant.GetPrim()).SetRotate(Gf.Vec3f(-45.0, 0.0, 0.0))


# -----------------------------------------------------------------------------
# Room walls
# -----------------------------------------------------------------------------
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

for wall in wall_specs:
    x, y = wall["pos"]
    sx, sy, sz = wall["scale"]

    world.scene.add(
        FixedCuboid(
            prim_path=f"/World/Room/{wall['name']}",
            name=wall["name"],
            position=np.array([x, y, sz / 2.0]),
            scale=np.array([sx, sy, sz]),
            color=np.array([0.5, 0.5, 0.5]),
        )
    )


# -----------------------------------------------------------------------------
# Obstacles
# -----------------------------------------------------------------------------
for obstacle in OBSTACLE_SPECS:
    x, y = obstacle["pos"]
    s = obstacle["size"]

    world.scene.add(
        FixedCuboid(
            prim_path=f"/World/Obstacles/{obstacle['name']}",
            name=obstacle["name"],
            position=np.array([x, y, s / 2.0]),
            scale=np.array([s, s, s]),
            color=np.array([0.8, 0.3, 0.1]),
        )
    )


# -----------------------------------------------------------------------------
# Goal marker
# -----------------------------------------------------------------------------
world.scene.add(
    VisualCylinder(
        prim_path="/World/Goal",
        name="goal_marker",
        position=GOAL_POSITION,
        radius=0.3,
        height=0.05,
        color=np.array([0.1, 0.9, 0.1]),
    )
)


# -----------------------------------------------------------------------------
# Import Go2 USD
# -----------------------------------------------------------------------------
print(f"Loading Go2 from: {GO2_USD_PATH}")

add_reference_to_stage(
    usd_path=str(GO2_USD_PATH),
    prim_path=GO2_PRIM_PATH,
)

go2_prim = stage.GetPrimAtPath(GO2_PRIM_PATH)

if not go2_prim.IsValid():
    raise RuntimeError(f"Go2 was not created at {GO2_PRIM_PATH}")

UsdGeom.XformCommonAPI(go2_prim).SetTranslate(
    Gf.Vec3d(
        float(GO2_POSITION[0]),
        float(GO2_POSITION[1]),
        float(GO2_POSITION[2]),
    )
)

print(f"Go2 imported into scene at: {GO2_PRIM_PATH}")


# -----------------------------------------------------------------------------
# Reset and run
# -----------------------------------------------------------------------------
world.reset()

print("Scene is running.")
print("This script only imports Go2 visually/physically.")
print("It does NOT control the robot and does NOT require articulation wrapping.")

for _ in range(args.steps):
    world.step(render=not args.headless)

simulation_app.close()