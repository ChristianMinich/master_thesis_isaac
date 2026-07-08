"""Build a simple Unitree Go2 scene starting from an empty Isaac Sim stage.

Run with the Isaac Sim python interpreter, e.g.:
    ./python.sh isaac_unitree_starter/create_empty_sim_scene.py
"""

# -- Launch Isaac Sim first (must happen before any other omni/isaac imports) --
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import numpy as np
import omni.usd
from pxr import Gf, UsdGeom, UsdLux

from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid, FixedCuboid
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.storage.native import get_assets_root_path

# ---------------------------------------------------------------------------
# Inline configuration
# ---------------------------------------------------------------------------
PHYSICS_DT = 1.0 / 200.0
RENDER_DT = 1.0 / 60.0

# Go2 USD asset. Adjust if your Isaac Sim / Nucleus version stores it elsewhere.
GO2_RELATIVE_PATH = "/Isaac/Robots/Unitree/Go2/go2.usd"  # <-- placeholder, version dependent
GO2_PRIM_PATH = "/World/Go2"
GO2_SPAWN_POSITION = (0.0, 0.0, 0.45)

# Simple obstacle layout (position xy, size). Extend or randomize later.
OBSTACLE_SPECS = [
    {"name": "block_1", "pos": (2.0, 1.0), "size": 0.4},
    {"name": "block_2", "pos": (3.5, -1.2), "size": 0.5},
    {"name": "block_3", "pos": (5.0, 0.5), "size": 0.3},
    {"name": "block_4", "pos": (4.0, 2.0), "size": 0.6},
]

WALL_SPECS = [
    # name, center (x, y), scale (x, y, z)
    {"name": "wall_left", "pos": (3.0, 3.5), "scale": (8.0, 0.2, 1.0)},
    {"name": "wall_right", "pos": (3.0, -3.5), "scale": (8.0, 0.2, 1.0)},
]


def add_lighting(stage):
    """Add a dome light and a distant light for basic illumination."""
    dome = UsdLux.DomeLight.Define(stage, "/World/Lights/DomeLight")
    dome.CreateIntensityAttr(1000.0)

    distant = UsdLux.DistantLight.Define(stage, "/World/Lights/DistantLight")
    distant.CreateIntensityAttr(3000.0)
    distant.CreateAngleAttr(0.5)
    UsdGeom.XformCommonAPI(distant.GetPrim()).SetRotate(Gf.Vec3f(-45.0, 0.0, 0.0))


def add_camera(stage):
    """Add a viewer camera looking at the robot spawn area."""
    camera = UsdGeom.Camera.Define(stage, "/World/ViewCamera")
    xform_api = UsdGeom.XformCommonAPI(camera.GetPrim())
    xform_api.SetTranslate(Gf.Vec3d(-3.0, -3.0, 2.5))
    xform_api.SetRotate(Gf.Vec3f(65.0, 0.0, -45.0))
    return camera


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


def add_obstacles(world: World):
    """Create simple static blocks and walls. Structured for later randomization."""
    for spec in OBSTACLE_SPECS:
        x, y = spec["pos"]
        s = spec["size"]
        world.scene.add(
            DynamicCuboid(
                prim_path=f"/World/Obstacles/{spec['name']}",
                name=spec["name"],
                position=np.array([x, y, s / 2.0]),
                scale=np.array([s, s, s]),
                color=np.array([0.8, 0.3, 0.1]),
            )
        )

    for spec in WALL_SPECS:
        x, y = spec["pos"]
        sx, sy, sz = spec["scale"]
        world.scene.add(
            FixedCuboid(
                prim_path=f"/World/Obstacles/{spec['name']}",
                name=spec["name"],
                position=np.array([x, y, sz / 2.0]),
                scale=np.array([sx, sy, sz]),
                color=np.array([0.5, 0.5, 0.5]),
            )
        )


def main():
    world = World(physics_dt=PHYSICS_DT, rendering_dt=RENDER_DT, stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()

    stage = omni.usd.get_context().get_stage()
    add_lighting(stage)
    add_camera(stage)
    add_go2(world)
    add_obstacles(world)

    world.reset()
    print("[INFO] Scene ready. Running simulation (close the window or Ctrl+C to stop).")

    while simulation_app.is_running():
        world.step(render=True)

    simulation_app.close()


if __name__ == "__main__":
    main()