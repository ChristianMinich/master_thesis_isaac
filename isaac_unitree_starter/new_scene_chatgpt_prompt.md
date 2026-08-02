# Prompt: generate a new Scene for the Go2 Isaac Sim project

Copy everything below the line into ChatGPT.

---

You are writing a new **Scene** for an existing Isaac Sim 6.0 Python project. The
project uses a small plugin architecture: a robot (a Unitree Go2 quadruped) is
spawned into a swappable "Scene" that only owns ground + static geometry. Scenes
are picked by name at runtime (e.g. `main.py --scene warehouse`).

## Your task

Design and write **one new Scene** with a different theme and different objects
than the existing "room" scene (an office room with tables/chairs — described
below). Pick a coherent theme (e.g. a warehouse, a garage, an outdoor loading
dock, a lab) and populate it with real furniture/props from the asset list
given at the end of this prompt — **do not invent asset file names or paths**;
only use paths from that list, or plain USD primitives (`Cube`, `Cylinder`,
`Sphere`, `Cone`) for anything the list doesn't cover.

Deliver:
1. A new file `scenes/<theme>_scene.py` implementing the `Scene` interface below,
   following the exact structure/conventions of the `RoomScene` reference example.
2. The two-line diff to `scenes/registry.py` that registers it.

## The `Scene` interface (`core/scene.py`) — do not modify this file

```python
"""Scene abstraction: the swappable environment a robot is dropped into.

A Scene owns only static/environment authoring (ground + geometry) -- it knows
nothing about robots. main.py creates the one shared PhysicsScene prim itself
(identical for every scene) before calling Scene.build().
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class SceneInfo:
    """Result of building a scene: what callers need to spawn a robot into it."""

    assets_root_path: str
    default_spawn_position: list[float]
    # (half_x, half_y) if this scene has a bounded floor plan a driver should stay
    # within (e.g. a random-walk controller's wall-avoidance), or None if unbounded.
    bounds_half_extent: tuple[float, float] | None = None


class Scene(ABC):
    """A swappable environment: ground + static geometry, authored onto the current stage."""

    name: str

    @abstractmethod
    def build(self) -> SceneInfo:
        """Author this scene's ground + static geometry onto the current USD stage.

        Returns:
            SceneInfo with the resolved assets root path and a suggested spawn position.
        """
        raise NotImplementedError
```

## Reference example to follow: `scenes/room_scene.py` (existing scene, for pattern only)

```python
"""RoomScene: a rectangular room with a real doorway, tables, chairs, and an obstacle.

Self-contained -- all room-specific tunables (dimensions, furniture asset names)
live here rather than in a shared config, so this scene is portable on its own.
Walls are primitive Cube colliders (no remote asset dependency); furniture
references real Isaac Sim office-prop assets confirmed to exist via
omni.client.list() against get_assets_root_path() (not guessed paths).
"""

from __future__ import annotations

import carb
import omni.usd
from isaacsim.core.experimental.utils.stage import define_prim
import isaacsim.core.experimental.utils.stage as stage_utils
from isaacsim.storage.native import get_assets_root_path
from pxr import Gf, Usd, UsdGeom, UsdPhysics

from core.scene import Scene, SceneInfo

GROUND_PRIM_PATH = "/World/ground"
GROUND_ENV_USD_RELATIVE_PATH = "/Isaac/Environments/Grid/default_environment.usd"
DEFAULT_SPAWN_POSITION = [0.0, 0.0, 0.5]

ROOM_HALF_X = 4.0  # meters, +/- extent along X
ROOM_HALF_Y = 3.0  # meters, +/- extent along Y
WALL_HEIGHT = 2.0
WALL_THICKNESS = 0.2
DOORWAY_WIDTH = 1.2  # physical gap left open in the south wall, centered on x=0

OFFICE_PROPS_RELATIVE_DIR = "/Isaac/Environments/Office/Props"
TABLE_ASSET_A = "SM_TableA.usd"
TABLE_ASSET_B = "SM_TableB.usd"
CHAIR_ASSET = "SM_ChairOffice.usd"


def _add_box_collider(prim_path: str, center, dimensions, color=(0.6, 0.6, 0.6)) -> None:
    """Create a static, collidable Cube prim.

    Args:
        prim_path: Stage path for the new prim.
        center: (x, y, z) world-space center of the box.
        dimensions: (x, y, z) full extents of the box in meters.
        color: Display color RGB (0-1 each), for basic visual distinction in the viewport.
    """
    prim = define_prim(prim_path, "Cube")
    cube = UsdGeom.Cube(prim)
    cube.CreateSizeAttr(1.0)
    cube.CreateDisplayColorAttr([Gf.Vec3f(*color)])

    xform = UsdGeom.Xformable(cube)
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(*center))
    xform.AddScaleOp().Set(Gf.Vec3f(dimensions[0], dimensions[1], dimensions[2]))

    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())


def _apply_collision_to_meshes(prim_path: str) -> None:
    """Walk a referenced asset's subtree and add convex-hull collision to every mesh.

    Referenced furniture assets ship as visual-only geometry; PhysX collision must be
    applied explicitly to each Mesh prim in the hierarchy.
    """
    stage = omni.usd.get_context().get_stage()
    root_prim = stage.GetPrimAtPath(prim_path)
    if not root_prim.IsValid():
        carb.log_warn(f"_apply_collision_to_meshes: prim not found at {prim_path}")
        return
    for prim in Usd.PrimRange(root_prim):
        if prim.IsA(UsdGeom.Mesh):
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI.Apply(prim)
            mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(prim)
            mesh_collision.CreateApproximationAttr().Set("convexHull")


def _reference_furniture(prim_path: str, usd_relative_name: str, position, assets_root_path: str) -> None:
    """Reference a real Isaac Sim office-furniture asset onto the stage with collision."""
    usd_path = assets_root_path + OFFICE_PROPS_RELATIVE_DIR + "/" + usd_relative_name
    prim = define_prim(prim_path, "Xform")
    prim.GetReferences().AddReference(usd_path)

    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(*position))

    _apply_collision_to_meshes(prim_path)


class RoomScene(Scene):
    """Rectangular room: four walls with a doorway, two tables, four chairs, one obstacle."""

    name = "room"

    def build(self) -> SceneInfo:
        assets_root_path = get_assets_root_path()
        if assets_root_path is None:
            carb.log_error("Could not find Isaac Sim assets folder")
            raise RuntimeError("Isaac Sim assets root path not found")

        stage_utils.add_reference_to_stage(
            usd_path=assets_root_path + GROUND_ENV_USD_RELATIVE_PATH,
            path=GROUND_PRIM_PATH,
        )

        self._build_walls()
        self._build_furniture(assets_root_path)

        return SceneInfo(
            assets_root_path=assets_root_path,
            default_spawn_position=list(DEFAULT_SPAWN_POSITION),
            bounds_half_extent=(ROOM_HALF_X, ROOM_HALF_Y),
        )

    def _build_walls(self) -> None:
        """Build four primitive walls with a physically open doorway in the south wall."""
        hx, hy, t, h = ROOM_HALF_X, ROOM_HALF_Y, WALL_THICKNESS, WALL_HEIGHT
        door_half = DOORWAY_WIDTH / 2.0
        z_center = h / 2.0

        _add_box_collider("/World/Room/WallNorth", center=(0.0, hy, z_center), dimensions=(2 * hx + 2 * t, t, h))

        south_segment_width = hx - door_half
        _add_box_collider(
            "/World/Room/WallSouthWest",
            center=(-(door_half + south_segment_width / 2.0), -hy, z_center),
            dimensions=(south_segment_width, t, h),
        )
        _add_box_collider(
            "/World/Room/WallSouthEast",
            center=(door_half + south_segment_width / 2.0, -hy, z_center),
            dimensions=(south_segment_width, t, h),
        )
        _add_box_collider("/World/Room/WallEast", center=(hx, 0.0, z_center), dimensions=(t, 2 * hy, h))
        _add_box_collider("/World/Room/WallWest", center=(-hx, 0.0, z_center), dimensions=(t, 2 * hy, h))

    def _build_furniture(self, assets_root_path: str) -> None:
        """Place two tables, four chairs, and one obstacle box, all with collision."""
        _reference_furniture("/World/Room/TableA", TABLE_ASSET_A, (2.0, 1.5, 0.0), assets_root_path)
        _reference_furniture("/World/Room/TableB", TABLE_ASSET_B, (-2.0, 1.5, 0.0), assets_root_path)
        _reference_furniture("/World/Room/ChairA1", CHAIR_ASSET, (2.0, 0.7, 0.0), assets_root_path)
        _reference_furniture("/World/Room/ChairA2", CHAIR_ASSET, (2.0, 2.3, 0.0), assets_root_path)
        _reference_furniture("/World/Room/ChairB1", CHAIR_ASSET, (-2.0, 0.7, 0.0), assets_root_path)
        _reference_furniture("/World/Room/ChairB2", CHAIR_ASSET, (-2.0, 2.3, 0.0), assets_root_path)

        # Additional simple obstacle (primitive fallback), placed clear of the doorway and tables.
        _add_box_collider(
            "/World/Room/ObstacleBox", center=(1.0, -1.2, 0.25), dimensions=(0.5, 0.5, 0.5), color=(0.8, 0.3, 0.2)
        )
```

## `scenes/registry.py` (existing file — show the two-line diff to add your scene here)

```python
"""Scene registry: name -> Scene class, so a scene can be picked by a plain string
(e.g. a --scene CLI arg) instead of importing a specific class.
"""

from __future__ import annotations

from core.scene import Scene
from scenes.empty_scene import EmptyScene
from scenes.room_scene import RoomScene

SCENE_REGISTRY: dict[str, type[Scene]] = {
    RoomScene.name: RoomScene,
    EmptyScene.name: EmptyScene,
}


def get_scene(name: str) -> Scene:
    """Instantiate the registered Scene for the given name.

    Args:
        name: Registry key (see SCENE_REGISTRY, e.g. "room" or "empty").

    Raises:
        ValueError: If no scene is registered under that name.
    """
    scene_cls = SCENE_REGISTRY.get(name)
    if scene_cls is None:
        available = ", ".join(sorted(SCENE_REGISTRY))
        raise ValueError(f"Unknown scene {name!r}. Available scenes: {available}")
    return scene_cls()
```

## Hard requirements

1. **Do not invent asset paths.** Only reference `.usd` files from the confirmed
   asset list below (each path is relative to `assets_root_path`, exactly the
   way `RoomScene` builds `assets_root_path + OFFICE_PROPS_RELATIVE_DIR + "/" + name`).
   If you want an object not on the list, build it from a USD primitive
   (`Cube`/`Cylinder`/`Sphere`/`Cone`) the same way `_add_box_collider` does —
   never guess a furniture file name that isn't listed.
2. **Every object must have collision.** Primitives get `UsdPhysics.CollisionAPI`
   directly (see `_add_box_collider`); referenced mesh assets need
   `_apply_collision_to_meshes` walking the subtree exactly like the reference
   example (referenced assets ship as visual-only geometry with no physics).
3. **Self-contained module.** All of your scene's tunables (dimensions, asset
   names, object positions) are module-level constants in your new file — do not
   add anything to a shared config file.
4. **Class attribute `name`** must be a short unique string (this is the
   `--scene <name>` value end users will type).
5. **Return a `SceneInfo`** with a real `assets_root_path` (from
   `get_assets_root_path()`), a `default_spawn_position` inside your scene
   that's clear of any obstacle, and `bounds_half_extent=(half_x, half_y)` if
   your layout has a bounded floor plan (omit / leave `None` if it's open/unbounded).
6. **Robot scale for spacing decisions:** the robot is roughly 0.57m long x
   0.32m wide x 0.54m tall standing. Leave at least ~1m of clearance in any
   walkway/doorway and keep obstacles far enough apart for it to path between
   them.
7. Physics/USD units are meters; `+Z` is up. Keep all new prims under a single
   top-level scope path (e.g. `/World/<YourScene>/...`), the same way
   `RoomScene` nests everything under `/World/Room/...`.
8. Don't add a doorway/walls at all if your theme doesn't call for an enclosed
   room — e.g. an outdoor loading dock could just be a bounded open area with
   scattered props and no walls. Match the theme.

## Confirmed real Isaac Sim asset paths (relative to `assets_root_path`)

Use only these (or plain primitives). Each is a real file confirmed to exist
via `omni.client.list()` against this project's Isaac Sim installation.

```
/Isaac/Environments/Grid/default_environment.usd        (ground plane, same as RoomScene uses)
/Isaac/Environments/Grid/gridroom_black.usd
/Isaac/Environments/Grid/gridroom_curved.usd

/Isaac/Environments/Office/Props/SM_TableA.usd
/Isaac/Environments/Office/Props/SM_TableB.usd
/Isaac/Environments/Office/Props/SM_TableC.usd
/Isaac/Environments/Office/Props/SM_TableD.usd
/Isaac/Environments/Office/Props/SM_TableD2.usd
/Isaac/Environments/Office/Props/SM_TableCoffee.usd
/Isaac/Environments/Office/Props/SM_TableRoundOverlay.usd
/Isaac/Environments/Office/Props/SM_TableRound_corner.usd
/Isaac/Environments/Office/Props/SM_TableRound_module.usd
/Isaac/Environments/Office/Props/SM_TableWorkSecurity.usd
/Isaac/Environments/Office/Props/SM_TableWorkingDouble.usd
/Isaac/Environments/Office/Props/SM_ChairOffice.usd
/Isaac/Environments/Office/Props/SM_ChairOffice_A.usd
/Isaac/Environments/Office/Props/SM_ChairOffice_A_184.usd
/Isaac/Environments/Office/Props/SM_Chair.usd
/Isaac/Environments/Office/Props/SM_Armchair.usd
/Isaac/Environments/Office/Props/SM_InformationDesk_A.usd
/Isaac/Environments/Office/Props/SM_InformationDesk_B.usd
/Isaac/Environments/Office/Props/SM_ReceptionTableCornerOutside.usd
/Isaac/Environments/Office/Props/SM_ReceptionTableStraight.usd
/Isaac/Environments/Office/Props/SM_SecretaryDeskA.usd
/Isaac/Environments/Office/Props/SM_SecretaryDeskB.usd
/Isaac/Environments/Office/Props/SM_BoxPortableA.usd  (also B..H, same pattern)
/Isaac/Environments/Office/Props/SM_ElectricRoom.usd
/Isaac/Environments/Office/Props/SM_Board.usd
/Isaac/Environments/Office/Props/SM_Briefcase.usd
/Isaac/Environments/Office/Props/SM_Extinguisher.usd
/Isaac/Environments/Office/Props/SM_PaperTablet.usd
/Isaac/Environments/Office/Props/SM_TabletPC.usd

/Isaac/Environments/Simple_Room/Props/table_low.usd
/Isaac/Environments/Simple_Room/Props/Towel_Room01_lamp.usd
/Isaac/Environments/Simple_Room/Props/Towel_Room01_window.usd
(the rest of the Towel_Room01_* set are wall/floor/window architectural pieces for a
 small bathroom-style room, not standalone furniture)

/Isaac/Environments/Simple_Warehouse/Props/S_TrafficCone.usd
/Isaac/Environments/Simple_Warehouse/Props/S_WetFloorSign.usd
/Isaac/Environments/Simple_Warehouse/warehouse.usd            (full pre-built warehouse environment)
/Isaac/Environments/Simple_Warehouse/warehouse_multiple_shelves.usd
/Isaac/Environments/Simple_Warehouse/warehouse_with_forklifts.usd
/Isaac/Environments/Digital_Twin_Warehouse/small_warehouse_digital_twin.usd

/Isaac/Props/Pallet/pallet.usd
/Isaac/Props/KLT_Bin/small_KLT.usd
/Isaac/Props/KLT_Bin/small_KLT_visual.usd
/Isaac/Props/Forklift/forklift.usd
/Isaac/Props/Dolly/dolly.usd
/Isaac/Props/Sektion_Cabinet/sektion_cabinet_instanceable.usd
/Isaac/Props/Blocks/basic_block.usd
/Isaac/Props/Blocks/block.usd
/Isaac/Props/Blocks/red_block.usd
/Isaac/Props/Blocks/green_block.usd
/Isaac/Props/Blocks/yellow_block.usd
/Isaac/Props/Blocks/nvidia_cube.usd
/Isaac/Props/Conveyors/ConveyorBelt_A01.usd  (through A18, same pattern)
/Isaac/Props/Camera/camera.usd
/Isaac/Props/Beaker/beaker_500ml.usd
```

If you want to double check any additional asset exists before using it, note
in your answer that it needs verification — don't silently assume it exists.

## Output format

Give me:
1. The full contents of the new `scenes/<theme>_scene.py` file.
2. The exact updated `SCENE_REGISTRY` dict and import line for `scenes/registry.py`.
3. A one-sentence description of the theme and what's different from `RoomScene`.
