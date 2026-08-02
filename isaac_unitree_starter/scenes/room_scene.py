"""RoomScene: a rectangular room with a real doorway, tables, chairs, and an obstacle.

Self-contained -- all room-specific tunables (dimensions, furniture asset names)
live here rather than in the shared app_config, so this scene is portable on its
own. Walls are primitive Cube colliders (no remote asset dependency); furniture
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

# Rectangular room centered on the origin (robot spawns at its center).
ROOM_HALF_X = 4.0  # meters, +/- extent along X
ROOM_HALF_Y = 3.0  # meters, +/- extent along Y
WALL_HEIGHT = 2.0
WALL_THICKNESS = 0.2
DOORWAY_WIDTH = 1.2  # physical gap left open in the south wall, centered on x=0

# Real Isaac Sim office furniture assets (paths confirmed to exist via
# omni.client.list() against get_assets_root_path(), not guessed).
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
