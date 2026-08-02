"""Headless physical validation of room collision geometry.

Spawns three Go2 robots into the same room and drives each at a different
feature under a fixed velocity command:
  1. straight backward into the solid West wall -> must be blocked.
  2. south through the doorway gap in the South wall -> must pass through.
  3. south into the solid South wall segment beside the doorway -> must be blocked.

This is the only reliable way to confirm "collision geometry on all objects"
and "a physically open doorway" -- checking prim/API existence alone can't
prove the doorway gap isn't accidentally covered by an overlapping collider.
"""

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

import omni.timeline
from isaacsim.core.deprecation_manager import import_module
from isaacsim.core.experimental.utils.stage import define_prim
from isaacsim.core.rendering_manager import RenderingManager
from isaacsim.core.simulation_manager import SimulationManager
from isaacsim.core.simulation_manager.impl.isaac_events import IsaacEvents

import app_config as config
from robots.go2.go2_controller import Go2Controller
from scenes.room_scene import ROOM_HALF_X, ROOM_HALF_Y, RoomScene, WALL_THICKNESS

torch = import_module("torch")


def main() -> None:
    SimulationManager.set_backend(config.PHYSICS_BACKEND)
    SimulationManager.set_physics_sim_device(config.PHYSICS_DEVICE)
    SimulationManager.set_physics_dt(config.PHYSICS_DT)
    RenderingManager.set_dt(config.RENDERING_DT)

    define_prim("/World/PhysicsScene", "PhysicsScene")
    RoomScene().build()

    go2_wall = Go2Controller(prim_path="/World/Go2Wall", position=[-3.4, 0.0, 0.5])
    go2_wall.spawn()
    go2_door = Go2Controller(prim_path="/World/Go2Door", position=[0.0, -2.5, 0.5])
    go2_door.spawn()
    go2_wallsouth = Go2Controller(prim_path="/World/Go2WallSouth", position=[-2.0, -2.5, 0.5])
    go2_wallsouth.spawn()
    # TableA is at (2.0, 1.5, 0); approach from below it along +y.
    go2_table = Go2Controller(prim_path="/World/Go2Table", position=[2.0, 0.5, 0.5])
    go2_table.spawn()

    cmd_backward = torch.tensor([-1.0, 0.0, 0.0], device=config.PHYSICS_DEVICE)
    cmd_south = torch.tensor([0.0, -1.0, 0.0], device=config.PHYSICS_DEVICE)
    cmd_north = torch.tensor([0.0, 1.0, 0.0], device=config.PHYSICS_DEVICE)

    def on_physics_step(dt: float, _context) -> None:
        go2_wall.on_physics_step(dt, cmd_backward)
        go2_door.on_physics_step(dt, cmd_south)
        go2_wallsouth.on_physics_step(dt, cmd_south)
        go2_table.on_physics_step(dt, cmd_north)

    SimulationManager.register_callback(on_physics_step, IsaacEvents.POST_PHYSICS_STEP)

    omni.timeline.get_timeline_interface().play()
    simulation_app.update()

    for step in range(220):
        simulation_app.update()
        if step % 40 == 0:
            wx = go2_wall.policy.robot.get_world_poses()[0].numpy()[0]
            dx = go2_door.policy.robot.get_world_poses()[0].numpy()[0]
            sx = go2_wallsouth.policy.robot.get_world_poses()[0].numpy()[0]
            tx = go2_table.policy.robot.get_world_poses()[0].numpy()[0]
            print(f">>> [rtest] step={step} wall_x={wx} door_pos={dx} wallsouth_pos={sx} table_pos={tx}", flush=True)

    wall_x_final = go2_wall.policy.robot.get_world_poses()[0].numpy()[0]
    door_pos_final = go2_door.policy.robot.get_world_poses()[0].numpy()[0]
    wallsouth_pos_final = go2_wallsouth.policy.robot.get_world_poses()[0].numpy()[0]
    table_pos_final = go2_table.policy.robot.get_world_poses()[0].numpy()[0]

    print(
        f">>> [rtest] FINAL wall_x={wall_x_final} door_pos={door_pos_final} "
        f"wallsouth_pos={wallsouth_pos_final} table_pos={table_pos_final}",
        flush=True,
    )

    wall_inner_face_x = -ROOM_HALF_X + WALL_THICKNESS / 2.0
    south_inner_face_y = -ROOM_HALF_Y + WALL_THICKNESS / 2.0

    assert wall_x_final[0] > wall_inner_face_x - 0.05, (
        f"West wall did not block the robot: x={wall_x_final[0]:.3f} tunneled past inner face "
        f"{wall_inner_face_x:.3f}"
    )
    print(">>> [rtest] PASS: West wall blocks the robot", flush=True)

    assert door_pos_final[1] < south_inner_face_y - 0.15, (
        f"Doorway did not let the robot through: y={door_pos_final[1]:.3f}, expected well past "
        f"{south_inner_face_y:.3f}"
    )
    print(">>> [rtest] PASS: doorway gap is physically open", flush=True)

    assert wallsouth_pos_final[1] > south_inner_face_y - 0.05, (
        f"Solid South wall segment did not block the robot: y={wallsouth_pos_final[1]:.3f} tunneled past "
        f"inner face {south_inner_face_y:.3f}"
    )
    print(">>> [rtest] PASS: solid South wall segment (beside doorway) blocks the robot", flush=True)

    # TableA sits at y=1.5; approaching from y=0.5 the robot must not tunnel well past the
    # table's near edge (table footprint is a few tenths of a meter, so allow generous margin).
    assert table_pos_final[1] < 1.3, (
        f"TableA did not block the robot: y={table_pos_final[1]:.3f}, expected to stop before y=1.3"
    )
    print(">>> [rtest] PASS: TableA (referenced furniture asset) blocks the robot", flush=True)

    print(">>> [rtest] ALL ROOM COLLISION TESTS PASSED", flush=True)
    simulation_app.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
        simulation_app.close()
        raise
