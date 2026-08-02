"""Headless validation that the saved scene (see save_scene.py) reloads with its
physics intact: the doorway is still passable and a solid wall still blocks --
not just "doesn't crash on load."
"""

import argparse
import os

from isaacsim import SimulationApp

parser = argparse.ArgumentParser()
parser.add_argument(
    "--scene-file",
    default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_scenes", "go2_room_scene.usd"),
)
args, _unknown = parser.parse_known_args()

simulation_app = SimulationApp({"headless": True, "open_usd": args.scene_file})

import omni.timeline
from isaacsim.core.deprecation_manager import import_module
from isaacsim.core.rendering_manager import RenderingManager
from isaacsim.core.simulation_manager import SimulationManager
from isaacsim.core.simulation_manager.impl.isaac_events import IsaacEvents

import app_config as config
from robots.go2.go2_controller import Go2Controller
from scenes.room_scene import ROOM_HALF_X, ROOM_HALF_Y, WALL_THICKNESS

torch = import_module("torch")


def main() -> None:
    SimulationManager.set_backend(config.PHYSICS_BACKEND)
    SimulationManager.set_physics_sim_device(config.PHYSICS_DEVICE)
    SimulationManager.set_physics_dt(config.PHYSICS_DT)
    RenderingManager.set_dt(config.RENDERING_DT)

    go2_door = Go2Controller(prim_path="/World/Go2Door", position=[0.0, -2.5, 0.5])
    go2_door.spawn()
    go2_wall = Go2Controller(prim_path="/World/Go2Wall", position=[-3.4, 0.0, 0.5])
    go2_wall.spawn()

    cmd_south = torch.tensor([0.0, -1.0, 0.0], device=config.PHYSICS_DEVICE)
    cmd_backward = torch.tensor([-1.0, 0.0, 0.0], device=config.PHYSICS_DEVICE)

    def on_physics_step(dt, _ctx):
        go2_door.on_physics_step(dt, cmd_south)
        go2_wall.on_physics_step(dt, cmd_backward)

    SimulationManager.register_callback(on_physics_step, IsaacEvents.POST_PHYSICS_STEP)

    omni.timeline.get_timeline_interface().play()
    simulation_app.update()

    for step in range(220):
        simulation_app.update()
        if step % 40 == 0:
            door_pos = go2_door.policy.robot.get_world_poses()[0].numpy()[0]
            wall_pos = go2_wall.policy.robot.get_world_poses()[0].numpy()[0]
            print(f">>> [ldtest] step={step} door_pos={door_pos} wall_pos={wall_pos}", flush=True)

    door_pos_final = go2_door.policy.robot.get_world_poses()[0].numpy()[0]
    wall_pos_final = go2_wall.policy.robot.get_world_poses()[0].numpy()[0]
    print(f">>> [ldtest] FINAL door_pos={door_pos_final} wall_pos={wall_pos_final}", flush=True)

    south_inner_face_y = -ROOM_HALF_Y + WALL_THICKNESS / 2.0
    wall_inner_face_x = -ROOM_HALF_X + WALL_THICKNESS / 2.0

    assert door_pos_final[1] < south_inner_face_y - 0.15, (
        f"doorway did not let the robot through after reload: y={door_pos_final[1]:.3f}"
    )
    print(">>> [ldtest] PASS: doorway is still physically open after save/load", flush=True)

    assert wall_pos_final[0] > wall_inner_face_x - 0.05, (
        f"West wall did not block the robot after reload: x={wall_pos_final[0]:.3f}"
    )
    print(">>> [ldtest] PASS: West wall still blocks the robot after save/load", flush=True)

    print(">>> [ldtest] ALL LOAD-SCENE PHYSICS CHECKS PASSED", flush=True)
    simulation_app.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
        simulation_app.close()
        raise
