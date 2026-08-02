"""Entry point: composition root wiring a Scene + Go2Robot + a command driver.

Spawns the official Unitree Go2 flat-terrain locomotion policy into a swappable
Scene (--scene room|empty). By default it's driven live via arrow-key keyboard
input (requires a viewport window with keyboard focus); --random-walk drives it
autonomously instead; --test runs a short headless scripted-command pass, for
automated validation.
"""

from isaacsim import SimulationApp

import argparse

parser = argparse.ArgumentParser(description="Go2 scene with arrow-key keyboard control")
parser.add_argument("--test", action="store_true", help="Run a short headless scripted validation pass and exit")
parser.add_argument("--headless", action="store_true", help="Run without a viewport window (disables keyboard control)")
parser.add_argument(
    "--scene",
    default="room",
    help="Which registered scene to build the robot into (see scenes/registry.py), e.g. 'room' or 'empty'",
)
parser.add_argument(
    "--scene-file",
    default=None,
    help="Load a previously saved scene (see save_scene.py) instead of building --scene from scratch",
)
parser.add_argument(
    "--random-walk",
    action="store_true",
    help="Drive the Go2 autonomously with a random-walk command generator instead of keyboard input",
)
parser.add_argument(
    "--show-sensors",
    action="store_true",
    help="Open a live 2D window showing the captured RGB, depth, and top-down LiDAR data",
)
args, _unknown = parser.parse_known_args()

# Loading a saved scene must happen as part of SimulationApp's own startup (via the
# "open_usd" launch config), not via a mid-session open_stage() call after extensions
# are already live -- that path was tried and crashed Kit's OmniGraph/RTX pipeline
# with a native "no null terminator at count" abort (see isaac_unitree_starter dev notes).
launch_config = {"headless": args.headless or args.test}
if args.scene_file:
    launch_config["open_usd"] = args.scene_file
simulation_app = SimulationApp(launch_config)

import carb
import omni.timeline
import omni.usd
from isaacsim.core.deprecation_manager import import_module
from isaacsim.core.experimental.utils.stage import define_prim
from isaacsim.core.rendering_manager import RenderingManager
from isaacsim.core.simulation_manager import SimulationManager
from isaacsim.core.simulation_manager.impl.isaac_events import IsaacEvents

import app_config as config
from controllers.keyboard_controller import KeyboardController
from controllers.random_walk_controller import RandomWalkController
from live_viewer import LiveViewer
from robots.go2.go2_robot import Go2Robot
from scenes.registry import get_scene

torch = import_module("torch")


def main() -> None:
    print(">>> [go2] configuring SimulationManager", flush=True)
    SimulationManager.set_backend(config.PHYSICS_BACKEND)
    SimulationManager.set_physics_sim_device(config.PHYSICS_DEVICE)
    SimulationManager.set_physics_dt(config.PHYSICS_DT)
    RenderingManager.set_dt(config.RENDERING_DT)

    bounds_half_extent = None
    if args.scene_file:
        print(f">>> [go2] saved scene loaded at startup: {args.scene_file}", flush=True)
        spawn_position = None  # saved scenes don't carry robot spawn info; Go2Robot uses its own default
    else:
        print(">>> [go2] creating physics scene", flush=True)
        define_prim("/World/PhysicsScene", "PhysicsScene")

        print(f">>> [go2] building scene: {args.scene!r}", flush=True)
        scene = get_scene(args.scene)
        scene_info = scene.build()
        spawn_position = scene_info.default_spawn_position
        bounds_half_extent = scene_info.bounds_half_extent

    print(">>> [go2] spawning Go2Robot (policy + sensors + logger)", flush=True)
    go2 = Go2Robot(position=spawn_position)
    print(">>> [go2] Go2Robot spawned", flush=True)

    viewer = None
    if args.show_sensors:
        if args.headless:
            carb.log_warn("--show-sensors has no effect with --headless (no display to draw to); ignoring.")
        else:
            viewer = LiveViewer()
            print(">>> [go2] live sensor viewer window opened", flush=True)

    omni.usd.get_context().get_selection().clear_selected_prim_paths()

    base_command = torch.zeros(3, device=config.PHYSICS_DEVICE)
    physics_step_count = 0

    def on_physics_step(dt: float, _context) -> None:
        nonlocal physics_step_count
        go2.on_physics_step(dt, base_command)
        physics_step_count += 1

    SimulationManager.register_callback(on_physics_step, IsaacEvents.POST_PHYSICS_STEP)

    keyboard = None
    if not args.test:
        keyboard = KeyboardController(command_device=config.PHYSICS_DEVICE, on_toggle_recording=go2.logger.toggle)
        keyboard.start()
        if args.random_walk:
            print(">>> [go2] random-walk mode active (Esc exit, L record still work if a window is open)",
                  flush=True)
        else:
            print(">>> [go2] keyboard control active: Up/Down move, Left/Right turn, Z/C strafe, "
                  "Space stop, L record, Esc exit", flush=True)

    driver = None
    if args.random_walk:
        driver = RandomWalkController(
            command_device=config.PHYSICS_DEVICE,
            go2_controller=go2.controller,
            bounds_half_extent=bounds_half_extent,
        )
    elif not args.test:
        driver = keyboard

    omni.timeline.get_timeline_interface().play()
    simulation_app.update()
    print(">>> [go2] entering main loop", flush=True)

    step = 0
    max_steps = 200 if args.test else None
    while simulation_app.is_running():
        simulation_app.update()

        if args.test:
            if SimulationManager.is_simulating():
                # Scripted command: stand for 1s, then walk forward for validation.
                if step < 100:
                    base_command[:] = torch.tensor([0.0, 0.0, 0.0], device=config.PHYSICS_DEVICE)
                else:
                    base_command[:] = torch.tensor([1.0, 0.0, 0.0], device=config.PHYSICS_DEVICE)
                if step % 20 == 0:
                    pos, _quat = go2.controller.policy.robot.get_world_poses()
                    print(f">>> [go2] step={step} ready={go2.ready} base_pos={pos.numpy()[0]}", flush=True)
        else:
            if keyboard.exit_requested:
                break
            if SimulationManager.is_simulating():
                base_command[:] = driver.update(config.RENDERING_DT)

        if SimulationManager.is_simulating() and go2.logger.is_recording:
            go2.log_step(
                step=step,
                sim_time=omni.timeline.get_timeline_interface().get_current_time(),
                physics_step=physics_step_count,
                command=base_command,
                action=go2.get_last_action(),
            )

        if viewer is not None and SimulationManager.is_simulating():
            viewer.update(go2.sensors.read_camera(), go2.sensors.read_lidar())

        step += 1
        if max_steps is not None and step >= max_steps:
            break

    if go2.logger.is_recording:
        go2.logger.stop()
    if keyboard is not None:
        keyboard.stop()
    if viewer is not None:
        viewer.close()

    print(">>> [go2] scene shutting down.", flush=True)
    simulation_app.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
        simulation_app.close()
        raise
