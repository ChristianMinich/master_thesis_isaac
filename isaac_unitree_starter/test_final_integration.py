"""Final integration test: the full production scene (room + Go2 + all sensors +
keyboard control + recording) run together end to end, headless.

Builds the exact scene main.py builds (physics scene, ground, walls+doorway,
tables/chairs/obstacle, Go2, sensors), drives the robot via synthetic keyboard
events through the doorway while recording, and validates the resulting
session on disk reflects a coherent run.
"""

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

import glob
import os
import shutil

import carb
import csv
import omni.timeline
from isaacsim.core.deprecation_manager import import_module
from isaacsim.core.experimental.utils.stage import define_prim
from isaacsim.core.rendering_manager import RenderingManager
from isaacsim.core.simulation_manager import SimulationManager
from isaacsim.core.simulation_manager.impl.isaac_events import IsaacEvents

import app_config as config
from controllers.keyboard_controller import KeyboardController, RECORD_TOGGLE_KEY
from data_logger import DataLogger
from robots.go2.go2_controller import Go2Controller
from robots.go2.sensor_manager import FOOT_LINK_RELATIVE_PATHS, SensorManager, transform_prim_paths
from scenes.room_scene import ROOM_HALF_Y, RoomScene, WALL_THICKNESS

torch = import_module("torch")


class FakeInput:
    def __init__(self, name: str) -> None:
        self.name = name


class FakeEvent:
    def __init__(self, event_type, name: str) -> None:
        self.type = event_type
        self.input = FakeInput(name)


def press(kb, name):
    kb._on_keyboard_event(FakeEvent(carb.input.KeyboardEventType.KEY_PRESS, name))


def release(kb, name):
    kb._on_keyboard_event(FakeEvent(carb.input.KeyboardEventType.KEY_RELEASE, name))


def main() -> None:
    test_recordings_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_test_recordings_final_tmp")
    if os.path.isdir(test_recordings_root):
        shutil.rmtree(test_recordings_root)

    print(">>> [final] configuring simulation manager", flush=True)
    SimulationManager.set_backend(config.PHYSICS_BACKEND)
    SimulationManager.set_physics_sim_device(config.PHYSICS_DEVICE)
    SimulationManager.set_physics_dt(config.PHYSICS_DT)
    RenderingManager.set_dt(config.RENDERING_DT)

    print(">>> [final] building physics scene + room", flush=True)
    define_prim("/World/PhysicsScene", "PhysicsScene")
    RoomScene().build()

    print(">>> [final] spawning Go2", flush=True)
    go2 = Go2Controller()
    go2.spawn()

    print(">>> [final] creating SensorManager", flush=True)
    sensors = SensorManager()

    logger = DataLogger(
        dof_names=go2.policy.robot.dof_names,
        foot_names=list(FOOT_LINK_RELATIVE_PATHS.keys()),
        transform_prim_paths=transform_prim_paths(go2.base_link_path),
        session_root=test_recordings_root,
    )
    kb = KeyboardController(command_device=config.PHYSICS_DEVICE, on_toggle_recording=logger.toggle)

    base_command = torch.zeros(3, device=config.PHYSICS_DEVICE)
    physics_step_count = 0

    def on_physics_step(dt, _ctx):
        nonlocal physics_step_count
        go2.on_physics_step(dt, base_command)
        physics_step_count += 1

    SimulationManager.register_callback(on_physics_step, IsaacEvents.POST_PHYSICS_STEP)

    omni.timeline.get_timeline_interface().play()
    simulation_app.update()

    def run_steps(n):
        for _ in range(n):
            simulation_app.update()
            if SimulationManager.is_simulating():
                base_command[:] = kb.update(config.RENDERING_DT)
                if logger.is_recording:
                    logger.log_step(
                        step=physics_step_count,
                        sim_time=omni.timeline.get_timeline_interface().get_current_time(),
                        physics_step=physics_step_count,
                        go2=go2,
                        sensors=sensors,
                        command=base_command,
                        action=go2.get_last_action(),
                    )

    print(">>> [final] letting robot settle", flush=True)
    run_steps(80)
    start_pos = go2.policy.robot.get_world_poses()[0].numpy()[0].copy()
    print(f">>> [final] start_pos={start_pos}", flush=True)

    print(f">>> [final] starting recording via synthetic '{RECORD_TOGGLE_KEY}' press", flush=True)
    press(kb, RECORD_TOGGLE_KEY)
    assert logger.is_recording
    release(kb, RECORD_TOGGLE_KEY)

    print(">>> [final] driving south ('C' strafe) toward and through the doorway", flush=True)
    press(kb, "C")
    run_steps(250)
    release(kb, "C")

    mid_pos = go2.policy.robot.get_world_poses()[0].numpy()[0].copy()
    print(f">>> [final] pos after driving through doorway={mid_pos}", flush=True)

    print(">>> [final] stopping via Space, a few more idle frames, then stopping recording", flush=True)
    press(kb, "SPACE")
    run_steps(30)
    press(kb, RECORD_TOGGLE_KEY)
    assert not logger.is_recording
    release(kb, RECORD_TOGGLE_KEY)

    south_inner_face_y = -ROOM_HALF_Y + WALL_THICKNESS / 2.0
    assert mid_pos[1] < south_inner_face_y - 0.15, (
        f"robot did not pass through the doorway during the integration run: y={mid_pos[1]:.3f}"
    )
    print(">>> [final] PASS: robot walked through the doorway during the recorded run", flush=True)

    session_dirs = glob.glob(os.path.join(test_recordings_root, "session_*"))
    assert len(session_dirs) == 1, f"expected exactly one session dir, found {session_dirs}"
    session_dir = session_dirs[0]

    csv_path = os.path.join(session_dir, "telemetry.csv")
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    print(f">>> [final] telemetry.csv rows={len(rows)}", flush=True)
    assert len(rows) == 250 + 30, f"unexpected row count {len(rows)}"

    # During the drive phase, cmd_vy should be the strafe-south value (matches KEY_COMMAND_MAP['C']);
    # during the final settle phase (after Space), it should be exactly zero.
    drive_rows = rows[:250]
    settle_rows = rows[250:]
    nonzero_vy = [r for r in drive_rows if abs(float(r["cmd_vy"])) > 0.01]
    assert len(nonzero_vy) == len(drive_rows), "expected every drive-phase row to have a nonzero cmd_vy"
    zero_vy_after_stop = all(abs(float(r["cmd_vx"])) < 1e-9 and abs(float(r["cmd_vy"])) < 1e-9 for r in settle_rows)
    assert zero_vy_after_stop, "expected zero command in every row after Space was pressed"
    print(">>> [final] PASS: recorded command column matches the actual drive -> stop sequence", flush=True)

    base_z_values = [float(r["base_pos_z"]) for r in rows]
    assert all(0.03 < z < 0.6 for z in base_z_values), f"base_pos_z out of sane range: {min(base_z_values)}..{max(base_z_values)}"
    print(">>> [final] PASS: base height stayed within a physically sane range throughout", flush=True)

    rgb_files = glob.glob(os.path.join(session_dir, "rgb", "*.png"))
    depth_files = glob.glob(os.path.join(session_dir, "depth", "*.npy"))
    lidar_files = glob.glob(os.path.join(session_dir, "lidar", "*.npz"))
    print(f">>> [final] rgb={len(rgb_files)} depth={len(depth_files)} lidar={len(lidar_files)}", flush=True)
    assert len(rgb_files) == len(rows)
    assert len(depth_files) == len(rows)
    assert len(lidar_files) > 0, "expected at least some LiDAR scans across 280 recorded frames"
    print(">>> [final] PASS: RGB/depth/LiDAR files all present on disk", flush=True)

    shutil.rmtree(test_recordings_root)
    print(">>> [final] ALL FINAL INTEGRATION CHECKS PASSED", flush=True)
    simulation_app.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
        simulation_app.close()
        raise
