"""Headless validation of the recording pipeline: 'R' toggles a session on/off,
files land on disk in the expected formats, and the CSV row count matches
frames actually recorded.
"""

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

import csv
import glob
import json
import os
import shutil

import carb
import numpy as np
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
from robots.go2.sensor_manager import CAMERA_HEIGHT, CAMERA_WIDTH, FOOT_LINK_RELATIVE_PATHS, SensorManager, transform_prim_paths
from scenes.empty_scene import EmptyScene

torch = import_module("torch")


class FakeInput:
    def __init__(self, name: str) -> None:
        self.name = name


class FakeEvent:
    def __init__(self, event_type, name: str) -> None:
        self.type = event_type
        self.input = FakeInput(name)


def main() -> None:
    test_recordings_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_test_recordings_tmp")
    if os.path.isdir(test_recordings_root):
        shutil.rmtree(test_recordings_root)

    SimulationManager.set_backend(config.PHYSICS_BACKEND)
    SimulationManager.set_physics_sim_device(config.PHYSICS_DEVICE)
    SimulationManager.set_physics_dt(config.PHYSICS_DT)
    RenderingManager.set_dt(config.RENDERING_DT)

    define_prim("/World/PhysicsScene", "PhysicsScene")
    EmptyScene().build()

    go2 = Go2Controller()
    go2.spawn()
    sensors = SensorManager()
    logger = DataLogger(
        dof_names=go2.policy.robot.dof_names,
        foot_names=list(FOOT_LINK_RELATIVE_PATHS.keys()),
        transform_prim_paths=transform_prim_paths(go2.base_link_path),
        session_root=test_recordings_root,
    )

    base_command = torch.tensor([0.3, 0.0, 0.0], device=config.PHYSICS_DEVICE)
    physics_step_count = 0

    def on_physics_step(dt, _ctx):
        nonlocal physics_step_count
        go2.on_physics_step(dt, base_command)
        physics_step_count += 1

    SimulationManager.register_callback(on_physics_step, IsaacEvents.POST_PHYSICS_STEP)

    kb = KeyboardController(command_device=config.PHYSICS_DEVICE, on_toggle_recording=logger.toggle)

    omni.timeline.get_timeline_interface().play()
    simulation_app.update()

    # Settle the robot before recording.
    for _ in range(60):
        simulation_app.update()

    assert not logger.is_recording
    kb._on_keyboard_event(FakeEvent(carb.input.KeyboardEventType.KEY_PRESS, RECORD_TOGGLE_KEY))
    assert logger.is_recording, f"'{RECORD_TOGGLE_KEY}' press should start a recording session"
    print(f">>> [ltest] PASS: '{RECORD_TOGGLE_KEY}' starts recording", flush=True)

    logged_steps = 0
    for step in range(120):
        simulation_app.update()
        if SimulationManager.is_simulating():
            logger.log_step(
                step=step,
                sim_time=omni.timeline.get_timeline_interface().get_current_time(),
                physics_step=physics_step_count,
                go2=go2,
                sensors=sensors,
                command=base_command,
                action=go2.get_last_action(),
            )
            logged_steps += 1

    kb._on_keyboard_event(FakeEvent(carb.input.KeyboardEventType.KEY_PRESS, RECORD_TOGGLE_KEY))
    assert not logger.is_recording, f"'{RECORD_TOGGLE_KEY}' press should stop recording"
    print(f">>> [ltest] PASS: '{RECORD_TOGGLE_KEY}' stops recording", flush=True)

    session_dirs = glob.glob(os.path.join(test_recordings_root, "session_*"))
    assert len(session_dirs) == 1, f"expected exactly one session dir, found {session_dirs}"
    session_dir = session_dirs[0]
    print(f">>> [ltest] session_dir={session_dir}", flush=True)

    metadata_path = os.path.join(session_dir, "session_metadata.json")
    assert os.path.isfile(metadata_path), "session_metadata.json missing"
    with open(metadata_path) as f:
        metadata = json.load(f)
    assert metadata["dof_names"] == list(go2.policy.robot.dof_names)
    print(">>> [ltest] PASS: session_metadata.json written with correct dof_names", flush=True)

    csv_path = os.path.join(session_dir, "telemetry.csv")
    assert os.path.isfile(csv_path), "telemetry.csv missing"
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    print(f">>> [ltest] telemetry.csv rows={len(rows)} (logged_steps={logged_steps})", flush=True)
    assert len(rows) == logged_steps, f"CSV row count {len(rows)} != logged steps {logged_steps}"
    print(">>> [ltest] PASS: CSV row count matches logged frames", flush=True)

    first_row = rows[0]
    assert "joint_pos_FL_hip_joint" in first_row, "expected per-joint columns in CSV"
    assert "foot_FL_in_contact" in first_row
    assert "camera_tf_pos_x" in first_row
    assert "base_pos_z" in first_row
    base_z_values = [float(r["base_pos_z"]) for r in rows]
    assert all(0.05 < z < 0.6 for z in base_z_values), f"unexpected base_pos_z range: {min(base_z_values)}..{max(base_z_values)}"
    print(">>> [ltest] PASS: CSV columns present and base height sane across all rows", flush=True)

    rgb_files = glob.glob(os.path.join(session_dir, "rgb", "*.png"))
    depth_files = glob.glob(os.path.join(session_dir, "depth", "*.npy"))
    print(f">>> [ltest] rgb_files={len(rgb_files)} depth_files={len(depth_files)}", flush=True)
    assert len(rgb_files) == logged_steps, f"expected {logged_steps} PNGs, found {len(rgb_files)}"
    assert len(depth_files) == logged_steps, f"expected {logged_steps} depth .npy files, found {len(depth_files)}"

    sample_depth = np.load(depth_files[0])
    print(f">>> [ltest] sample depth shape={sample_depth.shape} dtype={sample_depth.dtype}", flush=True)
    assert sample_depth.shape[:2] == (CAMERA_HEIGHT, CAMERA_WIDTH)
    print(">>> [ltest] PASS: RGB PNGs + depth NPY files written and shaped correctly", flush=True)

    lidar_files = glob.glob(os.path.join(session_dir, "lidar", "*.npz"))
    print(f">>> [ltest] lidar_files={len(lidar_files)} (0 is possible if no frame had a fresh scan)", flush=True)
    if lidar_files:
        sample = np.load(lidar_files[0])
        assert set(sample.keys()) == {"points", "range", "intensity"}
        print(f">>> [ltest] sample lidar npz: points={sample['points'].shape} range={sample['range'].shape}", flush=True)
        print(">>> [ltest] PASS: LiDAR NPZ files written with points/range/intensity", flush=True)

    shutil.rmtree(test_recordings_root)
    print(">>> [ltest] ALL DATA LOGGER TESTS PASSED", flush=True)
    simulation_app.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
        simulation_app.close()
        raise
