"""Headless validation of LiveViewer's panel-building logic against real sensor data.

Doesn't assert on the actual displayed window (nothing to "see" in an automated
headless run) -- validates the RGB/depth/LiDAR panel arrays it builds have the
right shape, dtype, and non-degenerate content.
"""

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

import numpy as np
import omni.timeline
from isaacsim.core.deprecation_manager import import_module
from isaacsim.core.experimental.utils.stage import define_prim
from isaacsim.core.rendering_manager import RenderingManager
from isaacsim.core.simulation_manager import SimulationManager
from isaacsim.core.simulation_manager.impl.isaac_events import IsaacEvents

import app_config as config
import live_viewer as live_viewer_module
from live_viewer import LiveViewer
from robots.go2.go2_controller import Go2Controller
from robots.go2.sensor_manager import SensorManager
from scenes.empty_scene import EmptyScene

torch = import_module("torch")


def main() -> None:
    SimulationManager.set_backend(config.PHYSICS_BACKEND)
    SimulationManager.set_physics_sim_device(config.PHYSICS_DEVICE)
    SimulationManager.set_physics_dt(config.PHYSICS_DT)
    RenderingManager.set_dt(config.RENDERING_DT)

    define_prim("/World/PhysicsScene", "PhysicsScene")
    EmptyScene().build()

    go2 = Go2Controller()
    go2.spawn()
    sensors = SensorManager()

    base_command = torch.tensor([0.5, 0.0, 0.0], device=config.PHYSICS_DEVICE)

    def on_physics_step(dt, _ctx):
        go2.on_physics_step(dt, base_command)

    SimulationManager.register_callback(on_physics_step, IsaacEvents.POST_PHYSICS_STEP)

    omni.timeline.get_timeline_interface().play()
    simulation_app.update()
    for _ in range(100):
        simulation_app.update()

    viewer = LiveViewer()
    print(">>> [vtest] LiveViewer created (omni.ui.Window + ByteImageProvider)", flush=True)

    camera_reading = sensors.read_camera()
    assert camera_reading["rgb"] is not None
    assert camera_reading["depth"] is not None

    rgb_panel = viewer._rgb_panel(camera_reading["rgb"])
    print(f">>> [vtest] rgb_panel shape={rgb_panel.shape} dtype={rgb_panel.dtype}", flush=True)
    assert rgb_panel.shape == (live_viewer_module.PANEL_HEIGHT, live_viewer_module.PANEL_WIDTH, 3)
    assert rgb_panel.dtype == np.uint8
    assert rgb_panel.std() > 1.0, "RGB panel looks flat/degenerate (near-zero variance)"
    print(">>> [vtest] PASS: RGB panel has correct shape/dtype and real content", flush=True)

    depth_panel = viewer._depth_panel(camera_reading["depth"])
    print(f">>> [vtest] depth_panel shape={depth_panel.shape} dtype={depth_panel.dtype}", flush=True)
    assert depth_panel.shape == (live_viewer_module.PANEL_HEIGHT, live_viewer_module.PANEL_WIDTH, 3)
    assert depth_panel.dtype == np.uint8
    print(">>> [vtest] PASS: depth panel has correct shape/dtype", flush=True)

    # Poll a few frames for a non-empty LiDAR scan (async render cadence, same as
    # test_sensors.py -- a single frame can legitimately read 0 points).
    lidar_reading = None
    for _ in range(20):
        simulation_app.update()
        candidate = sensors.read_lidar()
        if candidate["num_points"] > 0:
            lidar_reading = candidate
            break
    assert lidar_reading is not None, "LiDAR returned 0 points across 20 consecutive frames"

    lidar_panel = viewer._lidar_panel(lidar_reading["points"])
    print(f">>> [vtest] lidar_panel shape={lidar_panel.shape} dtype={lidar_panel.dtype} "
          f"nonzero_pixels={int(np.count_nonzero(lidar_panel))}", flush=True)
    assert lidar_panel.shape == (live_viewer_module.LIDAR_PANEL_SIZE, live_viewer_module.LIDAR_PANEL_SIZE, 3)
    assert np.count_nonzero(lidar_panel) > 0, "LiDAR panel is completely empty -- points weren't rasterized"
    print(">>> [vtest] PASS: LiDAR top-down panel has correct shape and rasterized points", flush=True)

    # Full update() path (RGB/depth resize + hstack + imshow) must not raise.
    viewer.update(camera_reading, lidar_reading)
    print(">>> [vtest] PASS: viewer.update() ran end to end without error", flush=True)

    viewer.close()
    print(">>> [vtest] ALL LIVE VIEWER TESTS PASSED", flush=True)
    simulation_app.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
        simulation_app.close()
        raise
