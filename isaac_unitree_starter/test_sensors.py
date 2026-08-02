"""Headless validation of the sensor suite: RGB, depth, LiDAR, IMU, foot contacts.

Builds the minimal floor + Go2 scene, attaches SensorManager, walks the robot
forward for a short scripted run, and prints/asserts basic shape and value
sanity for every sensor stream.
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
from robots.go2.go2_controller import Go2Controller
from robots.go2.sensor_manager import CAMERA_HEIGHT, CAMERA_WIDTH, SensorManager
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

    base_command = torch.tensor([0.5, 0.0, 0.0], device=config.PHYSICS_DEVICE)

    def on_physics_step(dt: float, _context) -> None:
        go2.on_physics_step(dt, base_command)

    SimulationManager.register_callback(on_physics_step, IsaacEvents.POST_PHYSICS_STEP)

    print(">>> [stest] creating SensorManager", flush=True)
    sensors = SensorManager()
    print(">>> [stest] SensorManager created", flush=True)

    omni.timeline.get_timeline_interface().play()
    simulation_app.update()

    for step in range(150):
        simulation_app.update()
        if step % 50 == 0:
            print(f">>> [stest] step={step} ready={go2.ready}", flush=True)

    print(">>> [stest] reading camera", flush=True)
    cam = sensors.read_camera()
    rgb, depth = cam["rgb"], cam["depth"]
    print(f">>> [stest] rgb: {'None' if rgb is None else (rgb.shape, rgb.dtype)}", flush=True)
    print(f">>> [stest] depth: {'None' if depth is None else (depth.shape, depth.dtype)}", flush=True)
    assert rgb is not None, "RGB camera returned no data"
    assert rgb.shape[0] == CAMERA_HEIGHT and rgb.shape[1] == CAMERA_WIDTH, f"unexpected rgb shape {rgb.shape}"
    assert depth is not None, "Depth camera returned no data"
    finite_depth = depth[np.isfinite(depth)]
    print(f">>> [stest] depth finite range: [{finite_depth.min() if finite_depth.size else float('nan'):.3f}, "
          f"{finite_depth.max() if finite_depth.size else float('nan'):.3f}]", flush=True)
    print(">>> [stest] PASS: camera RGB + depth", flush=True)

    print(">>> [stest] reading lidar (polling a few frames -- the RTX lidar's render product only "
          "refreshes on a subset of frames, so a single frame may legitimately read 0 points)", flush=True)
    lidar = None
    for _ in range(20):
        simulation_app.update()
        candidate = sensors.read_lidar()
        if candidate["num_points"] > 0:
            lidar = candidate
            break
    assert lidar is not None, "LiDAR returned 0 points across 20 consecutive frames"
    print(f">>> [stest] lidar num_points={lidar['num_points']} "
          f"points_shape={lidar['points'].shape} range_shape={lidar['range'].shape} "
          f"intensity_shape={lidar['intensity'].shape}", flush=True)
    print(f">>> [stest] lidar range stats: min={lidar['range'].min():.3f} max={lidar['range'].max():.3f}", flush=True)
    print(">>> [stest] PASS: lidar point cloud", flush=True)

    print(">>> [stest] reading imu", flush=True)
    imu = sensors.read_imu()
    print(f">>> [stest] imu linear_acceleration={imu['linear_acceleration']} "
          f"angular_velocity={imu['angular_velocity']} orientation={imu['orientation']}", flush=True)
    assert imu["linear_acceleration"].shape == (3,)
    assert imu["angular_velocity"].shape == (3,)
    assert imu["orientation"].shape == (4,)
    print(">>> [stest] PASS: IMU", flush=True)

    print(">>> [stest] reading foot contacts", flush=True)
    contacts = sensors.read_foot_contacts()
    for name, reading in contacts.items():
        print(f">>> [stest] foot {name}: in_contact={reading['in_contact']} force={reading['force']:.3f}", flush=True)
    assert set(contacts.keys()) == {"FL", "FR", "RL", "RR"}
    print(">>> [stest] PASS: foot contact sensors", flush=True)

    print(">>> [stest] ALL SENSOR TESTS PASSED", flush=True)
    simulation_app.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
        simulation_app.close()
        raise
