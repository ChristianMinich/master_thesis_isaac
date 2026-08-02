"""Headless integration test: synthetic arrow-key events -> KeyboardController -> Go2 policy.

Confirms the full command path (not just the key-accumulation math already
covered by test_keyboard_controller.py): a synthetic 'UP' press drives the
robot forward, and a synthetic 'LEFT' press turns it, exactly as it would from
a real keypress once a viewport window is present.
"""

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

import carb
import omni.timeline
from isaacsim.core.deprecation_manager import import_module
from isaacsim.core.experimental.utils.stage import define_prim
from isaacsim.core.rendering_manager import RenderingManager
from isaacsim.core.simulation_manager import SimulationManager
from isaacsim.core.simulation_manager.impl.isaac_events import IsaacEvents

import app_config as config
from controllers.keyboard_controller import KeyboardController
from robots.go2.go2_controller import Go2Controller
from scenes.empty_scene import EmptyScene
import isaacsim.core.experimental.utils.transform as transform_utils

torch = import_module("torch")


class FakeInput:
    def __init__(self, name: str) -> None:
        self.name = name


class FakeEvent:
    def __init__(self, event_type, name: str) -> None:
        self.type = event_type
        self.input = FakeInput(name)


def main() -> None:
    SimulationManager.set_backend(config.PHYSICS_BACKEND)
    SimulationManager.set_physics_sim_device(config.PHYSICS_DEVICE)
    SimulationManager.set_physics_dt(config.PHYSICS_DT)
    RenderingManager.set_dt(config.RENDERING_DT)

    define_prim("/World/PhysicsScene", "PhysicsScene")
    EmptyScene().build()

    go2 = Go2Controller()
    go2.spawn()

    base_command = torch.zeros(3, device=config.PHYSICS_DEVICE)

    def on_physics_step(dt: float, _context) -> None:
        go2.on_physics_step(dt, base_command)

    SimulationManager.register_callback(on_physics_step, IsaacEvents.POST_PHYSICS_STEP)

    kb = KeyboardController(command_device=config.PHYSICS_DEVICE)

    omni.timeline.get_timeline_interface().play()
    simulation_app.update()

    def run_steps(n: int) -> None:
        for _ in range(n):
            simulation_app.update()
            if SimulationManager.is_simulating():
                base_command[:] = kb.update(config.RENDERING_DT)

    # Let the robot settle to a standing pose first.
    run_steps(100)
    start_pos, start_quat = go2.policy.robot.get_world_poses()
    start_pos = start_pos.numpy()[0].copy()
    start_yaw = _yaw_from_quat(start_quat)
    print(f">>> [itest] settled start_pos={start_pos} start_yaw={start_yaw:.3f}", flush=True)

    # Synthetic 'UP' press -> should walk forward (+x in world frame at yaw=0).
    kb._on_keyboard_event(FakeEvent(carb.input.KeyboardEventType.KEY_PRESS, "UP"))
    run_steps(120)
    fwd_pos, _ = go2.policy.robot.get_world_poses()
    fwd_pos = fwd_pos.numpy()[0]
    forward_delta = float(((fwd_pos[0] - start_pos[0]) ** 2 + (fwd_pos[1] - start_pos[1]) ** 2) ** 0.5)
    print(f">>> [itest] after UP held: pos={fwd_pos} forward_delta={forward_delta:.3f}", flush=True)
    assert forward_delta > 0.3, f"expected robot to move forward under UP command, delta={forward_delta:.3f}"
    print(">>> [itest] PASS: synthetic UP command drives robot forward", flush=True)

    kb._on_keyboard_event(FakeEvent(carb.input.KeyboardEventType.KEY_RELEASE, "UP"))
    kb._on_keyboard_event(FakeEvent(carb.input.KeyboardEventType.KEY_PRESS, "LEFT"))
    run_steps(80)
    _, turn_quat = go2.policy.robot.get_world_poses()
    turn_yaw = _yaw_from_quat(turn_quat)
    yaw_delta = abs(turn_yaw - start_yaw)
    print(f">>> [itest] after LEFT held: yaw={turn_yaw:.3f} yaw_delta={yaw_delta:.3f}", flush=True)
    assert yaw_delta > 0.2, f"expected robot to turn under LEFT command, yaw_delta={yaw_delta:.3f}"
    print(">>> [itest] PASS: synthetic LEFT command turns robot", flush=True)

    kb._on_keyboard_event(FakeEvent(carb.input.KeyboardEventType.KEY_PRESS, "SPACE"))
    print(">>> [itest] ALL INTEGRATION TESTS PASSED", flush=True)
    simulation_app.close()


def _yaw_from_quat(quat_wp) -> float:
    import numpy as np

    rot = transform_utils.quaternion_to_rotation_matrix(quat_wp).numpy()[0]
    return float(np.arctan2(rot[1, 0], rot[0, 0]))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
        simulation_app.close()
        raise
