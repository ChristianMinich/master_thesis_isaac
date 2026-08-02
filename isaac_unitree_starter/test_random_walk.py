"""Headless validation of RandomWalkController: the command actually varies over
time, the robot actually moves, and wall-avoidance keeps it from wandering
outside the room's footprint for an extended run.
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
from controllers.random_walk_controller import RandomWalkController
from robots.go2.go2_controller import Go2Controller
from scenes.room_scene import ROOM_HALF_X, ROOM_HALF_Y, RoomScene

torch = import_module("torch")


def main() -> None:
    SimulationManager.set_backend(config.PHYSICS_BACKEND)
    SimulationManager.set_physics_sim_device(config.PHYSICS_DEVICE)
    SimulationManager.set_physics_dt(config.PHYSICS_DT)
    RenderingManager.set_dt(config.RENDERING_DT)

    define_prim("/World/PhysicsScene", "PhysicsScene")
    RoomScene().build()

    go2 = Go2Controller()
    go2.spawn()

    base_command = torch.zeros(3, device=config.PHYSICS_DEVICE)

    def on_physics_step(dt, _ctx):
        go2.on_physics_step(dt, base_command)

    SimulationManager.register_callback(on_physics_step, IsaacEvents.POST_PHYSICS_STEP)

    random_walk = RandomWalkController(
        command_device=config.PHYSICS_DEVICE,
        go2_controller=go2,
        bounds_half_extent=(ROOM_HALF_X, ROOM_HALF_Y),
        seed=42,
    )

    omni.timeline.get_timeline_interface().play()
    simulation_app.update()

    positions = []
    commands_seen = set()
    max_abs_x = 0.0
    max_abs_y = 0.0

    # ~4000 frames * RENDERING_DT(0.02s) = 80s of sim time -- long enough to cross
    # several random-command hold periods (3-7s each) and approach multiple walls,
    # enough to catch a robot that gets stuck circling at a wall instead of escaping.
    for step in range(4000):
        simulation_app.update()
        if SimulationManager.is_simulating():
            base_command[:] = random_walk.update(config.RENDERING_DT)
            commands_seen.add(tuple(round(v, 3) for v in base_command.detach().cpu().numpy().tolist()))

        if step % 100 == 0:
            pos = go2.policy.robot.get_world_poses()[0].numpy()[0]
            positions.append((step, pos.copy()))
            max_abs_x = max(max_abs_x, abs(float(pos[0])))
            max_abs_y = max(max_abs_y, abs(float(pos[1])))
            print(f">>> [rwtest] step={step} pos={pos} cmd={base_command.detach().cpu().numpy()}", flush=True)

    print(f">>> [rwtest] distinct commands issued: {len(commands_seen)}", flush=True)
    assert len(commands_seen) >= 3, f"expected the command to vary over 30s of sim time, saw {len(commands_seen)}"
    print(">>> [rwtest] PASS: command varies over time (not a single frozen value)", flush=True)

    total_travel = sum(
        float(((positions[i + 1][1][0] - positions[i][1][0]) ** 2 + (positions[i + 1][1][1] - positions[i][1][1]) ** 2) ** 0.5)
        for i in range(len(positions) - 1)
    )
    print(f">>> [rwtest] total planar travel distance over run: {total_travel:.3f} m", flush=True)
    assert total_travel > 1.0, f"expected the robot to actually wander, total travel was only {total_travel:.3f}m"
    print(">>> [rwtest] PASS: robot actually moved around, not stuck in place", flush=True)

    # Regression check for the earlier bug: a fixed-direction wall-avoidance turn could
    # trap the robot circling in a ~0.3m radius right at the wall, going nowhere for many
    # seconds. Detect that pattern directly: no 12-second window (6 samples @ 2s) should
    # have every consecutive step-to-step displacement below 5cm.
    max_stuck_run = 0
    current_run = 0
    for i in range(len(positions) - 1):
        d = float(
            ((positions[i + 1][1][0] - positions[i][1][0]) ** 2 + (positions[i + 1][1][1] - positions[i][1][1]) ** 2)
            ** 0.5
        )
        if d < 0.05:
            current_run += 1
            max_stuck_run = max(max_stuck_run, current_run)
        else:
            current_run = 0
    print(f">>> [rwtest] longest run of near-zero (<5cm/2s) displacement: {max_stuck_run} samples", flush=True)
    assert max_stuck_run < 6, (
        f"robot appears to have gotten stuck (circling in place) for {max_stuck_run * 2}s -- "
        "regression of the fixed-direction wall-avoidance trap"
    )
    print(">>> [rwtest] PASS: no extended stuck/circling-in-place window detected", flush=True)

    wall_limit_x = ROOM_HALF_X + 0.3  # small tolerance past the inner wall face
    wall_limit_y = ROOM_HALF_Y + 0.3
    print(f">>> [rwtest] max |x|={max_abs_x:.3f} (room half-extent {ROOM_HALF_X}), "
          f"max |y|={max_abs_y:.3f} (room half-extent {ROOM_HALF_Y})", flush=True)
    assert max_abs_x < wall_limit_x, f"robot escaped the room in X: max|x|={max_abs_x:.3f}"
    assert max_abs_y < wall_limit_y, f"robot escaped the room in Y: max|y|={max_abs_y:.3f}"
    print(">>> [rwtest] PASS: wall-avoidance kept the robot within the room's footprint", flush=True)

    print(">>> [rwtest] ALL RANDOM WALK TESTS PASSED", flush=True)
    simulation_app.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
        simulation_app.close()
        raise
