"""Go2Robot: the reusable unit -- Go2 policy + sensors + data logging bundled
behind one constructor.

Independent of which Scene it's placed in: only needs a prim_path/spawn
position, so the same Go2Robot works in RoomScene, EmptyScene, a future scene,
or a Replicator randomization script.
"""

from __future__ import annotations

from data_logger import DataLogger, DEFAULT_RECORDINGS_ROOT
from robots.go2.go2_controller import DEFAULT_PRIM_PATH, Go2Controller
from robots.go2.sensor_manager import (
    CAMERA_HEIGHT,
    CAMERA_WIDTH,
    FOOT_LINK_RELATIVE_PATHS,
    SensorManager,
    transform_prim_paths,
)


class Go2Robot:
    """Owns one Go2: articulation + locomotion policy + sensors + a data logger."""

    def __init__(
        self,
        prim_path: str = DEFAULT_PRIM_PATH,
        position: list[float] | None = None,
        recordings_root: str = DEFAULT_RECORDINGS_ROOT,
    ) -> None:
        self.controller = Go2Controller(prim_path=prim_path, position=position)
        self.controller.spawn()

        self.sensors = SensorManager(base_link_path=self.controller.base_link_path)

        self.logger = DataLogger(
            dof_names=self.controller.policy.robot.dof_names,
            foot_names=list(FOOT_LINK_RELATIVE_PATHS.keys()),
            transform_prim_paths=transform_prim_paths(self.controller.base_link_path),
            session_root=recordings_root,
            extra_metadata={"camera_resolution_hw": [CAMERA_HEIGHT, CAMERA_WIDTH]},
        )

    def on_physics_step(self, dt: float, command) -> None:
        self.controller.on_physics_step(dt, command)

    @property
    def ready(self) -> bool:
        return self.controller.ready

    def get_last_action(self):
        return self.controller.get_last_action()

    def log_step(self, *, step: int, sim_time: float, physics_step: int, command, action) -> None:
        """Record one frame of telemetry, if a recording session is active."""
        self.logger.log_step(
            step=step,
            sim_time=sim_time,
            physics_step=physics_step,
            go2=self.controller,
            sensors=self.sensors,
            command=command,
            action=action,
        )
