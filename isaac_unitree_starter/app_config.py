"""Truly global simulation settings -- shared by every scene/robot/controller,
not owned by any one component.

Everything scene-, robot-, or controller-specific now lives inside that
component's own module (see scenes/room_scene.py, robots/go2/*, controllers/*,
live_viewer.py, data_logger.py) so each is self-contained/portable on its own.
"""

from __future__ import annotations

# Matches the physics rate the official Go2 flat-terrain policy was trained
# with (isaacsim.robot.policy.examples interactive Go2 example).
PHYSICS_DT = 1.0 / 200.0  # 200 Hz physics
RENDERING_DT = 1.0 / 50.0  # 50 Hz rendering (4 physics steps per render)
PHYSICS_DEVICE = "cuda"
PHYSICS_BACKEND = "torch"
