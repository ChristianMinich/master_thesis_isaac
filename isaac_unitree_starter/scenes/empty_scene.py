"""EmptyScene: just the default ground plane, no walls/furniture.

Useful for quick headless validation and as a minimal environment for
Replicator/domain-randomization scripts that add their own randomized content.
"""

from __future__ import annotations

import carb
import isaacsim.core.experimental.utils.stage as stage_utils
from isaacsim.storage.native import get_assets_root_path

from core.scene import Scene, SceneInfo

GROUND_PRIM_PATH = "/World/ground"
GROUND_ENV_USD_RELATIVE_PATH = "/Isaac/Environments/Grid/default_environment.usd"
DEFAULT_SPAWN_POSITION = [0.0, 0.0, 0.5]


class EmptyScene(Scene):
    """Just the default Isaac Sim ground-plane environment, nothing else."""

    name = "empty"

    def build(self) -> SceneInfo:
        assets_root_path = get_assets_root_path()
        if assets_root_path is None:
            carb.log_error("Could not find Isaac Sim assets folder")
            raise RuntimeError("Isaac Sim assets root path not found")

        stage_utils.add_reference_to_stage(
            usd_path=assets_root_path + GROUND_ENV_USD_RELATIVE_PATH,
            path=GROUND_PRIM_PATH,
        )

        return SceneInfo(assets_root_path=assets_root_path, default_spawn_position=list(DEFAULT_SPAWN_POSITION))
