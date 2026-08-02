"""Build a registered scene (see scenes/registry.py) once and save it to a single
reloadable USD file, so future runs can skip re-downloading/re-authoring it.

Usage:
    <isaacsim>/python.sh save_scene.py [--scene room] [--out saved_scenes/go2_room_scene.usd]

Deliberately does NOT save the Go2 robot itself: main.py always spawns Go2Robot
fresh (it's fast, and it must run its own Python-side init -- policy loading,
physics-tensor-valid checks -- regardless of whether the prim pre-existed). Saving
and reloading an already-physics-initialized CUDA articulation was tried and
caused a native crash on the next simulation step; keeping the save purely to the
static scene sidesteps that entirely.
"""

from isaacsim import SimulationApp

import argparse
import os

parser = argparse.ArgumentParser(description="Save a registered scene to a reloadable USD file")
parser.add_argument("--scene", default="room", help="Which registered scene to build and save (see scenes/registry.py)")
parser.add_argument(
    "--out",
    default=None,
    help="Output .usd path (default: saved_scenes/go2_<scene>_scene.usd)",
)
args, _unknown = parser.parse_known_args()

simulation_app = SimulationApp({"headless": True})

import omni.usd
from isaacsim.core.experimental.utils.stage import define_prim
from isaacsim.core.simulation_manager import SimulationManager

import app_config as config
from scenes.registry import get_scene


def main() -> None:
    SimulationManager.set_backend(config.PHYSICS_BACKEND)
    SimulationManager.set_physics_sim_device(config.PHYSICS_DEVICE)
    SimulationManager.set_physics_dt(config.PHYSICS_DT)

    print(">>> [save_scene] creating physics scene", flush=True)
    define_prim("/World/PhysicsScene", "PhysicsScene")

    print(f">>> [save_scene] building scene: {args.scene!r}", flush=True)
    scene = get_scene(args.scene)
    scene.build()

    out_path = args.out or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "saved_scenes", f"go2_{args.scene}_scene.usd"
    )
    out_path = os.path.abspath(out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    stage = omni.usd.get_context().get_stage()
    print(f">>> [save_scene] exporting stage to {out_path}", flush=True)
    stage.Export(out_path)

    print(">>> [save_scene] DONE", flush=True)
    simulation_app.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
        simulation_app.close()
        raise
