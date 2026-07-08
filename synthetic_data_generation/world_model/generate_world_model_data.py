"""Generate synthetic world-model transition data with the Go2.

Collects flat (observation, action, next_observation, reward, done)
transitions — the canonical training format for predictive dynamics models
(Dreamer-style latent dynamics, JEPA-style predictors, or a simple
encoder-transition-predictor). Data is written as chunked .npz files via
``TransitionWriter``; episode boundaries are encoded by the ``done`` flag,
so a trainer must never predict across a (s_T, a, s_0) reset transition.

Stored per chunk (.npz, shapes are (steps, num_envs, ...)):
    observation      : proprioceptive observation vector (48-dim like Go2Env)
    action           : applied action in [-1, 1]
    next_observation : observation after the step
    reward           : per-step reward
    done             : episode termination flags (reset boundaries!)

TODO(world_model): add "rgb" / "depth" keys once cameras are attached to the
Go2 env — the generic key layout supports this without downstream changes.
TODO(policy): mix in a partially trained policy for nominal-behavior coverage
in addition to random exploration.

Usage (mock, runs anywhere):
    python synthetic_data_generation/world_model/generate_world_model_data.py --backend mock
Usage (Isaac Lab):
    ./isaaclab.sh -p synthetic_data_generation/world_model/generate_world_model_data.py \
        --backend isaac --headless
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate synthetic world-model transition data with Go2."
    )
    parser.add_argument("--config", type=str, default=None, help="Optional YAML config file.")
    parser.add_argument("--backend", type=str, default="mock", choices=["mock", "isaac"])
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--total-transitions", type=int, default=5000,
                        help="Total (s, a, s') tuples to collect across all envs.")
    parser.add_argument("--chunk-size", type=int, default=1000,
                        help="Transitions per saved .npz chunk file.")
    parser.add_argument("--policy", type=str, default="random",
                        choices=["random", "zero", "scripted_goal"])
    parser.add_argument("--noise-std", type=float, default=0.3,
                        help="Exploration noise for zero/scripted policies.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="data/synthetic/world_model_go2")
    return parser


def main() -> None:
    parser = build_parser()

    # Isaac Lab needs its app launched before anything else is imported.
    prelim, _ = parser.parse_known_args()
    simulation_app = None
    if prelim.backend == "isaac":
        from isaaclab.app import AppLauncher

        AppLauncher.add_app_launcher_args(parser)
        args = parser.parse_args()
        simulation_app = AppLauncher(args).app
    else:
        from synthetic_data_generation.common.config import parse_args_with_config

        args = parse_args_with_config(parser)

    import numpy as np

    from synthetic_data_generation.common.backends import make_backend
    from synthetic_data_generation.common.policies import make_policy
    from synthetic_data_generation.common.writer import TransitionWriter

    backend = make_backend(args.backend, num_envs=args.num_envs, seed=args.seed)
    policy = make_policy(args.policy, backend.action_dim, seed=args.seed,
                         noise_std=args.noise_std)
    writer = TransitionWriter(
        args.output_dir,
        chunk_size=args.chunk_size,
        metadata=vars(args) | {"approach": "world_model"},
    )

    obs, info = backend.reset(seed=args.seed)
    collected = 0

    while collected < args.total_transitions:
        action = policy(obs, info)
        next_obs, reward, done, info = backend.step(action)

        writer.add(
            {
                "observation": obs,
                "action": np.asarray(action),
                "next_observation": next_obs,
                "reward": reward,
                "done": done,
            }
        )

        # note: backends auto-reset finished envs, so when done[i] is True,
        # next_obs[i] already belongs to a NEW episode. The stored `done`
        # flag marks exactly these boundaries for the world-model trainer.
        obs = next_obs
        collected += args.num_envs

        if collected % (args.chunk_size * 5) < args.num_envs:
            print(f"[INFO] Collected {collected}/{args.total_transitions} transitions ...")

    manifest = writer.finalize()
    print(f"[INFO] Done. Collected ~{collected} transitions. Manifest: {manifest}")

    backend.close()
    if simulation_app is not None:
        simulation_app.close()


if __name__ == "__main__":
    main()