"""Generate synthetic RL rollout data with the Go2.

RL policies (PPO/SAC) are normally trained online (see
``isaac_unitree_starter/train_rl_go2.py``), but offline rollouts are still
useful for: offline RL / behavior cloning warm-starts, reward-model sanity
checks, and replay-buffer pre-filling.

Stored per episode (.npz, shapes are (T, num_envs, ...)):
    observation : proprioceptive observation vector (48-dim like Go2Env)
    action      : applied action in [-1, 1]
    reward      : per-step reward
    done        : episode termination flags
    goal_pos    : (T, N, 2) active goal (env-local frame)

TODO(rl): swap the placeholder policy for a partially trained checkpoint to
generate on-policy replay data instead of random exploration.

Usage (mock, runs anywhere):
    python synthetic_data_generation/rl/generate_rl_data.py --backend mock
Usage (Isaac Lab):
    ./isaaclab.sh -p synthetic_data_generation/rl/generate_rl_data.py \
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
    parser = argparse.ArgumentParser(description="Generate synthetic RL rollout data with Go2.")
    parser.add_argument("--config", type=str, default=None, help="Optional YAML config file.")
    parser.add_argument("--backend", type=str, default="mock", choices=["mock", "isaac"])
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--num-episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--policy", type=str, default="random",
                        choices=["random", "zero", "scripted_goal"])
    parser.add_argument("--noise-std", type=float, default=0.1,
                        help="Exploration noise for zero/scripted policies.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="data/synthetic/rl_go2")
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
    from synthetic_data_generation.common.writer import EpisodeWriter

    backend = make_backend(args.backend, num_envs=args.num_envs, seed=args.seed)
    policy = make_policy(args.policy, backend.action_dim, seed=args.seed,
                         noise_std=args.noise_std)
    writer = EpisodeWriter(args.output_dir, metadata=vars(args) | {"approach": "rl"})

    for episode in range(args.num_episodes):
        obs, info = backend.reset(seed=args.seed + episode)

        buffers: dict[str, list[np.ndarray]] = {
            "observation": [], "action": [], "reward": [], "done": [], "goal_pos": [],
        }

        for _ in range(args.max_steps):
            action = policy(obs, info)
            next_obs, reward, done, info = backend.step(action)

            buffers["observation"].append(obs)
            buffers["action"].append(np.asarray(action))
            buffers["reward"].append(reward)
            buffers["done"].append(done)
            buffers["goal_pos"].append(info["goal_pos"])

            obs = next_obs
            if done.all():
                break

        total_return = float(np.sum(np.stack(buffers["reward"])))
        path = writer.write_episode(buffers, total_return=total_return)
        print(
            f"[INFO] RL episode {episode}: {len(buffers['observation'])} steps, "
            f"return={total_return:.2f} -> {path}"
        )

    manifest = writer.finalize()
    print(f"[INFO] Done. Manifest: {manifest}")

    backend.close()
    if simulation_app is not None:
        simulation_app.close()


if __name__ == "__main__":
    main()