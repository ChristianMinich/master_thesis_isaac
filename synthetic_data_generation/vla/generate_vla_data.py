"""Generate synthetic VLA (vision-language-action) training data with the Go2.

Produces language-conditioned demonstration episodes: each episode pairs a
templated natural-language instruction (derived from the sampled goal
direction) with the observation/action stream of a goal-reaching rollout.
This is the canonical (instruction, trajectory) format for fine-tuning
OpenVLA/Octo-style models.

Stored per episode (.npz, shapes are (T, num_envs, ...)):
    observation : proprioceptive observation vector (48-dim like Go2Env)
    action      : applied action in [-1, 1]
    reward      : per-step reward
    done        : episode termination flags
    goal_pos    : (T, N, 2) active goal (env-local frame)
    instruction : (N,) language instruction per parallel env (unicode array)

TODO(vla): add camera sensors to the Go2 env and record "rgb" frames here —
real VLA training needs images. TODO(policy): replace the scripted expert
with a trained locomotion policy for realistic demonstrations.

Usage (mock, runs anywhere):
    python synthetic_data_generation/vla/generate_vla_data.py --backend mock
Usage (Isaac Lab):
    ./isaaclab.sh -p synthetic_data_generation/vla/generate_vla_data.py \
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
    parser = argparse.ArgumentParser(description="Generate synthetic VLA data with Go2.")
    parser.add_argument("--config", type=str, default=None, help="Optional YAML config file.")
    parser.add_argument("--backend", type=str, default="mock", choices=["mock", "isaac"])
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--num-episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--policy", type=str, default="scripted_goal",
                        choices=["random", "zero", "scripted_goal"])
    parser.add_argument("--noise-std", type=float, default=0.05,
                        help="Exploration noise for zero/scripted policies.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="data/synthetic/vla_go2")
    return parser


def instruction_from_goal(goal_xy) -> str:
    """Templated language instruction derived from the goal position.

    TODO(vla): diversify with an LLM-based paraphraser for language coverage.
    """
    import numpy as np

    x, y = float(goal_xy[0]), float(goal_xy[1])
    angle = np.degrees(np.arctan2(y, x)) % 360.0
    dist = float(np.hypot(x, y))
    directions = [
        (22.5, "forward"), (67.5, "forward-left"), (112.5, "left"),
        (157.5, "backward-left"), (202.5, "backward"), (247.5, "backward-right"),
        (292.5, "right"), (337.5, "forward-right"), (360.0, "forward"),
    ]
    for upper, name in directions:
        if angle <= upper:
            return f"walk {name} for about {dist:.0f} meters and stop at the goal"
    return f"walk to the goal {dist:.0f} meters away"


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
    writer = EpisodeWriter(args.output_dir, metadata=vars(args) | {"approach": "vla"})

    for episode in range(args.num_episodes):
        obs, info = backend.reset(seed=args.seed + episode)

        # one instruction per parallel env, fixed for the episode
        instructions = np.array(
            [instruction_from_goal(info["goal_pos"][i]) for i in range(args.num_envs)]
        )

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
            # TODO(vla): buffers["rgb"].append(camera_frame) once cameras exist

            obs = next_obs
            if done.all():
                break

        arrays = {key: np.stack(vals, axis=0) for key, vals in buffers.items()}
        arrays["instruction"] = instructions
        path = writer.output_dir / f"episode_{episode:05d}.npz"
        np.savez_compressed(path, **arrays)
        writer._register_file(path, arrays, episode_index=episode)
        print(
            f"[INFO] VLA episode {episode}: {arrays['observation'].shape[0]} steps, "
            f"instruction[0]='{instructions[0]}' -> {path}"
        )

    manifest = writer.finalize()
    print(f"[INFO] Done. Manifest: {manifest}")

    backend.close()
    if simulation_app is not None:
        simulation_app.close()


if __name__ == "__main__":
    main()