"""Generate synthetic SLAM training/evaluation data with the Go2.

Produces trajectory episodes containing ground-truth poses plus a drifting
noisy pose estimate (stand-in for a real SLAM/odometry front-end). This data
trains/evaluates the pose-monitoring SLAM baseline and is directly usable
with the ATE/RPE metrics in ``isaac_unitree_starter/slam_pose_eval_stub.py``.

Stored per episode (.npz, shapes are (T, num_envs, ...)):
    observation   : proprioceptive observation vector
    action        : applied action
    gt_position   : (T, N, 3) ground-truth base position (env-local frame)
    est_position  : (T, N, 3) noisy/drifting pose estimate
    done          : episode termination flags

TODO(slam): when a camera is added to the Go2 env, also record "rgb" /
"depth" + intrinsics here and feed them to ORB-SLAM3 instead of the
NoisyPoseEstimator.

Usage (mock, runs anywhere):
    python synthetic_data_generation/slam/generate_slam_data.py --backend mock
Usage (Isaac Lab):
    ./isaaclab.sh -p synthetic_data_generation/slam/generate_slam_data.py \
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
    parser = argparse.ArgumentParser(description="Generate synthetic SLAM pose data with Go2.")
    parser.add_argument("--config", type=str, default=None, help="Optional YAML config file.")
    parser.add_argument("--backend", type=str, default="mock", choices=["mock", "isaac"])
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--num-episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--policy", type=str, default="scripted_goal",
                        choices=["random", "zero", "scripted_goal"])
    parser.add_argument("--pose-noise-std", type=float, default=0.005,
                        help="Per-step drift noise of the simulated pose estimator.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="data/synthetic/slam_go2")
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
    policy = make_policy(args.policy, backend.action_dim, seed=args.seed)
    writer = EpisodeWriter(args.output_dir, metadata=vars(args) | {"approach": "slam"})

    rng = np.random.default_rng(args.seed)

    for episode in range(args.num_episodes):
        obs, info = backend.reset(seed=args.seed + episode)

        # one independent drift accumulator per parallel env
        drift = np.zeros((args.num_envs, 3))

        buffers: dict[str, list[np.ndarray]] = {
            "observation": [], "action": [], "gt_position": [],
            "est_position": [], "done": [],
        }

        for _ in range(args.max_steps):
            action = policy(obs, info)
            next_obs, _reward, done, info = backend.step(action)

            gt_pos = info["robot_pos"]
            drift += rng.normal(0.0, args.pose_noise_std, size=drift.shape)
            est_pos = gt_pos + drift

            buffers["observation"].append(obs)
            buffers["action"].append(np.asarray(action))
            buffers["gt_position"].append(gt_pos)
            buffers["est_position"].append(est_pos)
            buffers["done"].append(done)

            obs = next_obs
            if done.all():
                break

        path = writer.write_episode(buffers)
        print(f"[INFO] SLAM episode {episode}: {len(buffers['observation'])} steps -> {path}")

    manifest = writer.finalize()
    print(f"[INFO] Done. Manifest: {manifest}")

    backend.close()
    if simulation_app is not None:
        simulation_app.close()


if __name__ == "__main__":
    main()