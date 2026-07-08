"""Minimal VLA-style data collection for the Go2 Isaac Lab environment.

Runs the environment with a placeholder policy (random or scripted actions)
and saves per-step data (observation, action, reward, done, goal) as .npz
episode files. The record dict is structured so RGB, depth, LiDAR, language
commands, or expert actions can be added later without changing the format.

Run with the Isaac Lab python interpreter, e.g.:
    ./isaaclab.sh -p isaac_unitree_starter/collect_vla_data_go2.py --headless
"""

import argparse

from isaaclab.app import AppLauncher

# -- launch the simulation app first (before importing anything else) --
parser = argparse.ArgumentParser(description="Collect VLA-style data with Go2.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# -- everything else can be imported now --
import os

import numpy as np
import torch

from go2_lab_env import Go2Env, Go2EnvCfg

# ---------------------------------------------------------------------------
# Inline configuration
# ---------------------------------------------------------------------------
NUM_ENVS = 4
NUM_EPISODES = 3
MAX_STEPS_PER_EPISODE = 300
OUTPUT_DIR = "data/vla_go2"
POLICY_MODE = "random"  # "random" | "zero"  (replace with a real policy later)


def placeholder_policy(obs: torch.Tensor, num_actions: int, device: str) -> torch.Tensor:
    """Placeholder action source. Replace with a trained or scripted policy."""
    if POLICY_MODE == "random":
        return torch.rand(obs.shape[0], num_actions, device=device) * 2.0 - 1.0
    return torch.zeros(obs.shape[0], num_actions, device=device)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    env_cfg = Go2EnvCfg()
    env_cfg.scene.num_envs = NUM_ENVS
    env = Go2Env(cfg=env_cfg)

    for episode in range(NUM_EPISODES):
        obs_dict, _ = env.reset()
        obs = obs_dict["policy"]

        # per-step buffers; extend with "rgb", "depth", "lidar",
        # "language_command", "expert_action", ... later
        buffers = {
            "observation": [],
            "action": [],
            "reward": [],
            "done": [],
            "goal_pos": [],
        }

        for _ in range(MAX_STEPS_PER_EPISODE):
            action = placeholder_policy(obs, env.cfg.action_space, env.device)

            next_obs_dict, reward, terminated, truncated, _ = env.step(action)
            done = terminated | truncated

            buffers["observation"].append(obs.cpu().numpy())
            buffers["action"].append(action.cpu().numpy())
            buffers["reward"].append(reward.cpu().numpy())
            buffers["done"].append(done.cpu().numpy())
            buffers["goal_pos"].append(env._goal_pos.cpu().numpy())

            obs = next_obs_dict["policy"]

            if done.all():
                break

        # stack to arrays of shape (T, num_envs, ...) and save one file per episode
        data = {key: np.stack(vals, axis=0) for key, vals in buffers.items()}
        out_path = os.path.join(OUTPUT_DIR, f"episode_{episode:04d}.npz")
        np.savez_compressed(out_path, **data)
        print(
            f"[INFO] Saved episode {episode}: {data['observation'].shape[0]} steps, "
            f"{NUM_ENVS} envs -> {out_path}"
        )

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()