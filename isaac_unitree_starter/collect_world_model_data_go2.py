"""Minimal world-model transition data collection for the Go2 Isaac Lab env.

Collects (observation, action, next_observation, reward, done) transitions,
which is the canonical training format for predictive dynamics / world models
(Dreamer-style latent dynamics, JEPA-style predictors, or a simple
encoder-transition-predictor). Data is saved as .npz chunk files.

The stored keys are deliberately generic so that image observations
("rgb", "depth") or privileged state can be added later without changing
the downstream loading code.

Run with the Isaac Lab python interpreter, e.g.:
    ./isaaclab.sh -p isaac_unitree_starter/collect_world_model_data_go2.py --headless
"""

import argparse

from isaaclab.app import AppLauncher

# -- launch the simulation app first (before importing anything else) --
parser = argparse.ArgumentParser(description="Collect world-model transition data with Go2.")
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
NUM_ENVS = 8
TOTAL_TRANSITIONS = 5_000  # total (s, a, s') tuples to collect across all envs
CHUNK_SIZE = 1_000         # transitions per saved .npz file
OUTPUT_DIR = "data/world_model_go2"
POLICY_MODE = "random"     # "random" | "zero"  (replace with a trained policy later)
ACTION_NOISE_STD = 0.3     # exploration noise if POLICY_MODE == "zero"


def exploration_policy(obs: torch.Tensor, num_actions: int, device: str) -> torch.Tensor:
    """Action source for data collection.

    For world-model training you usually want broad state coverage, so random
    or noisy actions are a reasonable starting point. Later, replace this with
    a trained RL policy plus exploration noise to collect *nominal* navigation
    behavior (as planned in the thesis expose).
    """
    if POLICY_MODE == "random":
        return torch.rand(obs.shape[0], num_actions, device=device) * 2.0 - 1.0
    noise = torch.randn(obs.shape[0], num_actions, device=device) * ACTION_NOISE_STD
    return noise.clamp(-1.0, 1.0)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    env_cfg = Go2EnvCfg()
    env_cfg.scene.num_envs = NUM_ENVS
    env = Go2Env(cfg=env_cfg)

    obs_dict, _ = env.reset()
    obs = obs_dict["policy"]

    # transition buffers; extend with "rgb", "depth", "robot_pose", ... later
    buffers = {
        "observation": [],
        "action": [],
        "next_observation": [],
        "reward": [],
        "done": [],
    }

    collected = 0
    chunk_index = 0

    while collected < TOTAL_TRANSITIONS:
        action = exploration_policy(obs, env.cfg.action_space, env.device)
        next_obs_dict, reward, terminated, truncated, _ = env.step(action)
        next_obs = next_obs_dict["policy"]
        done = terminated | truncated

        buffers["observation"].append(obs.cpu().numpy())
        buffers["action"].append(action.cpu().numpy())
        buffers["next_observation"].append(next_obs.cpu().numpy())
        buffers["reward"].append(reward.cpu().numpy())
        buffers["done"].append(done.cpu().numpy())

        # note: `done` flags mark episode boundaries -- a world model trainer
        # must not predict across a (s_T, a, s_0) reset transition.
        obs = next_obs
        collected += NUM_ENVS

        # flush a chunk to disk
        if len(buffers["observation"]) * NUM_ENVS >= CHUNK_SIZE:
            data = {key: np.stack(vals, axis=0) for key, vals in buffers.items()}
            out_path = os.path.join(OUTPUT_DIR, f"transitions_{chunk_index:04d}.npz")
            np.savez_compressed(out_path, **data)
            print(
                f"[INFO] Saved chunk {chunk_index}: "
                f"{data['observation'].shape[0]} steps x {NUM_ENVS} envs -> {out_path}"
            )
            buffers = {key: [] for key in buffers}
            chunk_index += 1

    # flush remainder
    if buffers["observation"]:
        data = {key: np.stack(vals, axis=0) for key, vals in buffers.items()}
        out_path = os.path.join(OUTPUT_DIR, f"transitions_{chunk_index:04d}.npz")
        np.savez_compressed(out_path, **data)
        print(f"[INFO] Saved final chunk {chunk_index} -> {out_path}")

    print(f"[INFO] Done. Collected ~{collected} transitions into {OUTPUT_DIR}/")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()