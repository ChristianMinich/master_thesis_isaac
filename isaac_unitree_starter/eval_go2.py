"""Minimal evaluation / rollout script for the Go2 Isaac Lab environment.

Runs a short rollout with a placeholder policy and prints basic episode
statistics: total reward, episode length, success flag, and termination
reason. Replace `policy()` with a trained policy for real evaluation.

Run with the Isaac Lab python interpreter, e.g.:
    ./isaaclab.sh -p isaac_unitree_starter/eval_go2.py --headless
"""

import argparse

from isaaclab.app import AppLauncher

# -- launch the simulation app first (before importing anything else) --
parser = argparse.ArgumentParser(description="Evaluate a policy on Go2.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# -- everything else can be imported now --
import torch

from go2_lab_env import Go2Env, Go2EnvCfg

# ---------------------------------------------------------------------------
# Inline configuration
# ---------------------------------------------------------------------------
NUM_ENVS = 1
MAX_STEPS = 400
SUCCESS_DIST = 0.5  # counted as success if the robot gets this close to goal


def policy(obs: torch.Tensor, num_actions: int, device: str) -> torch.Tensor:
    """Placeholder policy (zero actions -> hold default pose).

    Replace with a trained policy, e.g. load a checkpoint and run inference.
    """
    return torch.zeros(obs.shape[0], num_actions, device=device)


def main():
    env_cfg = Go2EnvCfg()
    env_cfg.scene.num_envs = NUM_ENVS
    env = Go2Env(cfg=env_cfg)

    obs_dict, _ = env.reset()
    obs = obs_dict["policy"]

    total_reward = torch.zeros(NUM_ENVS, device=env.device)
    episode_length = 0
    success = False
    termination_reason = "max_steps_reached"

    for step in range(MAX_STEPS):
        action = policy(obs, env.cfg.action_space, env.device)
        obs_dict, reward, terminated, truncated, _ = env.step(action)
        obs = obs_dict["policy"]

        total_reward += reward
        episode_length = step + 1

        # check success (close to goal)
        dist = env._dist_to_goal()
        if bool((dist < SUCCESS_DIST).any()):
            success = True
            termination_reason = "goal_reached"
            break

        if bool(terminated.any()):
            # distinguish fallen vs out-of-bounds using the same checks as the env
            base_height = env._robot.data.root_pos_w[:, 2]
            if bool((base_height < env.cfg.min_base_height).any()):
                termination_reason = "fallen"
            else:
                termination_reason = "out_of_bounds"
            break

        if bool(truncated.any()):
            termination_reason = "timeout"
            break

    print("=" * 50)
    print("Rollout summary")
    print("=" * 50)
    print(f"  total reward       : {total_reward.mean().item():.3f}")
    print(f"  episode length     : {episode_length} steps")
    print(f"  success            : {success}")
    print(f"  termination reason : {termination_reason}")
    print(f"  final dist to goal : {env._dist_to_goal().mean().item():.3f} m")
    print("=" * 50)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()