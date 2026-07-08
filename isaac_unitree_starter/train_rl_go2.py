"""Minimal RL training script for the Go2 Isaac Lab environment.

Uses skrl PPO (the simplest Isaac Lab compatible RL workflow). To swap the
algorithm later, replace the agent construction block below.

Run with the Isaac Lab python interpreter, e.g.:
    ./isaaclab.sh -p isaac_unitree_starter/train_rl_go2.py --headless
"""

import argparse

from isaaclab.app import AppLauncher

# -- launch the simulation app first (before importing anything else) --
parser = argparse.ArgumentParser(description="Train PPO on Go2.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# -- everything else can be imported now --
import torch
import torch.nn as nn

from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.trainers.torch import SequentialTrainer

from isaaclab_rl.skrl import SkrlVecEnvWrapper

from go2_lab_env import Go2Env, Go2EnvCfg

# ---------------------------------------------------------------------------
# Inline hyperparameters (minimal, just to verify training starts)
# ---------------------------------------------------------------------------
NUM_ENVS = 64
TOTAL_TIMESTEPS = 5_000
ROLLOUT_STEPS = 24
LEARNING_RATE = 3e-4
HIDDEN_SIZES = (256, 128)


# ---------------------------------------------------------------------------
# Simple MLP policy and value models
# ---------------------------------------------------------------------------
def make_mlp(in_dim: int, out_dim: int) -> nn.Sequential:
    layers = []
    prev = in_dim
    for h in HIDDEN_SIZES:
        layers += [nn.Linear(prev, h), nn.ELU()]
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


class Policy(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions=True)
        self.net = make_mlp(self.num_observations, self.num_actions)
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs, role):
        return self.net(inputs["states"]), self.log_std_parameter, {}


class Value(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self)
        self.net = make_mlp(self.num_observations, 1)

    def compute(self, inputs, role):
        return self.net(inputs["states"]), {}


def main():
    # environment
    env_cfg = Go2EnvCfg()
    env_cfg.scene.num_envs = NUM_ENVS
    env = Go2Env(cfg=env_cfg)
    env = SkrlVecEnvWrapper(env)

    device = env.device

    # models
    models = {
        "policy": Policy(env.observation_space, env.action_space, device),
        "value": Value(env.observation_space, env.action_space, device),
    }

    # memory
    memory = RandomMemory(memory_size=ROLLOUT_STEPS, num_envs=env.num_envs, device=device)

    # PPO agent -- replace this block to use a different algorithm
    agent_cfg = PPO_DEFAULT_CONFIG.copy()
    agent_cfg.update(
        {
            "rollouts": ROLLOUT_STEPS,
            "learning_epochs": 5,
            "mini_batches": 4,
            "discount_factor": 0.99,
            "lambda": 0.95,
            "learning_rate": LEARNING_RATE,
            "entropy_loss_scale": 0.01,
            "experiment": {"directory": "runs/go2_ppo", "write_interval": 100},
        }
    )
    agent = PPO(
        models=models,
        memory=memory,
        cfg=agent_cfg,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
    )

    # train
    trainer = SequentialTrainer(cfg={"timesteps": TOTAL_TIMESTEPS, "headless": True}, env=env, agents=agent)
    trainer.train()

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()