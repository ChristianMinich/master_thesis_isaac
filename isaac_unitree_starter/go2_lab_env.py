"""Minimal Isaac Lab direct-workflow environment for the Unitree Go2.

Suitable as a base for RL training, VLA-style data collection, or navigation
experiments. Observations, rewards, and terminations are kept in small,
separate methods so they can be swapped out later.

This module only defines the environment. It is imported by
`train_rl_go2.py`, `collect_vla_data_go2.py`, and `eval_go2.py`, which are
responsible for launching the simulation app first.
"""

from __future__ import annotations

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

# Pre-configured Go2 asset shipped with Isaac Lab. If your Isaac Lab version
# uses a different module layout, adjust this import (this is the most
# version-dependent line in the file).
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG  # noqa: F401


@configclass
class Go2EnvCfg(DirectRLEnvCfg):
    """Inline configuration for the Go2 environment."""

    # env
    decimation = 4
    episode_length_s = 20.0
    action_scale = 0.25  # joint position offsets around default pose
    action_space = 12
    observation_space = 48
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 200, render_interval=4)

    # terrain (flat plane)
    terrain: TerrainImporterCfg = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0, dynamic_friction=1.0, restitution=0.0
        ),
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=64, env_spacing=4.0, replicate_physics=True
    )

    # robot
    robot: ArticulationCfg = UNITREE_GO2_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # task
    goal_radius = 8.0  # goals sampled on a circle of this radius around origin
    min_base_height = 0.20  # terminate if base drops below this (fallen)
    max_dist_from_origin = 15.0  # terminate if robot leaves allowed area

    # reward weights (replace/extend later)
    rew_progress_weight = 2.0
    rew_alive_weight = 0.5
    rew_action_rate_weight = -0.01
    rew_joint_vel_weight = -0.0005


class Go2Env(DirectRLEnv):
    """Minimal goal-directed locomotion environment for the Unitree Go2."""

    cfg: Go2EnvCfg

    def __init__(self, cfg: Go2EnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        self._previous_actions = torch.zeros_like(self._actions)

        # goal position per environment (xy, in env-local frame)
        self._goal_pos = torch.zeros(self.num_envs, 2, device=self.device)

        # cached default joint positions used as action offset center
        self._default_joint_pos = self._robot.data.default_joint_pos.clone()

        # distance to goal at previous step (for progress reward)
        self._prev_dist_to_goal = torch.zeros(self.num_envs, device=self.device)

    # ------------------------------------------------------------------
    # Scene setup
    # ------------------------------------------------------------------
    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        # terrain
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # clone environments and add a light
        self.scene.clone_environments(copy_from_source=False)
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------
    def _pre_physics_step(self, actions: torch.Tensor):
        self._previous_actions = self._actions.clone()
        self._actions = actions.clone().clamp(-1.0, 1.0)

    def _apply_action(self):
        # joint position targets = default pose + scaled action
        targets = self._default_joint_pos + self.cfg.action_scale * self._actions
        self._robot.set_joint_position_target(targets)

    # ------------------------------------------------------------------
    # Observations (replace/extend later: add RGB, depth, LiDAR, etc.)
    # ------------------------------------------------------------------
    def _get_observations(self) -> dict:
        base_lin_vel = self._robot.data.root_lin_vel_b  # (N, 3)
        base_ang_vel = self._robot.data.root_ang_vel_b  # (N, 3)
        joint_pos = self._robot.data.joint_pos - self._default_joint_pos  # (N, 12)
        joint_vel = self._robot.data.joint_vel  # (N, 12)
        goal_dir = self._goal_direction()  # (N, 2)
        dist = self._dist_to_goal().unsqueeze(-1)  # (N, 1)

        obs = torch.cat(
            [
                base_lin_vel,          # 3
                base_ang_vel,          # 3
                joint_pos,             # 12
                joint_vel,             # 12
                self._previous_actions,  # 12
                goal_dir,              # 2
                dist,                  # 1
                self._projected_gravity(),  # 3
            ],
            dim=-1,
        )
        return {"policy": obs}

    def _projected_gravity(self) -> torch.Tensor:
        return self._robot.data.projected_gravity_b

    def _base_pos_local(self) -> torch.Tensor:
        """Robot base xy position in the env-local frame."""
        return self._robot.data.root_pos_w[:, :2] - self.scene.env_origins[:, :2]

    def _dist_to_goal(self) -> torch.Tensor:
        return torch.norm(self._goal_pos - self._base_pos_local(), dim=-1)

    def _goal_direction(self) -> torch.Tensor:
        vec = self._goal_pos - self._base_pos_local()
        return vec / (torch.norm(vec, dim=-1, keepdim=True) + 1e-6)

    # ------------------------------------------------------------------
    # Rewards (replace later with task-specific terms)
    # ------------------------------------------------------------------
    def _get_rewards(self) -> torch.Tensor:
        dist = self._dist_to_goal()
        progress = self._prev_dist_to_goal - dist
        self._prev_dist_to_goal = dist

        action_rate = torch.sum(torch.square(self._actions - self._previous_actions), dim=-1)
        joint_vel_penalty = torch.sum(torch.square(self._robot.data.joint_vel), dim=-1)

        reward = (
            self.cfg.rew_progress_weight * progress
            + self.cfg.rew_alive_weight * self.step_dt
            + self.cfg.rew_action_rate_weight * action_rate
            + self.cfg.rew_joint_vel_weight * joint_vel_penalty
        )
        return reward

    # ------------------------------------------------------------------
    # Terminations: fallen, out of bounds, timeout
    # ------------------------------------------------------------------
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        base_height = self._robot.data.root_pos_w[:, 2]
        fallen = base_height < self.cfg.min_base_height

        out_of_bounds = torch.norm(self._base_pos_local(), dim=-1) > self.cfg.max_dist_from_origin

        terminated = fallen | out_of_bounds
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, time_out

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES
        super()._reset_idx(env_ids)

        # reset robot root state to default, offset by env origin
        root_state = self._robot.data.default_root_state[env_ids].clone()
        root_state[:, :3] += self.scene.env_origins[env_ids]
        self._robot.write_root_pose_to_sim(root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids)

        # reset joints to default
        joint_pos = self._robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self._robot.data.default_joint_vel[env_ids].clone()
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # reset action buffers
        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0

        # sample a new goal on a circle around the env origin
        angle = torch.rand(len(env_ids), device=self.device) * 2.0 * torch.pi
        self._goal_pos[env_ids, 0] = self.cfg.goal_radius * torch.cos(angle)
        self._goal_pos[env_ids, 1] = self.cfg.goal_radius * torch.sin(angle)

        self._prev_dist_to_goal[env_ids] = torch.norm(
            self._goal_pos[env_ids] - self._base_pos_local()[env_ids], dim=-1
        )