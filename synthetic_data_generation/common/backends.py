"""Simulation backends for the Go2 synthetic data generators.

Every generator script talks to a small ``Go2Backend`` interface so the same
script can run:

- ``--backend mock``  : a deterministic kinematic point-robot stand-in for the
  Go2 (numpy only, runs anywhere — laptops, CI). Observation/action layout
  mirrors ``isaac_unitree_starter/go2_lab_env.py`` (obs_dim=48, act_dim=12).
- ``--backend isaac`` : the real Isaac Lab Go2 environment. Requires launching
  through the Isaac Lab interpreter; the AppLauncher must be created by the
  entry script BEFORE this backend is constructed (see generator scripts).

Backend API (numpy in / numpy out):

    obs, info = backend.reset(seed)
    obs, reward, done, info = backend.step(action)   # action: (N, act_dim)

``info`` always contains:
    "goal_direction" : (N, 2) unit vector toward the goal (env-local frame)
    "goal_pos"       : (N, 2) goal position (env-local frame)
    "robot_pos"      : (N, 3) robot base position (env-local frame)
"""

from __future__ import annotations

from typing import Any

import numpy as np

# Observation layout copied from go2_lab_env.Go2Env._get_observations:
#   base_lin_vel(3) base_ang_vel(3) joint_pos(12) joint_vel(12)
#   prev_actions(12) goal_dir(2) dist(1) projected_gravity(3)  => 48
GO2_OBS_DIM = 48
GO2_ACT_DIM = 12


class MockGo2Backend:
    """Kinematic point-robot approximation of the Go2 navigation task.

    Dynamics: ``action[:, 0:2]`` is interpreted as a body-velocity command
    (scaled by ``max_speed``); the base integrates that velocity. Joint values
    are synthesized as a simple gait-like oscillation so the 48-dim
    observation vector is fully populated and shaped like the Isaac env.

    This gives structurally correct (obs, action, reward, done) streams for
    developing and unit-testing data pipelines and models without Isaac.
    """

    def __init__(
        self,
        num_envs: int = 4,
        goal_radius: float = 8.0,
        max_speed: float = 1.5,
        dt: float = 0.02,
        max_episode_steps: int = 500,
        goal_tolerance: float = 0.5,
        max_dist_from_origin: float = 15.0,
        seed: int = 0,
    ):
        self.num_envs = num_envs
        self.obs_dim = GO2_OBS_DIM
        self.action_dim = GO2_ACT_DIM
        self.goal_radius = goal_radius
        self.max_speed = max_speed
        self.dt = dt
        self.max_episode_steps = max_episode_steps
        self.goal_tolerance = goal_tolerance
        self.max_dist_from_origin = max_dist_from_origin
        self._rng = np.random.default_rng(seed)

        self._pos = np.zeros((num_envs, 3))
        self._vel = np.zeros((num_envs, 3))
        self._goal = np.zeros((num_envs, 2))
        self._prev_action = np.zeros((num_envs, self.action_dim))
        self._steps = np.zeros(num_envs, dtype=np.int64)
        self._phase = np.zeros(num_envs)
        self._prev_dist = np.zeros(num_envs)

    # -- helpers ----------------------------------------------------------
    def _dist_to_goal(self) -> np.ndarray:
        return np.linalg.norm(self._goal - self._pos[:, :2], axis=-1)

    def _goal_direction(self) -> np.ndarray:
        vec = self._goal - self._pos[:, :2]
        return vec / (np.linalg.norm(vec, axis=-1, keepdims=True) + 1e-6)

    def _observation(self) -> np.ndarray:
        n = self.num_envs
        base_lin_vel = self._vel
        base_ang_vel = np.zeros((n, 3))
        # synthetic gait oscillation so joint channels carry a signal
        speeds = np.linalg.norm(self._vel[:, :2], axis=-1, keepdims=True)
        gait = np.sin(self._phase[:, None] + np.linspace(0, 2 * np.pi, GO2_ACT_DIM)[None, :])
        joint_pos = 0.1 * gait * np.clip(speeds, 0.0, 1.0)
        joint_vel = 0.5 * np.cos(self._phase[:, None]) * gait
        goal_dir = self._goal_direction()
        dist = self._dist_to_goal()[:, None]
        projected_gravity = np.tile(np.array([0.0, 0.0, -1.0]), (n, 1))
        return np.concatenate(
            [
                base_lin_vel,        # 3
                base_ang_vel,        # 3
                joint_pos,           # 12
                joint_vel,           # 12
                self._prev_action,   # 12
                goal_dir,            # 2
                dist,                # 1
                projected_gravity,   # 3
            ],
            axis=-1,
        ).astype(np.float32)

    def _info(self) -> dict[str, Any]:
        return {
            "goal_direction": self._goal_direction(),
            "goal_pos": self._goal.copy(),
            "robot_pos": self._pos.copy(),
        }

    def _reset_envs(self, mask: np.ndarray) -> None:
        idx = np.where(mask)[0]
        if idx.size == 0:
            return
        self._pos[idx] = 0.0
        self._pos[idx, 2] = 0.35  # nominal Go2 base height
        self._vel[idx] = 0.0
        self._prev_action[idx] = 0.0
        self._steps[idx] = 0
        self._phase[idx] = 0.0
        angle = self._rng.uniform(0.0, 2.0 * np.pi, size=idx.size)
        self._goal[idx, 0] = self.goal_radius * np.cos(angle)
        self._goal[idx, 1] = self.goal_radius * np.sin(angle)
        self._prev_dist[idx] = np.linalg.norm(self._goal[idx] - self._pos[idx, :2], axis=-1)

    # -- API ---------------------------------------------------------------
    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._reset_envs(np.ones(self.num_envs, dtype=bool))
        return self._observation(), self._info()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        command = action[:, :2] * self.max_speed
        self._vel[:, :2] = command
        self._pos[:, :2] += self._vel[:, :2] * self.dt
        self._phase += 2.0 * np.pi * self.dt * (1.0 + np.linalg.norm(command, axis=-1))
        self._steps += 1

        dist = self._dist_to_goal()
        progress = self._prev_dist - dist
        self._prev_dist = dist
        action_rate = np.sum((action - self._prev_action) ** 2, axis=-1)
        reward = 2.0 * progress + 0.5 * self.dt - 0.01 * action_rate
        self._prev_action = action

        reached = dist < self.goal_tolerance
        out_of_bounds = np.linalg.norm(self._pos[:, :2], axis=-1) > self.max_dist_from_origin
        timeout = self._steps >= self.max_episode_steps
        done = reached | out_of_bounds | timeout

        obs = self._observation()
        info = self._info()
        info["reached_goal"] = reached
        self._reset_envs(done)  # auto-reset like vectorized Isaac Lab envs
        return obs, reward.astype(np.float32), done, info

    def close(self) -> None:
        pass


class IsaacGo2Backend:
    """Adapter around the Isaac Lab ``Go2Env``.

    IMPORTANT: the Isaac Lab AppLauncher must already be running before this
    class is instantiated (each generator script handles that when
    ``--backend isaac`` is selected).
    """

    def __init__(self, num_envs: int = 4):
        import sys
        from pathlib import Path

        import torch  # noqa: F401  (available inside the Isaac interpreter)

        # make the starter env importable regardless of CWD
        starter_dir = Path(__file__).resolve().parents[2] / "isaac_unitree_starter"
        if str(starter_dir) not in sys.path:
            sys.path.insert(0, str(starter_dir))

        from go2_lab_env import Go2Env, Go2EnvCfg  # type: ignore[import-not-found]

        env_cfg = Go2EnvCfg()
        env_cfg.scene.num_envs = num_envs
        self._env = Go2Env(cfg=env_cfg)
        self.num_envs = num_envs
        self.obs_dim = env_cfg.observation_space
        self.action_dim = env_cfg.action_space

    def _info(self) -> dict[str, Any]:
        env = self._env
        robot_pos = (
            env._robot.data.root_pos_w[:, :3] - env.scene.env_origins[:, :3]
        ).cpu().numpy()
        return {
            "goal_direction": env._goal_direction().cpu().numpy(),
            "goal_pos": env._goal_pos.cpu().numpy(),
            "robot_pos": robot_pos,
        }

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        obs_dict, _ = self._env.reset(seed=seed)
        return obs_dict["policy"].cpu().numpy(), self._info()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        import torch

        action_t = torch.as_tensor(action, dtype=torch.float32, device=self._env.device)
        obs_dict, reward, terminated, truncated, _ = self._env.step(action_t)
        done = (terminated | truncated).cpu().numpy()
        info = self._info()
        info["reached_goal"] = np.zeros(self.num_envs, dtype=bool)  # TODO(isaac): expose from env
        return (
            obs_dict["policy"].cpu().numpy(),
            reward.cpu().numpy(),
            done,
            info,
        )

    def close(self) -> None:
        self._env.close()


def make_backend(name: str, num_envs: int, seed: int = 0) -> MockGo2Backend | IsaacGo2Backend:
    """Factory used by every generator script (``--backend`` flag)."""
    if name == "mock":
        return MockGo2Backend(num_envs=num_envs, seed=seed)
    if name == "isaac":
        return IsaacGo2Backend(num_envs=num_envs)
    raise ValueError(f"Unknown backend '{name}'. Choose: mock | isaac")