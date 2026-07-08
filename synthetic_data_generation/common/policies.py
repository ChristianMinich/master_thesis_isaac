"""Action sources (policies) used to generate synthetic Go2 data.

All policies operate on numpy arrays with the batched shapes used by the
backends: observations ``(N, obs_dim)`` -> actions ``(N, act_dim)`` in
``[-1, 1]``. They are intentionally simple; each generator script documents
where a trained policy should be plugged in later (TODO markers).
"""

from __future__ import annotations

from typing import Protocol

import numpy as np


class Policy(Protocol):
    def __call__(self, obs: np.ndarray, info: dict) -> np.ndarray: ...


class RandomPolicy:
    """Uniform random actions in [-1, 1]. Good for broad state coverage
    (world-model data) but produces poor navigation behavior."""

    def __init__(self, action_dim: int, seed: int = 0):
        self.action_dim = action_dim
        self._rng = np.random.default_rng(seed)

    def __call__(self, obs: np.ndarray, info: dict) -> np.ndarray:
        return self._rng.uniform(-1.0, 1.0, size=(obs.shape[0], self.action_dim))


class ZeroPolicy:
    """Zero actions plus optional Gaussian exploration noise."""

    def __init__(self, action_dim: int, noise_std: float = 0.0, seed: int = 0):
        self.action_dim = action_dim
        self.noise_std = noise_std
        self._rng = np.random.default_rng(seed)

    def __call__(self, obs: np.ndarray, info: dict) -> np.ndarray:
        noise = self._rng.normal(0.0, self.noise_std, size=(obs.shape[0], self.action_dim))
        return np.clip(noise, -1.0, 1.0)


class ScriptedGoalPolicy:
    """Scripted expert that walks toward the goal.

    Uses ``info["goal_direction"]`` (unit vector, shape ``(N, 2)``) provided
    by the backends. On the mock backend this steers the kinematic robot
    directly; on the Isaac backend it produces a crude "lean toward goal"
    joint bias — replace it with a trained locomotion policy for realistic
    gait data.

    TODO(policy): load a trained RL checkpoint (see train_rl_go2.py) and use
    it here for high-quality expert demonstrations.
    """

    def __init__(self, action_dim: int, gain: float = 1.0, noise_std: float = 0.05, seed: int = 0):
        self.action_dim = action_dim
        self.gain = gain
        self.noise_std = noise_std
        self._rng = np.random.default_rng(seed)

    def __call__(self, obs: np.ndarray, info: dict) -> np.ndarray:
        goal_dir = info["goal_direction"]  # (N, 2)
        num_envs = obs.shape[0]
        action = np.zeros((num_envs, self.action_dim))
        # spread the 2D command over the action vector; the mock backend
        # interprets dims 0/1 as vx/vy command, remaining dims are ignored.
        action[:, 0] = self.gain * goal_dir[:, 0]
        action[:, 1] = self.gain * goal_dir[:, 1]
        action += self._rng.normal(0.0, self.noise_std, size=action.shape)
        return np.clip(action, -1.0, 1.0)


def make_policy(name: str, action_dim: int, seed: int = 0, noise_std: float = 0.1) -> Policy:
    """Factory used by every generator script (``--policy`` flag)."""
    if name == "random":
        return RandomPolicy(action_dim, seed=seed)
    if name == "zero":
        return ZeroPolicy(action_dim, noise_std=noise_std, seed=seed)
    if name == "scripted_goal":
        return ScriptedGoalPolicy(action_dim, noise_std=noise_std, seed=seed)
    raise ValueError(f"Unknown policy '{name}'. Choose: random | zero | scripted_goal")