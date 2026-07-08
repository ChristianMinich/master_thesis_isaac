"""RL navigation policy plugin (PPO-style placeholder).

Agent: a tiny deterministic linear "policy" over the goal-relative state,
seeded reproducibly. Trainer: a placeholder interactive loop that counts
simulation steps and writes a checkpoint stub, so effort accounting and the
pipeline shape are real.

TODO(rl): replace with a real PPO/SAC implementation trained in Isaac Lab
(vectorized envs) or via a library (e.g. rsl_rl / skrl / stable-baselines3).
Only this plugin changes — the Trainer port, benchmark runner, and adapters
stay untouched.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any, Mapping

from navbench.core.agent import Agent
from navbench.core.dataset import EpisodeDataset
from navbench.core.plugin import ApproachPlugin, register_approach
from navbench.core.scenario import Scenario
from navbench.core.simulator import SimulatorAdapter
from navbench.core.trainer import Trainer, TrainingResult
from navbench.core.types import Action, Observation


class RlPolicyAgent(Agent):
    """Placeholder policy: proportional goal-seeking with seeded exploration noise.

    TODO(rl): replace with a neural policy network (torch) loaded from a
    checkpoint; keep the Agent interface identical.
    """

    def __init__(self, gain: float = 1.0, exploration_noise: float = 0.0) -> None:
        self._gain = gain
        self._noise = exploration_noise
        self._rng = random.Random(0)

    def reset(self, scenario: Scenario, seed: int) -> None:
        self._rng = random.Random(seed)

    def act(self, observation: Observation) -> Action:
        state = observation.robot_state or (0.0, 0.0, 0.0, 0.0)
        dx, dy = float(state[2]), float(state[3])  # goal-relative deltas
        vx = max(-1.0, min(1.0, self._gain * dx + self._rng.gauss(0.0, self._noise)))
        vy = max(-1.0, min(1.0, self._gain * dy + self._rng.gauss(0.0, self._noise)))
        return Action(command=(vx, vy, 0.0))

    def load_checkpoint(self, path: Path) -> None:
        # TODO(rl): load real network weights.
        data = json.loads(path.read_text(encoding="utf-8"))
        self._gain = float(data.get("gain", self._gain))


class RlPolicyTrainer(Trainer):
    """Placeholder interactive trainer: rolls the policy in the simulator.

    TODO(rl): replace the loop body with PPO updates (advantage estimation,
    minibatch epochs). The signature and TrainingResult stay identical.
    """

    def __init__(self, total_steps: int = 1000, scenario: Scenario | None = None) -> None:
        self._total_steps = total_steps
        self._scenario = scenario

    def train(
        self,
        agent: Agent,
        dataset: EpisodeDataset,
        simulator: SimulatorAdapter | None,
        output_dir: Path,
        seed: int,
    ) -> TrainingResult:
        start = time.perf_counter()
        steps_done = 0
        if simulator is not None and self._scenario is not None:
            simulator.load_scenario(self._scenario)
            agent.reset(self._scenario, seed=seed)
            obs = simulator.reset(seed=seed)
            while steps_done < self._total_steps:
                result = simulator.step(agent.act(obs))
                obs = result.observation
                steps_done += 1
                if result.terminated or result.truncated:
                    obs = simulator.reset(seed=seed + steps_done)
        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = output_dir / "policy.json"
        checkpoint.write_text(json.dumps({"gain": 1.0}), encoding="utf-8")
        return TrainingResult(
            approach_name="rl_policy",
            checkpoint_path=checkpoint,
            num_steps=steps_done,
            wall_clock_seconds=time.perf_counter() - start,
            hyperparameters={"total_steps": self._total_steps, "algo": "TODO: PPO"},
        )


@register_approach()
class RlPolicyPlugin(ApproachPlugin):
    @property
    def name(self) -> str:
        return "rl_policy"

    def build_agent(self, config: Mapping[str, Any]) -> Agent:
        return RlPolicyAgent(
            gain=float(config.get("gain", 1.0)),
            exploration_noise=float(config.get("exploration_noise", 0.0)),
        )

    def build_trainer(self, config: Mapping[str, Any]) -> Trainer:
        return RlPolicyTrainer(total_steps=int(config.get("total_steps", 1000)))

    def describe(self) -> Mapping[str, Any]:
        return {
            "paradigm": "Reinforcement Learning navigation policy",
            "reference": "Schulman et al. 2017 (PPO); Mittal et al. 2025 (Isaac Lab)",
            "modalities": ["robot_state"],
            "trainable": True,
        }