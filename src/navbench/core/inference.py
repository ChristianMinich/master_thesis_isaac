"""InferenceRunner: executes one Agent on one Scenario through a SimulatorAdapter.

This is the single episode-execution loop of the benchmark. It is pure
orchestration: it owns no simulator, no agent, and no dataset — everything is
injected (dependency inversion). It also records episodes so that the same
runner can be used for dataset generation and for evaluation rollouts.
"""

from __future__ import annotations

import logging
import time
from typing import Sequence

from navbench.core.dataset import Episode, WritableEpisodeDataset
from navbench.core.agent import Agent
from navbench.core.scenario import Scenario
from navbench.core.simulator import SimulatorAdapter
from navbench.core.types import Action, Observation

logger = logging.getLogger(__name__)


class InferenceRunner:
    """Runs episodes: agent + scenario + simulator -> recorded Episode.

    Deterministic given (scenario, seed) and deterministic agent/adapter
    implementations.
    """

    def __init__(self, simulator: SimulatorAdapter) -> None:
        self._simulator = simulator

    def run_episode(
        self,
        agent: Agent,
        scenario: Scenario,
        seed: int,
        episode_id: str | None = None,
        record_to: WritableEpisodeDataset | None = None,
        split: str = "test",
    ) -> Episode:
        """Execute a single episode and return the recorded Episode."""
        eid = episode_id or f"{scenario.scenario_id}-seed{seed}"
        logger.info(
            "episode.start",
            extra={"episode_id": eid, "scenario_id": scenario.scenario_id, "seed": seed},
        )
        start = time.perf_counter()

        self._simulator.load_scenario(scenario)
        agent.reset(scenario, seed=seed)
        obs = self._simulator.reset(seed=seed)

        observations: list[Observation] = [obs]
        actions: list[Action] = []
        rewards: list[float] = []
        success = False
        info: dict[str, object] = {}

        for _ in range(scenario.max_steps):
            action = agent.act(obs)
            result = self._simulator.step(action)

            actions.append(action)
            rewards.append(result.reward)
            observations.append(result.observation)
            obs = result.observation
            info = dict(result.info)

            if result.terminated or result.truncated:
                success = bool(result.info.get("success", False))
                break

        elapsed = time.perf_counter() - start
        episode = Episode(
            episode_id=eid,
            scenario=scenario,
            observations=tuple(observations),
            actions=tuple(actions),
            rewards=tuple(rewards),
            success=success,
            seed=seed,
            metadata={
                "wall_clock_seconds": elapsed,
                "final_info": info,
                "is_nominal": scenario.is_nominal,
            },
        )
        if record_to is not None:
            record_to.add(episode, split=split)

        logger.info(
            "episode.end",
            extra={
                "episode_id": eid,
                "success": success,
                "length": episode.length,
                "wall_clock_seconds": round(elapsed, 4),
            },
        )
        return episode

    def run_many(
        self,
        agent: Agent,
        scenarios: Sequence[Scenario],
        seeds: Sequence[int],
        record_to: WritableEpisodeDataset | None = None,
        split: str = "test",
    ) -> list[Episode]:
        """Run every scenario with every seed (full cross product)."""
        episodes: list[Episode] = []
        for scenario in scenarios:
            for seed in seeds:
                episodes.append(
                    self.run_episode(
                        agent, scenario, seed=seed, record_to=record_to, split=split
                    )
                )
        return episodes