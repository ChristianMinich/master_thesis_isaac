"""VLA-inspired semantic navigation plugin.

Agent: a placeholder "semantic grounding" module that parses the scenario's
language instruction with a trivial keyword parser and steers toward the goal,
emitting an instruction-match confidence in ``Action.info`` (used later by
semantic-correctness metrics).

TODO(vla): replace the keyword parser + controller with a real VLA
operationalization, e.g. fine-tuned OpenVLA / Octo (Kim et al. 2025; Octo
Model Team 2024) consuming ``Observation.rgb`` + ``Observation.instruction``,
or a modular VLM-plus-controller. Only this plugin changes; core stays fixed.
"""

from __future__ import annotations

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


class VlaSemanticAgent(Agent):
    """Placeholder instruction-conditioned agent (composition over inheritance)."""

    def __init__(self, gain: float = 1.0) -> None:
        self._gain = gain
        self._instruction: str = ""

    def reset(self, scenario: Scenario, seed: int) -> None:
        self._instruction = scenario.instruction or ""

    def act(self, observation: Observation) -> Action:
        # TODO(vla): real semantic grounding of instruction + image -> action.
        # Placeholder: keyword check ("goal"/"target"/"zone") as a stand-in for
        # instruction-match confidence.
        text = (observation.instruction or self._instruction).lower()
        confidence = 1.0 if any(k in text for k in ("goal", "target", "zone")) else 0.5
        state = observation.robot_state or (0.0, 0.0, 0.0, 0.0)
        dx, dy = float(state[2]), float(state[3])
        vx = max(-1.0, min(1.0, self._gain * dx * confidence))
        vy = max(-1.0, min(1.0, self._gain * dy * confidence))
        return Action(command=(vx, vy, 0.0), info={"instruction_match": confidence})


class VlaAdaptationTrainer(Trainer):
    """Placeholder fine-tuning/adaptation pipeline over the episode dataset.

    TODO(vla): replace with real adaptation: action-space mapping, LoRA
    fine-tuning of an open VLA checkpoint on Episode observations/instructions,
    with train/val splits from the common EpisodeDataset.
    """

    def __init__(self, epochs: int = 1) -> None:
        self._epochs = epochs

    def train(
        self,
        agent: Agent,
        dataset: EpisodeDataset,
        simulator: SimulatorAdapter | None,
        output_dir: Path,
        seed: int,
    ) -> TrainingResult:
        start = time.perf_counter()
        n = len(dataset.episode_ids("train")) if "train" in dataset.splits() else 0
        output_dir.mkdir(parents=True, exist_ok=True)
        return TrainingResult(
            approach_name="vla_semantic",
            checkpoint_path=None,
            num_episodes_used=n * self._epochs,
            wall_clock_seconds=time.perf_counter() - start,
            hyperparameters={"epochs": self._epochs, "base_model": "TODO: OpenVLA/Octo"},
        )


@register_approach()
class VlaSemanticPlugin(ApproachPlugin):
    @property
    def name(self) -> str:
        return "vla_semantic"

    def build_agent(self, config: Mapping[str, Any]) -> Agent:
        return VlaSemanticAgent(gain=float(config.get("gain", 1.0)))

    def build_trainer(self, config: Mapping[str, Any]) -> Trainer:
        return VlaAdaptationTrainer(epochs=int(config.get("epochs", 1)))

    def describe(self) -> Mapping[str, Any]:
        return {
            "paradigm": "Vision-Language-Action semantic navigation",
            "reference": "Kim et al. 2025 (OpenVLA); Octo Model Team 2024",
            "modalities": ["rgb", "instruction", "robot_state"],
            "trainable": True,
        }