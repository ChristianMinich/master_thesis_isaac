"""World Model / predictive dynamics plugin.

Agent: a placeholder predictive module that maintains a trivial linear
dynamics prediction of the next robot state, compares it against the actual
observation, and reports the prediction error as a deviation signal via
``Action.info["prediction_error"]`` / ``info["deviation_flag"]``. Control is a
simple goal-seeking law (composition).

TODO(wm): replace the linear predictor with a real learned world model
(Dreamer-style RSSM, Hafner et al. 2023, or a JEPA-inspired latent predictor)
trained on the common EpisodeDataset. Only this plugin changes; core stays
fixed.
"""

from __future__ import annotations

import math
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


class WorldModelAgent(Agent):
    """Placeholder predictive-dynamics agent with deviation flagging."""

    def __init__(
        self,
        gain: float = 1.0,
        step_size: float = 0.1,
        deviation_threshold: float = 0.05,
    ) -> None:
        self._gain = gain
        self._step_size = step_size
        self._threshold = deviation_threshold
        self._predicted_xy: tuple[float, float] | None = None

    def reset(self, scenario: Scenario, seed: int) -> None:
        self._predicted_xy = None

    def act(self, observation: Observation) -> Action:
        state = observation.robot_state or (0.0, 0.0, 0.0, 0.0)
        x, y = float(state[0]), float(state[1])
        dx, dy = float(state[2]), float(state[3])

        # Deviation recognition: compare last prediction vs. actual state.
        # TODO(wm): replace with latent-space prediction error of a learned
        # world model (e.g. RSSM posterior/prior KL or reconstruction error).
        prediction_error = 0.0
        if self._predicted_xy is not None:
            prediction_error = math.dist(self._predicted_xy, (x, y))
        deviation_flag = prediction_error > self._threshold

        vx = max(-1.0, min(1.0, self._gain * dx))
        vy = max(-1.0, min(1.0, self._gain * dy))

        # Predict next state with the (placeholder) internal dynamics model.
        self._predicted_xy = (x + vx * self._step_size, y + vy * self._step_size)

        return Action(
            command=(vx, vy, 0.0),
            info={
                "prediction_error": prediction_error,
                "deviation_flag": deviation_flag,
            },
        )


class WorldModelTrainer(Trainer):
    """Placeholder offline trainer over the common episode dataset.

    TODO(wm): replace with real world-model learning: sequence batches from
    EpisodeDataset, latent dynamics training (Dreamer/JEPA-style), and
    checkpointing of the learned model.
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
            approach_name="world_model",
            checkpoint_path=None,
            num_episodes_used=n * self._epochs,
            wall_clock_seconds=time.perf_counter() - start,
            hyperparameters={"epochs": self._epochs, "model": "TODO: Dreamer/JEPA-style"},
        )


@register_approach()
class WorldModelPlugin(ApproachPlugin):
    @property
    def name(self) -> str:
        return "world_model"

    def build_agent(self, config: Mapping[str, Any]) -> Agent:
        return WorldModelAgent(
            gain=float(config.get("gain", 1.0)),
            step_size=float(config.get("step_size", 0.1)),
            deviation_threshold=float(config.get("deviation_threshold", 0.05)),
        )

    def build_trainer(self, config: Mapping[str, Any]) -> Trainer:
        return WorldModelTrainer(epochs=int(config.get("epochs", 1)))

    def describe(self) -> Mapping[str, Any]:
        return {
            "paradigm": "World Model / predictive dynamics",
            "reference": "Hafner et al. 2023 (DreamerV3); LeCun 2022 (JEPA)",
            "modalities": ["robot_state"],
            "trainable": True,
        }