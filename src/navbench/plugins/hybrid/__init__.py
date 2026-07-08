"""Hybrid approach plugin (optional): composition of other approaches.

Demonstrates composition over inheritance at the approach level: the hybrid
agent OWNS a spatial pose-monitoring component and a learned control policy
component and mediates between them — it does not subclass either.

Placeholder composition: SLAM-style dead-reckoning pose estimate (spatial
grounding) + RL-style goal-seeking control. The pose estimate and control are
combined so that the controller acts on the *estimated* state while the pose
monitor's drift is exposed for evaluation.

TODO(hybrid): swap the components for the real ones (e.g. ORB-SLAM3
localization feeding a trained RL policy, or a world-model deviation detector
gating a VLA policy). Only this plugin changes; core stays fixed.
"""

from __future__ import annotations

from typing import Any, Mapping

from navbench.core.agent import Agent
from navbench.core.plugin import ApproachPlugin, register_approach
from navbench.core.scenario import Scenario
from navbench.core.trainer import NoOpTrainer, Trainer
from navbench.core.types import Action, Observation
from navbench.plugins.rl_policy import RlPolicyAgent
from navbench.plugins.slam_baseline import SlamBaselineAgent


class HybridAgent(Agent):
    """Composes a pose-monitoring component and a control-policy component."""

    def __init__(self, localizer: Agent, controller: Agent) -> None:
        self._localizer = localizer
        self._controller = controller

    def reset(self, scenario: Scenario, seed: int) -> None:
        self._localizer.reset(scenario, seed)
        self._controller.reset(scenario, seed)

    def act(self, observation: Observation) -> Action:
        # Spatial component produces the pose estimate (and its drift signal).
        loc_action = self._localizer.act(observation)
        # Learned component produces the control command.
        ctrl_action = self._controller.act(observation)
        # Mediate: use controller command, carry localization info alongside.
        info = dict(ctrl_action.info)
        info.update(loc_action.info)  # e.g. "estimated_pose"
        return Action(command=ctrl_action.command, info=info)

    def close(self) -> None:
        self._localizer.close()
        self._controller.close()


@register_approach()
class HybridPlugin(ApproachPlugin):
    @property
    def name(self) -> str:
        return "hybrid"

    def build_agent(self, config: Mapping[str, Any]) -> Agent:
        localizer = SlamBaselineAgent(
            gain=float(config.get("localizer_gain", 1.0)),
            step_size=float(config.get("step_size", 0.1)),
        )
        controller = RlPolicyAgent(
            gain=float(config.get("controller_gain", 1.0)),
            exploration_noise=float(config.get("exploration_noise", 0.0)),
        )
        return HybridAgent(localizer=localizer, controller=controller)

    def build_trainer(self, config: Mapping[str, Any]) -> Trainer:
        # TODO(hybrid): train sub-components via their own trainers if needed.
        return NoOpTrainer(approach_name=self.name)

    def describe(self) -> Mapping[str, Any]:
        return {
            "paradigm": "Hybrid (spatial localization + learned control)",
            "composition": ["slam_baseline", "rl_policy"],
            "modalities": ["robot_state", "goal_position"],
            "trainable": False,
        }