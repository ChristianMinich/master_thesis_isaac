"""Metric port and core capability metrics.

Metrics consume recorded Episodes and produce named scalar results. They are
registered per benchmark run via config, so adding a new metric never requires
touching the evaluator or runner.

The exposé's evaluation dimensions map to metric families:
- navigation success (success rate, completion time, path efficiency)
- spatial accuracy (trajectory error vs. ground truth)
- deviation recognition (precision/recall/F1, detection delay) — TODO once
  perturbation labels are produced by real adapters.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Mapping, Sequence

from navbench.core.dataset import Episode


@dataclass(frozen=True)
class MetricResult:
    """A named scalar with optional per-episode breakdown."""

    name: str
    value: float
    per_episode: Mapping[str, float] = field(default_factory=dict)


class Metric(ABC):
    """Port: computes one evaluation quantity over a set of episodes."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique metric name used in reports and configs."""

    @abstractmethod
    def compute(self, episodes: Sequence[Episode]) -> MetricResult:
        """Aggregate the metric over episodes."""


class SuccessRate(Metric):
    """Fraction of episodes that reached the goal."""

    @property
    def name(self) -> str:
        return "success_rate"

    def compute(self, episodes: Sequence[Episode]) -> MetricResult:
        per = {e.episode_id: float(e.success) for e in episodes}
        value = sum(per.values()) / len(per) if per else 0.0
        return MetricResult(self.name, value, per)


class MeanEpisodeLength(Metric):
    """Average number of control steps per episode (proxy for completion time)."""

    @property
    def name(self) -> str:
        return "mean_episode_length"

    def compute(self, episodes: Sequence[Episode]) -> MetricResult:
        per = {e.episode_id: float(e.length) for e in episodes}
        value = sum(per.values()) / len(per) if per else 0.0
        return MetricResult(self.name, value, per)


class MeanReturn(Metric):
    """Average undiscounted return per episode."""

    @property
    def name(self) -> str:
        return "mean_return"

    def compute(self, episodes: Sequence[Episode]) -> MetricResult:
        per = {e.episode_id: float(sum(e.rewards)) for e in episodes}
        value = sum(per.values()) / len(per) if per else 0.0
        return MetricResult(self.name, value, per)


class PathEfficiency(Metric):
    """Straight-line spawn->goal distance divided by traveled path length.

    1.0 is a perfectly straight path; lower values indicate detours.
    Requires ground-truth poses in observations (simulation privilege).
    """

    @property
    def name(self) -> str:
        return "path_efficiency"

    def compute(self, episodes: Sequence[Episode]) -> MetricResult:
        per: dict[str, float] = {}
        for e in episodes:
            poses = [o.robot_pose for o in e.observations if o.robot_pose is not None]
            if len(poses) < 2:
                continue
            traveled = 0.0
            for a, b in zip(poses, poses[1:]):
                traveled += math.dist(a.position, b.position)
            straight = math.dist(poses[0].position, e.scenario.goal_position)
            per[e.episode_id] = straight / traveled if traveled > 1e-9 else 0.0
        value = sum(per.values()) / len(per) if per else 0.0
        return MetricResult(self.name, value, per)


class TrajectoryError(Metric):
    """Mean absolute trajectory error between an agent's pose estimate and ground truth.

    Consumes ``Action.info["estimated_pose"]`` (produced e.g. by SLAM-style
    agents) and ground-truth ``Observation.robot_pose``. Episodes without pose
    estimates are skipped, so the metric composes cleanly with non-spatial
    approaches.
    """

    @property
    def name(self) -> str:
        return "mean_trajectory_error"

    def compute(self, episodes: Sequence[Episode]) -> MetricResult:
        per: dict[str, float] = {}
        for e in episodes:
            errors: list[float] = []
            for obs, act in zip(e.observations[1:], e.actions):
                est = act.info.get("estimated_pose")
                if est is None or obs.robot_pose is None:
                    continue
                errors.append(math.dist(est.position, obs.robot_pose.position))
            if errors:
                per[e.episode_id] = sum(errors) / len(errors)
        value = sum(per.values()) / len(per) if per else 0.0
        return MetricResult(self.name, value, per)


# TODO(metrics): add deviation-recognition metrics (precision, recall, F1,
# AUROC, detection delay after perturbation onset) once adapters emit
# perturbation onset labels in Episode.metadata and agents emit deviation
# scores in Action.info["deviation_score"]. These belong here as new Metric
# implementations — no evaluator/runner changes needed.


def default_metrics() -> list[Metric]:
    """Metric set used when a config does not specify metrics explicitly."""
    return [SuccessRate(), MeanEpisodeLength(), MeanReturn(), PathEfficiency(), TrajectoryError()]