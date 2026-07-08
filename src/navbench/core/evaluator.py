"""Evaluator: applies a configurable set of Metrics to recorded episodes.

The evaluator is metric-agnostic — it never knows which approach produced the
episodes. Capability-based comparison (exposé §4.10) is achieved by grouping
results per approach and per scenario tag (nominal vs. perturbed).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Mapping, Sequence

from navbench.core.dataset import Episode
from navbench.core.metrics import Metric, MetricResult

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EpisodeRecord:
    """Associates a recorded episode with the approach that produced it."""

    approach_name: str
    episode: Episode


class Evaluator:
    """Computes all configured metrics per approach and per scenario tag group.

    Metrics are injected (dependency inversion); adding a metric only means
    adding it to the list — no evaluator changes.
    """

    def __init__(self, metrics: Sequence[Metric]) -> None:
        if not metrics:
            raise ValueError("Evaluator requires at least one metric")
        names = [m.name for m in metrics]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate metric names: {names}")
        self._metrics = tuple(metrics)

    @property
    def metric_names(self) -> tuple[str, ...]:
        return tuple(m.name for m in self._metrics)

    def evaluate(
        self, records: Sequence[EpisodeRecord]
    ) -> Mapping[str, Mapping[str, MetricResult]]:
        """Return {approach_name: {metric_name: MetricResult}}."""
        by_approach: dict[str, list[Episode]] = {}
        for record in records:
            by_approach.setdefault(record.approach_name, []).append(record.episode)

        results: dict[str, dict[str, MetricResult]] = {}
        for approach, episodes in by_approach.items():
            results[approach] = {}
            for metric in self._metrics:
                result = metric.compute(episodes)
                results[approach][metric.name] = result
                logger.info(
                    "metric.computed",
                    extra={
                        "approach": approach,
                        "metric": metric.name,
                        "value": round(result.value, 6),
                        "n_episodes": len(episodes),
                    },
                )
        return results

    def evaluate_grouped(
        self, records: Sequence[EpisodeRecord]
    ) -> Mapping[str, Mapping[str, Mapping[str, MetricResult]]]:
        """Return {group: {approach: {metric: MetricResult}}}.

        Groups episodes into "nominal" and "perturbed" so robustness deltas
        (exposé evaluation dimension "Robustness") can be reported directly.
        """
        groups: dict[str, list[EpisodeRecord]] = {"nominal": [], "perturbed": []}
        for record in records:
            key = "nominal" if record.episode.scenario.is_nominal else "perturbed"
            groups[key].append(record)
        return {
            group: self.evaluate(recs) for group, recs in groups.items() if recs
        }