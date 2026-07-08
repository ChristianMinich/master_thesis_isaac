"""BenchmarkRun / BenchmarkRunner: config-driven end-to-end orchestration.

Pipeline per approach:
  1. resolve plugin via PluginRegistry (importing ``plugin_module`` from config)
  2. build agent + trainer (factories from the plugin)
  3. train / adapt (optional, controlled by config)
  4. roll out all scenarios x seeds through the injected SimulatorAdapter
  5. evaluate all recorded episodes with the configured metrics
  6. persist a machine-readable results file

The runner never references a concrete approach or simulator class —
everything arrives via the registry, config, and constructor injection.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from navbench.core.config import BenchmarkConfig
from navbench.core.dataset import InMemoryEpisodeDataset, WritableEpisodeDataset
from navbench.core.evaluator import EpisodeRecord, Evaluator
from navbench.core.inference import InferenceRunner
from navbench.core.metrics import Metric, MetricResult, default_metrics
from navbench.core.plugin import PluginRegistry
from navbench.core.seeding import derive_seed, seed_everything
from navbench.core.simulator import SimulatorAdapter
from navbench.core.trainer import TrainingResult

logger = logging.getLogger(__name__)

# Factory signature for creating a simulator from its config section.
SimulatorFactory = Callable[[Mapping[str, Any]], SimulatorAdapter]


@dataclass(frozen=True)
class BenchmarkRun:
    """Immutable record of one completed benchmark run."""

    run_name: str
    master_seed: int
    approach_names: Sequence[str]
    training_results: Mapping[str, TrainingResult]
    results: Mapping[str, Mapping[str, Mapping[str, MetricResult]]]  # group -> approach -> metric
    records: Sequence[EpisodeRecord]
    wall_clock_seconds: float
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_summary(self) -> dict[str, Any]:
        """JSON-serializable summary (scalar metric values only)."""
        return {
            "run_name": self.run_name,
            "master_seed": self.master_seed,
            "approaches": list(self.approach_names),
            "wall_clock_seconds": round(self.wall_clock_seconds, 4),
            "training": {
                name: {
                    "num_steps": tr.num_steps,
                    "num_episodes_used": tr.num_episodes_used,
                    "wall_clock_seconds": round(tr.wall_clock_seconds, 4),
                    "checkpoint": str(tr.checkpoint_path) if tr.checkpoint_path else None,
                }
                for name, tr in self.training_results.items()
            },
            "results": {
                group: {
                    approach: {m: round(res.value, 6) for m, res in metrics.items()}
                    for approach, metrics in approaches.items()
                }
                for group, approaches in self.results.items()
            },
            "metadata": dict(self.metadata),
        }


class BenchmarkRunner:
    """Orchestrates a full benchmark run from a BenchmarkConfig.

    Dependencies:
    - ``simulator_factory``: builds the SimulatorAdapter from the config's
      ``simulator`` section (dependency inversion — core never imports
      concrete adapters; the entry-point script wires in the factory).
    - ``registry``: PluginRegistry used to resolve approaches by name.
    - ``metric_factories``: name -> Metric factory, so configs can select
      metrics by name; unknown names fail fast.
    """

    def __init__(
        self,
        simulator_factory: SimulatorFactory,
        registry: PluginRegistry | None = None,
        metric_factories: Mapping[str, Callable[[], Metric]] | None = None,
    ) -> None:
        self._simulator_factory = simulator_factory
        self._registry = registry if registry is not None else PluginRegistry.default()
        self._metric_factories = dict(metric_factories or {})

    def _build_metrics(self, names: Sequence[str]) -> list[Metric]:
        if not names:
            return default_metrics()
        available = {m.name: m for m in default_metrics()}
        metrics: list[Metric] = []
        for name in names:
            if name in self._metric_factories:
                metrics.append(self._metric_factories[name]())
            elif name in available:
                metrics.append(available[name])
            else:
                known = sorted(set(available) | set(self._metric_factories))
                raise KeyError(f"unknown metric {name!r}; known: {known}")
        return metrics

    def run(self, config: BenchmarkConfig, train: bool = False) -> BenchmarkRun:
        """Execute the benchmark described by ``config``.

        ``train=True`` runs each approach's Trainer before evaluation;
        ``train=False`` evaluates agents as-built (optionally from checkpoint).
        """
        start = time.perf_counter()
        seed_everything(config.master_seed)
        logger.info(
            "benchmark.start",
            extra={
                "run_name": config.run_name,
                "master_seed": config.master_seed,
                "n_scenarios": len(config.scenarios),
                "approaches": [a.name for a in config.approaches],
            },
        )

        evaluator = Evaluator(self._build_metrics(config.metrics))
        records: list[EpisodeRecord] = []
        training_results: dict[str, TrainingResult] = {}
        rollout_dataset: WritableEpisodeDataset = InMemoryEpisodeDataset()

        output_dir = config.output_dir / config.run_name
        output_dir.mkdir(parents=True, exist_ok=True)

        for approach_cfg in config.approaches:
            # 1. Resolve plugin from config (dynamic import triggers registration).
            if approach_cfg.name not in self._registry:
                self._registry.load_from_module(approach_cfg.plugin_module)
            plugin = self._registry.get(approach_cfg.name)

            # 2. Build agent and trainer via plugin factories.
            agent = plugin.build_agent(approach_cfg.params)
            if approach_cfg.checkpoint:
                agent.load_checkpoint(Path(approach_cfg.checkpoint))

            # 3. Optional training / adaptation.
            simulator = self._simulator_factory(config.simulator)
            try:
                if train:
                    trainer = plugin.build_trainer(approach_cfg.train)
                    train_seed = derive_seed(config.master_seed, approach_cfg.name, "train")
                    training_results[approach_cfg.name] = trainer.train(
                        agent=agent,
                        dataset=rollout_dataset,
                        simulator=simulator,
                        output_dir=output_dir / approach_cfg.name,
                        seed=train_seed,
                    )

                # 4. Roll out scenarios x episode repetitions.
                runner = InferenceRunner(simulator)
                for scenario in config.scenarios:
                    for i in range(config.episodes_per_scenario):
                        episode_seed = derive_seed(
                            config.master_seed, scenario.scenario_id, approach_cfg.name, i
                        )
                        episode = runner.run_episode(
                            agent,
                            scenario,
                            seed=episode_seed,
                            episode_id=f"{approach_cfg.name}-{scenario.scenario_id}-{i}",
                        )
                        records.append(
                            EpisodeRecord(approach_name=approach_cfg.name, episode=episode)
                        )
            finally:
                simulator.close()
                agent.close()

        # 5. Evaluate per nominal/perturbed group.
        results = evaluator.evaluate_grouped(records)

        elapsed = time.perf_counter() - start
        run = BenchmarkRun(
            run_name=config.run_name,
            master_seed=config.master_seed,
            approach_names=tuple(a.name for a in config.approaches),
            training_results=training_results,
            results=results,
            records=tuple(records),
            wall_clock_seconds=elapsed,
            metadata={"simulator": dict(config.simulator)},
        )

        # 6. Persist machine-readable summary.
        summary_path = output_dir / "results.json"
        summary_path.write_text(json.dumps(run.to_summary(), indent=2), encoding="utf-8")
        logger.info(
            "benchmark.end",
            extra={
                "run_name": config.run_name,
                "wall_clock_seconds": round(elapsed, 4),
                "results_file": str(summary_path),
            },
        )
        return run