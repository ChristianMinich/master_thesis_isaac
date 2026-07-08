"""YAML config loading for benchmark runs (Hydra-style composition by path).

A benchmark config references scenario-suite and approach configs by relative
file path, so runs are fully declarative and versionable. Only PyYAML is
required; swapping in Hydra/OmegaConf later would only touch this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from navbench.core.scenario import Perturbation, Scenario


@dataclass(frozen=True)
class ApproachConfig:
    """Parsed approach config (one YAML file per approach)."""

    name: str
    plugin_module: str
    params: Mapping[str, Any] = field(default_factory=dict)
    train: Mapping[str, Any] = field(default_factory=dict)
    checkpoint: str | None = None


@dataclass(frozen=True)
class BenchmarkConfig:
    """Parsed top-level benchmark run config."""

    run_name: str
    master_seed: int
    simulator: Mapping[str, Any]
    scenarios: Sequence[Scenario]
    approaches: Sequence[ApproachConfig]
    episodes_per_scenario: int
    metrics: Sequence[str]
    output_dir: Path


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"expected a mapping at top level of {path}")
    return data


def load_scenarios(path: Path) -> list[Scenario]:
    """Load a scenario suite YAML into Scenario objects."""
    data = _load_yaml(path)
    scenarios: list[Scenario] = []
    for entry in data.get("scenarios", []):
        perturbations = tuple(
            Perturbation(
                kind=p["kind"],
                onset_step=int(p.get("onset_step", 0)),
                params=p.get("params", {}),
            )
            for p in entry.get("perturbations", [])
        )
        scenarios.append(
            Scenario(
                scenario_id=entry["scenario_id"],
                scene=entry["scene"],
                robot=entry.get("robot", "unitree_go2"),
                spawn_pose=tuple(entry.get("spawn_pose", (0.0, 0.0, 0.0))),  # type: ignore[arg-type]
                goal_position=tuple(entry.get("goal_position", (5.0, 5.0, 0.0))),  # type: ignore[arg-type]
                goal_radius=float(entry.get("goal_radius", 0.5)),
                instruction=entry.get("instruction"),
                max_steps=int(entry.get("max_steps", 500)),
                perturbations=perturbations,
                tags=tuple(entry.get("tags", ())),
            )
        )
    if not scenarios:
        raise ValueError(f"no scenarios defined in {path}")
    return scenarios


def load_approach_config(path: Path) -> ApproachConfig:
    """Load one approach YAML."""
    data = _load_yaml(path)
    return ApproachConfig(
        name=data["name"],
        plugin_module=data["plugin_module"],
        params=data.get("params", {}),
        train=data.get("train", {}),
        checkpoint=data.get("checkpoint"),
    )


def load_benchmark_config(path: Path) -> BenchmarkConfig:
    """Load a benchmark run YAML, resolving referenced config files.

    Relative paths inside the config are resolved against the current working
    directory first, then against the config file's directory.
    """
    data = _load_yaml(path)

    def resolve(ref: str) -> Path:
        candidate = Path(ref)
        if candidate.exists():
            return candidate
        fallback = path.parent / ref
        if fallback.exists():
            return fallback
        raise FileNotFoundError(f"referenced config not found: {ref} (from {path})")

    scenarios = load_scenarios(resolve(data["scenarios"]))
    approaches = [load_approach_config(resolve(ref)) for ref in data["approaches"]]

    return BenchmarkConfig(
        run_name=data["run_name"],
        master_seed=int(data.get("master_seed", 0)),
        simulator=data.get("simulator", {"type": "mock"}),
        scenarios=scenarios,
        approaches=approaches,
        episodes_per_scenario=int(data.get("episodes_per_scenario", 1)),
        metrics=tuple(data.get("metrics", ())),
        output_dir=Path(data.get("output_dir", "runs")),
    )