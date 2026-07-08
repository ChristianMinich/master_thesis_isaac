# navbench — Project Information & Settings

Simulation-based benchmark for comparing approaches to robot navigation
(SLAM, VLA, RL, World Models) using NVIDIA Isaac Sim / Isaac Lab.

This document is the single source of truth for the project's architecture,
conventions, settings, and extension workflow. See `README.md` for the quick
start; see `expose_christian_minich.pdf` for the scientific context (master's
thesis exposé, Universität Osnabrück, Design Science Research methodology).

---

## 1. Project Goal

Build a **reproducible, component-based research codebase** for a
simulation-based robot-navigation benchmark. The benchmark compares four
approach classes on one *common episode dataset* under shared, capability-based
evaluation dimensions:

| Approach       | Core capability                              | Operationalization (planned)                             |
|----------------|----------------------------------------------|----------------------------------------------------------|
| SLAM           | Localization, mapping, pose consistency      | ORB-SLAM3 / visual odometry / pose tracking (external)    |
| VLA            | Language/goal-conditioned action selection   | OpenVLA/Octo-style fine-tuned or VLA-inspired module      |
| RL             | Reward-based policy learning                 | PPO/SAC-style policy trained in Isaac Lab                 |
| World Model    | Predictive dynamics, deviation recognition   | Dreamer-style / JEPA-inspired predictive model            |
| Hybrid (opt.)  | Composition of the above                     | e.g. SLAM localization + RL control                       |

The thesis follows Design Science Research: the benchmark demonstrator is the
central artifact; the result is a differentiated evaluation matrix, not a
single ranking.

## 2. Architecture: Ports and Adapters (Plugin Architecture)

**Central requirement: algorithms must be easy to replace.** Training,
inference, evaluation, benchmarking, metrics, datasets, and simulation access
are connected through **stable interfaces (ports)**, never hardcoded
implementations.

```
                    ┌─────────────────────────────────────────┐
                    │            navbench.core                │
                    │  (interfaces + orchestration ONLY;      │
                    │   no Isaac, no torch, no numpy)         │
                    │                                         │
   configs (YAML) ─▶│  BenchmarkRunner ── InferenceRunner     │
                    │        │                 │              │
                    │  PluginRegistry     Evaluator/Metrics   │
                    │        │                 │              │
                    │  ApproachPlugin     EpisodeDataset      │
                    │   Agent/Trainer     Scenario            │
                    │        │            SimulatorAdapter ◀──┼── port
                    └────────┼─────────────────┼──────────────┘
                             │                 │
              ┌──────────────┴───────┐   ┌─────┴──────────────────┐
              │  navbench.plugins    │   │  navbench.sim          │
              │  (adapters: one      │   │  (adapters:            │
              │   folder per         │   │   MockSimulatorAdapter │
              │   approach)          │   │   IsaacSimAdapter      │
              │  slam_baseline/      │   │   IsaacLabAdapter)     │
              │  rl_policy/          │   └────────────────────────┘
              │  vla_semantic/       │
              │  world_model/        │
              │  hybrid/             │
              └──────────────────────┘
```

Rules enforced by design (and by tests):

1. `navbench.core` defines **interfaces only** plus generic orchestration.
   It never imports plugins, Isaac, torch, or numpy.
2. Concrete approaches are **separate plugin folders** under
   `src/navbench/plugins/<name>/` (or entirely external packages).
3. The **benchmark runner loads approaches from config** via the
   `PluginRegistry` (`plugin_module` = dotted import path; importing the module
   triggers `@register_approach` registration).
4. **Adding a new approach from a new paper = add one plugin folder + one YAML
   config file.** No core, dataset, evaluator, or simulation code changes.
5. Isaac Sim / Isaac Lab stay behind `SimulatorAdapter`. **Core logic runs
   without Isaac installed** (the `MockSimulatorAdapter` provides a kinematic
   point-robot world for tests and CI).

## 3. Core Abstractions (ports)

| Abstraction        | Module                        | Responsibility |
|--------------------|-------------------------------|----------------|
| `Pose`, `Observation`, `Action`, `StepResult` | `core/types.py` | Framework-free value types exchanged at all boundaries |
| `Scenario`, `Perturbation` | `core/scenario.py`   | Declarative task variant: scene, robot, spawn/goal, instruction, perturbations |
| `Episode`, `EpisodeDataset`, `WritableEpisodeDataset`, `InMemoryEpisodeDataset` | `core/dataset.py` | Common episode dataset (train/val/test splits; nominal vs. perturbed) |
| `Agent`            | `core/agent.py`               | The single decision-making interface every approach implements (`reset`, `act`) |
| `Trainer`, `TrainingResult`, `NoOpTrainer` | `core/trainer.py` | Training / adaptation / calibration pipelines with effort accounting |
| `InferenceRunner`  | `core/inference.py`           | The one episode-execution loop (agent + scenario + simulator → Episode) |
| `Metric`, `MetricResult` + concrete metrics | `core/metrics.py` | Success rate, episode length, return, path efficiency, trajectory error; TODO deviation-recognition metrics |
| `Evaluator`, `EpisodeRecord` | `core/evaluator.py` | Applies configured metrics per approach and per nominal/perturbed group |
| `BenchmarkRun`, `BenchmarkRunner` | `core/benchmark.py` | End-to-end orchestration from config: load plugins → train → roll out → evaluate → persist results |
| `ApproachPlugin`, `PluginRegistry`, `@register_approach` | `core/plugin.py` | The extension seam: factory bundle per approach + name→plugin registry |
| `SimulatorAdapter` | `core/simulator.py`           | The only simulator access port (`load_scenario`, `reset`, `step`, `close`) |
| `derive_seed`, `seed_everything` | `core/seeding.py` | Deterministic SHA-256 based sub-seed derivation from one master seed |
| `configure_logging`, `JsonLineFormatter` | `core/logging_utils.py` | Structured JSON-lines logging for parseable run records |
| config loader      | `core/config.py`              | YAML loading/validation into typed run configs |

Simulator adapters (`src/navbench/sim/`):

- `MockSimulatorAdapter` — deterministic kinematic point-robot world; used by
  tests, CI, and any machine without Isaac. Supports basic perturbation kinds.
- `IsaacSimAdapter` — skeleton with `TODO(isaac)` markers where the real
  Isaac Sim (omni.isaac / USD stage) APIs must be connected.
- `IsaacLabAdapter` — skeleton with `TODO(isaaclab)` markers for Isaac Lab
  environments (vectorized RL training workflows).

## 4. Repository Layout

```
master_thesis_isaac/
├── PROJECT.md                  # This file: architecture & settings reference
├── README.md                   # Quick start & usage
├── pyproject.toml              # Packaging, deps, pytest/mypy/ruff settings
├── expose_christian_minich.pdf # Thesis exposé (context only)
├── configs/
│   ├── benchmarks/             # Full benchmark run configs (what to compare)
│   │   └── smoke.yaml          # Mock-simulator smoke benchmark (no Isaac)
│   ├── scenarios/              # Scenario suites (nominal + perturbed variants)
│   │   └── corridor_v0.yaml
│   └── approaches/             # One config file per approach plugin
│       ├── slam_baseline.yaml
│       ├── rl_policy.yaml
│       ├── vla_semantic.yaml
│       ├── world_model.yaml
│       └── hybrid.yaml
├── src/navbench/
│   ├── core/                   # Ports + orchestration (NO Isaac imports)
│   ├── sim/                    # Simulator adapters (mock, Isaac Sim, Isaac Lab)
│   └── plugins/                # One folder per approach
│       ├── slam_baseline/      # Pose-monitoring / SLAM baseline
│       ├── rl_policy/          # RL navigation policy (PPO-style, TODO real impl)
│       ├── vla_semantic/       # VLA-inspired semantic navigation (TODO real model)
│       ├── world_model/        # Predictive dynamics module (TODO real model)
│       └── hybrid/             # Optional hybrid composition
├── scripts/
│   ├── run_benchmark.py        # Entry point: config → BenchmarkRunner → results
│   ├── generate_dataset.py     # Roll out scripted agent to build episode dataset
│   └── train_approach.py       # Train/adapt a single approach from config
├── tests/                      # pytest suite (runs WITHOUT Isaac)
│   ├── test_core_imports.py    # Core imports cleanly, no Isaac dependency
│   ├── test_seeding.py         # Determinism of seed derivation
│   ├── test_dataset.py         # EpisodeDataset behavior
│   ├── test_mock_simulator.py  # Mock adapter determinism & termination
│   ├── test_inference_runner.py# Episode loop invariants
│   ├── test_evaluator.py       # Metrics + grouping
│   ├── test_benchmark_runner.py# End-to-end config-driven run on mock sim
│   └── test_plugin_extensibility.py  # PROOF: dummy new approach added
│                               #  without modifying any core code
├── data/episodes/              # Generated episode data (gitignored payloads)
├── runs/                       # Benchmark run outputs (results.json, logs)
└── docs/                       # Additional documentation
```

## 5. Configuration Conventions (YAML, Hydra-style layering)

A benchmark run config composes scenario suites and approach configs by path:

```yaml
# configs/benchmarks/smoke.yaml
run_name: smoke
master_seed: 42
simulator:
  type: mock            # mock | isaac_sim | isaac_lab
scenarios: configs/scenarios/corridor_v0.yaml
approaches:
  - configs/approaches/slam_baseline.yaml
  - configs/approaches/rl_policy.yaml
episodes_per_scenario: 3
metrics: [success_rate, mean_episode_length, mean_return, path_efficiency, mean_trajectory_error]
output_dir: runs
```

Each approach config names its plugin and carries approach-specific params:

```yaml
# configs/approaches/rl_policy.yaml
name: rl_policy
plugin_module: navbench.plugins.rl_policy   # imported → registers plugin
params:
  hidden_sizes: [128, 128]
  ...
train:
  total_steps: 100000
  ...
```

`plugin_module` may point to **any importable module**, including packages
outside this repository — that is how a new paper's approach is integrated.

## 6. Settings Summary

| Setting            | Value |
|--------------------|-------|
| Package name       | `navbench` (src layout, `src/navbench/`) |
| Python             | ≥ 3.10, fully typed (`mypy --strict`) |
| Runtime dependency | `PyYAML` only (core); numpy/torch optional, seeded opportunistically |
| Dev tools          | `pytest`, `mypy`, `ruff` (line length 100) |
| Isaac Sim / Lab    | NOT pip deps; installed via NVIDIA tooling, accessed only through adapters |
| Logging            | Structured JSON lines (`navbench` logger), stderr + optional file per run |
| Seeding            | One `master_seed` per run; per-episode seeds via SHA-256 `derive_seed(master, scenario_id, approach, index)` |
| Robot (planned)    | Unitree Go2 quadruped (per exposé time plan) |
| Hardware (planned) | RTX 5090 for Isaac rendering/execution; A100 for heavy training |
| Action space (default) | Continuous `[v_x, v_y, yaw_rate]`, bounds ±1.0 |
| Episode invariant  | `len(actions) == len(observations) - 1 == len(rewards)` |
| Dataset splits     | `train` (nominal), `val`, `test` (incl. perturbed variants) |

## 7. Design Principles Applied

- **Dependency inversion** — core depends on abstractions; adapters/plugins
  depend on core, never the reverse.
- **SOLID** — single-responsibility modules; open for extension via plugins
  and metrics, closed for modification of core; Liskov-safe minimal `Agent`
  port; interface segregation (read-only vs. writable datasets); DI everywhere
  (runner receives simulator, evaluator receives metrics, trainer receives
  dataset/simulator).
- **Composition over inheritance** — approaches wrap internals (policy nets,
  SLAM handles, planners) as members; the hybrid plugin composes other agents
  instead of subclassing them.
- **Reproducibility** — deterministic seeds, declarative scenarios, recorded
  episodes, JSON-line logs, effort accounting in `TrainingResult`.
- **TODO placeholders** — real external systems (ORB-SLAM3, OpenVLA/Octo,
  Dreamer-style models, Isaac APIs) are marked with `TODO(...)` comments at
  the exact integration points; they are intentionally not reimplemented here.

## 8. How to Add a New Approach (from a new paper)

1. Create `src/navbench/plugins/<my_approach>/` with:
   - `agent.py` — implement `Agent` (`reset`, `act`), wrapping the paper's
     model/controller via composition.
   - `trainer.py` — implement `Trainer` (or reuse `NoOpTrainer`).
   - `plugin.py` — an `ApproachPlugin` subclass decorated with
     `@register_approach()`.
   - `__init__.py` — import `plugin` so registration runs on module import.
2. Add `configs/approaches/<my_approach>.yaml` with
   `plugin_module: navbench.plugins.<my_approach>` and its params.
3. Reference that YAML in a benchmark config. Done — **no core changes**.

`tests/test_plugin_extensibility.py` proves this workflow by defining a dummy
approach entirely inside the test module and running it through the unmodified
benchmark pipeline.

## 9. Status / Roadmap

- [x] Core ports and orchestration (runs without Isaac)
- [x] Mock simulator adapter, plugin skeletons, configs, scripts, tests
- [ ] Connect `IsaacSimAdapter` to a real USD scene + Unitree Go2 asset (TODO)
- [ ] Connect `IsaacLabAdapter` to an Isaac Lab task for RL training (TODO)
- [ ] On-disk episode dataset backend (HDF5/Zarr) for real sensor data (TODO)
- [ ] Deviation-recognition metrics (precision/recall/F1/AUROC, detection delay) (TODO)
- [ ] Real approach integrations: ORB-SLAM3, PPO (Isaac Lab), OpenVLA/Octo adaptation, Dreamer/JEPA-style WM (TODO)