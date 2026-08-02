# navbench — Plugin-Based Robot-Navigation Benchmark (Isaac Sim / Isaac Lab)

Reproducible, component-based research codebase for a simulation-based
robot-navigation benchmark, built for a master's thesis comparing different
navigation paradigms (SLAM, RL, VLA, World Models) under nominal and
perturbed conditions.

**Key property: algorithms are replaceable plugins.** Core benchmark logic
defines stable interfaces (ports) only; concrete approaches, simulators, and
metrics are injected via config. Adding a new approach from a new paper means
adding one plugin folder + one YAML file — never editing core code.

## Architecture (ports & adapters)

```
┌───────────────────────────────────────────────────────────────────┐
│  scripts/run_benchmark.py  (composition root: wires everything)   │
└───────────────┬───────────────────────────────────────────────────┘
                │ config (YAML)
┌───────────────▼───────────────────────────────────────────────────┐
│  navbench.core   — interfaces + orchestration ONLY                │
│                                                                    │
│  BenchmarkRunner ─ uses ─▶ PluginRegistry ─ resolves ─▶ ApproachPlugin (port) │
│        │                                                           │
│        ├─▶ InferenceRunner ─▶ Agent (port) + SimulatorAdapter (port)│
│        ├─▶ Trainer (port)   + EpisodeDataset (port)                │
│        └─▶ Evaluator ─▶ Metric (port) ─▶ BenchmarkRun (result)     │
└───────────────┬───────────────────────────────┬───────────────────┘
                │ implemented by                │ implemented by
┌───────────────▼──────────────┐  ┌─────────────▼─────────────────┐
│  navbench.plugins.*          │  │  navbench.sim.*               │
│  slam_baseline / rl_policy / │  │  mock (runs anywhere)         │
│  vla_semantic / world_model /│  │  isaac_sim  (TODO wiring)     │
│  hybrid                      │  │  isaac_lab  (TODO wiring)     │
└──────────────────────────────┘  └───────────────────────────────┘
```

- **Dependency inversion:** `navbench.core` never imports a concrete
  approach or simulator. The entry-point script injects `make_simulator`
  (factory) and the `PluginRegistry` resolves approaches from config
  (`plugin_module` dotted path).
- **Isaac stays behind adapters:** `IsaacSimAdapter` / `IsaacLabAdapter`
  import Isaac lazily; core + mock run on any machine without Isaac
  installed (verified by tests).
- **Composition over inheritance:** agents wrap their internals (estimator,
  policy, planner) as members; no deep class hierarchies.

## Core abstractions

| Abstraction | Module | Role |
|---|---|---|
| `Observation`, `Action`, `Pose`, `StepResult` | `core.types` | framework-free data exchanged at every boundary |
| `Scenario`, `Perturbation` | `core.scenario` | declarative task variants (data, not behavior) |
| `EpisodeDataset`, `Episode` | `core.dataset` | common episode storage port (in-memory impl. included) |
| `Agent` | `core.agent` | the ONE interface every approach implements |
| `Trainer`, `TrainingResult`, `NoOpTrainer` | `core.trainer` | training / adaptation / calibration port |
| `InferenceRunner` | `core.inference` | single episode-execution loop |
| `Evaluator`, `EpisodeRecord` | `core.evaluator` | metric-agnostic evaluation, grouped by nominal/perturbed |
| `Metric`, `MetricResult` | `core.metrics` | success rate, episode length, return, path efficiency, trajectory error |
| `ApproachPlugin`, `PluginRegistry`, `@register_approach` | `core.plugin` | the extension seam |
| `BenchmarkRun`, `BenchmarkRunner` | `core.benchmark` | config-driven end-to-end orchestration |
| `SimulatorAdapter` | `core.simulator` | simulation access port |
| `MockSimulatorAdapter` / `IsaacSimAdapter` / `IsaacLabAdapter` | `sim.*` | interchangeable backends |

Reproducibility: `core.seeding` provides `seed_everything(master_seed)` and
deterministic `derive_seed(master_seed, *components)` per
(approach, scenario, episode). Logging: `core.logging_utils` emits structured
JSON events (`benchmark.start`, `episode.end`, `metric.computed`, ...).

## Quick start (no Isaac required)

```bash
python3 -m venv .venv
.venv/bin/pip install pytest pyyaml

# run the tests
.venv/bin/python -m pytest tests -q

# fast smoke benchmark (1 approach)
.venv/bin/python scripts/run_benchmark.py configs/benchmarks/mock_smoke.yaml

# full comparison of all 5 built-in approach plugins on the mock simulator
.venv/bin/python scripts/run_benchmark.py configs/benchmarks/mock_full_comparison.yaml
```

Results land in `runs/<run_name>/results.json`, grouped by scenario tag
(`nominal` vs `perturbed`) and approach.

Switching to Isaac later only changes the config's `simulator` section
(`type: isaac_sim | isaac_lab`) once the adapter TODOs are wired — no
approach or core code changes.

## Repository layout

```
configs/
  approaches/    one YAML per approach (name, plugin_module, params, train, checkpoint)
  benchmarks/    run definitions (simulator, scenarios, approaches, seeds, metrics)
  scenarios/     scenario suites incl. perturbations
src/navbench/
  core/          interfaces + orchestration (NO Isaac, NO concrete approaches)
  plugins/       built-in approach plugins (one folder each)
  sim/           simulator adapters (mock, isaac_sim, isaac_lab)
scripts/
  run_benchmark.py   composition root / CLI entry point
tests/           pipeline, determinism, and extensibility tests
data/episodes/   shared episode dataset storage
runs/            benchmark outputs (results.json per run)
```

## Built-in approach plugins

All are typed skeletons with clear `TODO(...)` markers where real systems
plug in — none of ORB-SLAM3 / OpenVLA / Dreamer / Isaac internals is
reimplemented here:

- `slam_baseline` — goal-seeking controller + dead-reckoning pose monitor;
  TODO: bridge to a real SLAM system (e.g. ORB-SLAM3).
- `rl_policy` — linear reactive policy; TODO: PPO/SAC training via Isaac Lab.
- `vla_semantic` — instruction-conditioned controller; TODO: real VLA
  (e.g. OpenVLA) inference/adaptation.
- `world_model` — learned-dynamics predictor + planner stub; TODO:
  Dreamer-style latent dynamics model.
- `hybrid` — composes a localizer with a learned controller (optional).

## Adding a new approach from a new paper

1. **Create one plugin folder** `src/navbench/plugins/<my_approach>/__init__.py`
   (or any external package):

   ```python
   from navbench.core.agent import Agent
   from navbench.core.plugin import ApproachPlugin, register_approach
   from navbench.core.trainer import NoOpTrainer

   class MyAgent(Agent):
       def reset(self, scenario, seed): ...
       def act(self, observation): ...   # -> Action

   @register_approach()
   class MyPlugin(ApproachPlugin):
       @property
       def name(self): return "my_approach"
       def build_agent(self, config): return MyAgent(**config)
       def build_trainer(self, config): return NoOpTrainer(self.name)
   ```

2. **Create one config file** `configs/approaches/my_approach.yaml`:

   ```yaml
   name: my_approach
   plugin_module: navbench.plugins.my_approach
   params: { ... }
   ```

3. **Reference it** in a benchmark YAML under `approaches:` and run.

That's it — the runner imports `plugin_module`, the decorator registers the
plugin, and it is trained/evaluated/compared identically to all others.
`tests/test_extensibility.py` proves this end-to-end with a dummy plugin
that lives entirely outside `navbench` (in `tests/dummy_approach.py`) and is
loaded purely via config, with zero core modifications.

## Testing

```bash
.venv/bin/python -m pytest tests -q
```

- `test_pipeline.py` — core imports without Isaac, mock-simulator
  determinism, full 5-approach comparison run from the shipped config.
- `test_extensibility.py` — dummy new-paper plugin loads via registry,
  runs a full benchmark without core changes, and results are
  seed-reproducible.