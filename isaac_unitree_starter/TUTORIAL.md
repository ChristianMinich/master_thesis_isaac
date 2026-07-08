# Isaac Sim + Isaac Lab Starter Pipeline for the Master's Thesis
## Simulation-Based Benchmarking of Approaches to Robot Navigation (Unitree Go2)

This folder contains the **first runnable starter pipeline** for the thesis
*"Design, Implementation, and Evaluation of a Simulation-Based Benchmark for
Comparing Approaches to Robot Navigation"* (see `expose_christian_minich.pdf`).

It is a beginner-friendly, step-by-step tutorial plus a minimal set of
scripts. It is **not** the final full system — it corresponds to phases 1–4
of the exposé's time plan (technical familiarization → simulation
environment construction → first runnable end-to-end skeleton).

---

## Table of Contents

1. [What the thesis project is trying to build](#1-what-the-thesis-project-is-trying-to-build)
2. [How Isaac Sim is used in this project](#2-how-isaac-sim-is-used-in-this-project)
3. [How Isaac Lab is used in this project](#3-how-isaac-lab-is-used-in-this-project)
4. [What should be implemented first](#4-what-should-be-implemented-first)
5. [Creating the Unitree Go2 environment](#5-creating-the-unitree-go2-environment)
6. [Generating synthetic navigation data](#6-generating-synthetic-navigation-data)
7. [How RL training fits into the project](#7-how-rl-training-fits-into-the-project)
8. [How VLA-style data collection fits in](#8-how-vla-style-data-collection-fits-in)
9. [How world model data collection fits in](#9-how-world-model-data-collection-fits-in)
10. [How SLAM / pose-based evaluation is added later](#10-how-slam--pose-based-evaluation-is-added-later)
11. [How the navigation approaches will be benchmarked](#11-how-the-navigation-approaches-will-be-benchmarked)
12. [What the current starter pipeline contains](#12-what-the-current-starter-pipeline-contains)
13. [What still needs to be implemented later](#13-what-still-needs-to-be-implemented-later)
14. [Quick-start: running the pipeline](#14-quick-start-running-the-pipeline)

---

## 1. What the thesis project is trying to build

The thesis follows a **Design Science Research** methodology. Its central
artifact is a **simulation-based benchmark demonstrator** that compares four
approach classes to robot navigation in one shared, reproducible setting:

| Approach | Core capability | What "training" means here |
|---|---|---|
| **SLAM** | Localization, mapping, pose consistency | Configuration/calibration, no neural training |
| **VLA** | Vision-language-conditioned action selection | Fine-tuning / adapting an open VLA model, or a smaller VLA-inspired module |
| **RL** | Reward-based policy learning | Training a PPO-style policy in simulation |
| **World Model (WM)** | Predictive dynamics, deviation recognition | Training an encoder + transition predictor on logged sequences |

Key design decisions from the exposé that this starter pipeline reflects:

- **Robot**: Unitree Go2 quadruped ("robot dog").
- **Simulator**: NVIDIA Isaac Sim (scene, sensors, rendering) + Isaac Lab
  (learning workflows), with an RTX 5090 as the local rendering/training GPU
  and an A100 for heavy neural training. MuJoCo/MJX is an optional
  complementary environment (not part of this starter).
- **Data**: a project-specific **common episode dataset** generated in
  simulation. All approaches consume the *same* underlying navigation
  episodes (method-specific views of it), which is central to comparison
  fairness.
- **Evaluation**: capability-based, not a single ranking. Metrics include
  trajectory error (SLAM), navigation success and robustness (RL/VLA),
  semantic correctness (VLA), prediction error and deviation-detection delay
  (WM), plus training/compute/implementation effort for all learned parts.
- **Perturbations**: obstacle changes, sensor noise, layout variation,
  localization drift, instruction mismatch etc. are injected in controlled
  ways to test robustness and deviation recognition.

The exact navigation use case is deliberately left flexible in the exposé.
This starter therefore uses the simplest meaningful task: **goal-directed
locomotion of the Go2 on flat ground with simple obstacles**, which can be
grown into corridor/warehouse navigation later.

---

## 2. How Isaac Sim is used in this project

**Isaac Sim** is the underlying robotics simulator (built on Omniverse/USD).
In this thesis it is responsible for:

- **Scene construction**: ground plane, lights, walls, obstacles, target
  zones — all represented as USD prims on a stage.
- **Robot assets**: loading the Unitree Go2 USD asset (articulation with
  12 actuated joints).
- **Synthetic sensor generation**: RGB / depth cameras, later possibly
  LiDAR and IMU — this is what makes SLAM and VLA evaluation possible.
- **Physics**: PhysX simulation of contacts, joints, and rigid bodies.
- **Domain randomization**: varying lighting, obstacle positions, textures
  to create the perturbed episodes required by the benchmark.

In this starter, `create_empty_sim_scene.py` is the pure-Isaac-Sim script:
it builds a scene from an empty stage (ground, lights, camera, Go2,
obstacles, walls) and runs the interactive simulation loop. Use it to learn
the USD/Isaac Sim workflow and to prototype scene layouts visually.

Important Isaac Sim conventions used in the scripts:

- `SimulationApp(...)` / `AppLauncher` **must be created before importing
  any other `omni`/`isaacsim`/`isaaclab` module**. This is why every script
  has the "launch first" block at the top.
- Scripts are run with the simulator's own Python:
  `./python.sh script.py` (Isaac Sim) or `./isaaclab.sh -p script.py`
  (Isaac Lab).
- Assets are referenced from the Nucleus asset root
  (`get_assets_root_path()`); the exact Go2 asset path is version-dependent
  and is marked as a placeholder in the code.

---

## 3. How Isaac Lab is used in this project

**Isaac Lab** sits on top of Isaac Sim and provides the **robot-learning
layer**: vectorized environments (thousands of parallel copies on one GPU),
standard RL interfaces, sensor configs, terrains, and pre-configured robot
assets (including `UNITREE_GO2_CFG`).

In this thesis, Isaac Lab is used for:

- **The training environment**: a `DirectRLEnv` subclass defines the Go2
  navigation task (observations, actions, rewards, terminations, resets).
- **RL training**: PPO training with many parallel environments — this is
  the RL pillar of the benchmark.
- **Data generation**: the same environment is stepped with scripted or
  trained policies to log episodes for VLA and world model training.
- **Reproducible episode execution**: resets, seeds, and episode boundaries
  are handled by the environment, which is what the benchmark needs for
  fair, repeatable comparisons.

This starter uses Isaac Lab's **direct workflow** (`DirectRLEnv`) because it
keeps everything in one readable class. Isaac Lab also offers a
manager-based workflow (composable observation/reward/termination managers);
you can migrate later if configurability becomes more important than
simplicity.

---

## 4. What should be implemented first

Follow the exposé's time plan. Order of implementation:

1. **Working environment** (exposé phase 1): install Isaac Sim + Isaac Lab
   on the RTX 5090 machine, verify the bundled examples run, locate the Go2
   assets. Output: documented setup, first runnable tests.
2. **Empty scene → simple Go2 scene** (phase 3): run and modify
   `create_empty_sim_scene.py`. Learn USD prims, lighting, obstacles.
3. **Isaac Lab environment** (phase 3): understand `go2_lab_env.py` — the
   observation/reward/termination/reset structure is the core of everything
   that follows.
4. **First runnable end-to-end pipeline** (phase 4): run
   `train_rl_go2.py` for a short training run, `eval_go2.py` for a rollout,
   and the two data-collection scripts. The goal is *stability, logging,
   and completeness*, not performance.
5. Only then: reward tuning, obstacles inside the Lab env, cameras,
   perturbations, and the other approach modules (phases 5–8).

Rule of thumb from the exposé: *"the system must first become runnable,
then initial simulations and early comparisons can begin, and only
afterward can systematic benchmarking be refined and executed."*

---

## 5. Creating the Unitree Go2 environment

Two levels exist in this starter:

### a) Interactive Isaac Sim scene (`create_empty_sim_scene.py`)

- Creates a `World` with physics at 200 Hz and rendering at 60 Hz.
- Adds a ground plane, dome + distant light, a viewer camera.
- References the Go2 USD asset onto the stage
  (`GO2_RELATIVE_PATH` is a **placeholder** — check your Isaac Sim version's
  asset browser under `Isaac/Robots/Unitree/...`).
- Adds a simple navigation layout: four blocks and two walls, defined in
  `OBSTACLE_SPECS` / `WALL_SPECS` dictionaries so they are trivial to extend
  or randomize later.

### b) Isaac Lab RL environment (`go2_lab_env.py`)

- `Go2EnvCfg` (a `@configclass`) defines everything declaratively:
  simulation dt, decimation, flat-plane terrain, `InteractiveSceneCfg`
  with `num_envs` parallel copies, and the robot via
  `UNITREE_GO2_CFG.replace(prim_path="/World/envs/env_.*/Robot")`.
- **Task**: reach a goal sampled on an 8 m circle around each env origin.
- **Actions (12-dim)**: joint position offsets around the default pose,
  scaled by `action_scale`.
- **Observations (48-dim)**: base linear/angular velocity, joint
  positions/velocities, previous actions, goal direction + distance,
  projected gravity. No camera yet — deliberately, to keep the first RL
  loop fast and simple.
- **Rewards**: progress toward the goal + small alive bonus − action-rate
  and joint-velocity penalties. All weights are config fields.
- **Terminations**: fallen (base too low), out of bounds, timeout.

Everything task-specific lives in small named methods
(`_get_observations`, `_get_rewards`, `_get_dones`, `_reset_idx`) so each
piece can be swapped independently as the thesis scenario matures.

---

## 6. Generating synthetic navigation data

The exposé defines a **common episode dataset** as the primary data source.
Each episode should eventually contain synchronized: RGB/depth, camera and
robot poses, obstacle/target positions, robot states, actions, language or
goal descriptions, navigation-state labels, and perturbation/failure labels.

The starter implements the *skeleton* of this:

- `collect_vla_data_go2.py` → **episode-structured** data
  (one `.npz` per episode: observation, action, reward, done, goal).
- `collect_world_model_data_go2.py` → **transition-structured** data
  (chunked `.npz` files of `(s, a, s', r, done)` tuples).

Both use dict-of-arrays buffers whose keys can be extended (add `"rgb"`,
`"depth"`, `"robot_pose"`, `"language_command"`, `"perturbation_label"`, …)
without changing the storage format. This is intentional: the final common
episode schema grows out of these buffers.

Later steps (per exposé): scenario randomization to avoid memorizing one
layout, train/validation/test splits, nominal vs. perturbed episode
labeling, and onset times for perturbation events.

---

## 7. How RL training fits into the project

RL is one of the four compared approaches: a **learned navigation/control
paradigm** trained through interaction. In the benchmark it will be
evaluated on navigation success, reward, robustness to perturbations,
sample efficiency, recovery behavior, and training stability — and its
training cost (episodes, GPU hours) is itself an evaluation result.

`train_rl_go2.py` implements the minimal loop:

- Launches the app via `AppLauncher` (supports `--headless`).
- Builds `Go2Env` with 64 parallel envs, wraps it with
  `SkrlVecEnvWrapper`.
- Defines tiny MLP Gaussian policy + value models and a **PPO** agent
  (skrl), matching the exposé's "PPO/SAC-style policy trained in Isaac
  Lab".
- Trains for 5,000 timesteps — enough to *verify the pipeline runs*, not
  to learn locomotion. Scale `NUM_ENVS` and `TOTAL_TIMESTEPS` up (thousands
  of envs, millions of steps) once everything is stable.
- Logs to `runs/go2_ppo` (checkpoints + TensorBoard-style logs), which
  becomes the checkpoint/experiment-tracking basis required by the
  exposé's "Model Training Scope" section.

To swap the algorithm later, only the agent-construction block needs to be
replaced (e.g. skrl SAC, or the RSL-RL / rl_games wrappers that Isaac Lab
also supports).

---

## 8. How VLA-style data collection fits in

VLA models connect **images + language instructions + actions**. The thesis
will not pretrain a foundation model; feasible options are fine-tuning an
open VLA (OpenVLA/Octo-style), adapting one to the Go2 action space, or
training a smaller VLA-inspired module. All of these need
**demonstration-style episode data** in the form
*(observation, instruction, action)*.

`collect_vla_data_go2.py` provides the collection skeleton:

- Steps the environment with a placeholder policy and stores per-step
  observation/action/reward/done/goal buffers, one `.npz` per episode.
- The buffers are the insertion points for the real VLA data later:
  - `"rgb"` — add a camera to the env (`isaaclab.sensors.CameraCfg` /
    TiledCamera) and record images.
  - `"language_command"` — generate templated instructions from the episode
    layout, e.g. *"walk to the target zone behind the grey wall"* (the goal
    position and obstacle specs are already logged, so instructions can be
    synthesized from them).
  - `"expert_action"` — replace the random placeholder policy with the
    trained RL policy so the dataset contains *successful* demonstrations.

Typical later pipeline: train RL → roll it out with cameras + instruction
templates → convert episodes to the VLA fine-tuning format → fine-tune /
adapt → evaluate under instruction variation.

---

## 9. How world model data collection fits in

World Models learn **predictive dynamics**: given state/observation and
action, predict the next state (Dreamer-style latent dynamics or
JEPA-style representation prediction). In the benchmark, the WM is
evaluated on prediction error, temporal consistency, and especially
**deviation recognition**: if predicted and observed dynamics diverge, that
divergence is a perturbation/failure signal with a measurable detection
delay.

`collect_world_model_data_go2.py` collects the canonical training format:
`(observation, action, next_observation, reward, done)` transitions,
written as chunked `.npz` files. Notes:

- `done` flags mark episode boundaries — the WM trainer must not learn
  across reset transitions.
- Random actions give broad state coverage for a first dynamics model.
  Per the exposé, the WM should ultimately be trained on **nominal**
  navigation behavior (trained-policy rollouts) with perturbed episodes
  reserved for validation/evaluation of deviation detection.
- Extend the buffers with images once cameras exist; the encoder then
  moves from state-based to pixel-based.

The WM training script itself (encoder + transition predictor + rollout
prediction evaluation) is a later step; the data format here is what it
will consume.

---

## 10. How SLAM / pose-based evaluation is added later

SLAM is the **spatial baseline**: it is configured/calibrated, not trained.
Its benchmark role is localization accuracy, map consistency, and
detection of geometric deviations. The exposé notes that full SLAM may not
be necessary at first — visual odometry, pose tracking, or
ground-truth-plus-noise localization are acceptable baselines.

`slam_pose_eval_stub.py` prepares this incrementally:

1. **Already usable now**: `absolute_trajectory_error` (ATE),
   `relative_pose_error` (RPE), and `final_drift` — the standard SLAM
   trajectory metrics, operating on plain `(T, 3)` pose arrays. They work
   on any logged trajectory (the env exposes ground-truth robot poses via
   `robot.data.root_pos_w`).
2. **Controllable baseline**: `NoisyPoseEstimator` simulates drifting
   localization from ground truth. This lets the full evaluation pipeline
   run before any real SLAM exists, and doubles as a *perturbation
   injector* (localization drift is one of the planned perturbation types).
3. **Placeholder**: `ORBSlam3Stub` documents the real integration plan —
   add an RGB(-D) camera to the Go2, feed images (± IMU) to ORB-SLAM3,
   convert its trajectory to the world frame, and evaluate with the same
   ATE/RPE functions.

The script runs with plain Python (no Isaac Sim needed):
`python isaac_unitree_starter/slam_pose_eval_stub.py`.

---

## 11. How the navigation approaches will be benchmarked

The exposé's central methodological point: SLAM, VLA, RL, and WM are **not
interchangeable**, so the comparison is **capability-based**, not one
aggregated ranking. The benchmark will:

- Run all approaches on the **same scenario and perturbation set**, built
  from the same common episode dataset / environment.
- Report per-capability metrics:
  - *Spatial accuracy* (ATE, RPE, drift, map consistency) — mainly SLAM.
  - *Navigation success* (success rate, completion time, path efficiency,
    collisions, recovery) — mainly RL and VLA.
  - *Semantic correctness* (goal fulfillment under instruction variation,
    instruction-mismatch detection) — mainly VLA.
  - *Deviation recognition* (precision, recall, F1, AUROC, false-positive
    rate on nominal episodes, detection delay after perturbation onset) —
    mainly WM.
  - *Robustness* under lighting/layout/noise variation — all approaches.
  - *Training, data, and compute efficiency* + *interpretability and
    implementation effort* — documented for every learned component.
- Keep training / adaptation / evaluation explicitly separated, and
  document what was pretrained, fine-tuned, or trained from scratch.

The `eval_go2.py` rollout script is the seed of this: it already reports
success, episode length, termination reason (goal_reached / fallen /
out_of_bounds / timeout), reward, and final goal distance for one policy.
The full benchmark generalizes this to many episodes, many approaches, and
the metric families above, producing the exposé's evaluation matrix.

---

## 12. What the current starter pipeline contains

```
isaac_unitree_starter/
├── TUTORIAL.md                        ← this file
├── validate_go2_setup.py             ← staged smoke-test: 8 checks (sim launch,
│                                        imports, assets, scene build, Go2 load,
│                                        physics settle, movement through waypoints)
│                                        with PASS/FAIL summary
├── create_empty_sim_scene.py          ← pure Isaac Sim: empty stage → ground,
│                                        lights, camera, Go2 asset, obstacle +
│                                        wall layout, interactive sim loop
├── create_room_scene.py               ← Isaac Sim: 10m×10m room with 4 walls,
│                                        6 obstacles, green goal marker, Go2
├── generate_room_scene_data.py        ← ego-camera traversal through the room
│                                        scene, saves RGB + depth + pose + goal
│                                        as .npz chunks + scene_metadata.json
├── generate_vla_sensor_data_go2.py    ← full multi-sensor VLA data: camera,
│                                        depth, LiDAR, IMU, proprioception,
│                                        actions, language instruction; per-episode
│                                        .npz + dataset_metadata.json
├── go2_lab_env.py                     ← Isaac Lab DirectRLEnv for the Go2:
│                                        goal-reaching task, obs/reward/
│                                        termination/reset structure
├── train_rl_go2.py                    ← minimal PPO training (skrl) on the
│                                        Go2 env, checkpoints in runs/go2_ppo
├── collect_vla_data_go2.py            ← episode-structured data collection
│                                        (VLA-style), .npz per episode into
│                                        data/vla_go2/
├── collect_world_model_data_go2.py    ← transition-structured data collection
│                                        (world-model-style), chunked .npz into
│                                        data/world_model_go2/
├── slam_pose_eval_stub.py             ← ATE/RPE metrics (working), noisy-pose
│                                        baseline (working), ORB-SLAM3 stub
└── eval_go2.py                        ← single rollout + summary statistics
                                         (success, termination reason, reward)
```

Design choices, deliberately:

- **One file per pipeline stage**, each independently runnable — easy to
  modify without touching the rest.
- **Inline configuration constants** at the top of each script instead of a
  config framework — appropriate for the "first runnable" phase; migrate to
  YAML/Hydra configs when the benchmark matures.
- **Placeholders only where versions are uncertain**: the Go2 USD asset
  path, the `UNITREE_GO2_CFG` import location, and the ORB-SLAM3
  integration are the marked version-/asset-dependent points.

---

## 13. What still needs to be implemented later

Roughly in the exposé's implementation order:

- [ ] **Verify version-specific points** on the actual RTX 5090 machine:
      Go2 asset path, `isaaclab_assets` import path, skrl wrapper API.
- [ ] **Obstacles + target zones inside the Lab env** (currently only in the
      Isaac Sim scene script) and collision-aware terminations/penalties.
- [ ] **Cameras and depth sensors** on the Go2 (`CameraCfg`/TiledCamera) and
      image logging in both data-collection scripts.
- [ ] **Scenario randomization**: layout, obstacle positions, lighting,
      goal placement; reproducible seeds; train/val/test scenario splits.
- [ ] **Perturbation injection framework**: obstacle displacement, sensor
      noise, localization drift, instruction mismatch — with onset-time
      labels written into the episode data.
- [ ] **Common episode dataset schema**: consolidate the two data formats
      into one episode schema with all modalities + labels.
- [ ] **Serious RL training**: reward shaping for locomotion + navigation
      (or a hierarchical setup: pretrained locomotion policy + navigation
      commands), thousands of envs, millions of steps, checkpoint
      management, hyperparameter documentation.
- [ ] **World model training script**: encoder + transition predictor,
      rollout prediction, deviation-score computation, detection-delay
      evaluation.
- [ ] **VLA module**: instruction templating, dataset conversion,
      fine-tuning/adaptation of an OpenVLA/Octo-style model or a smaller
      VLA-inspired module.
- [ ] **Real SLAM / visual odometry integration** replacing
      `NoisyPoseEstimator`.
- [ ] **Benchmark runner + evaluation matrix**: multi-episode, multi-approach
      evaluation producing the capability matrix; result tables and plots.
- [ ] Optional: **hybrid architecture** experiments and a **MuJoCo/MJX**
      control-focused side benchmark.

---

## 14. Quick-start: running the pipeline

### Prerequisites

All simulator scripts must run on a machine with **Isaac Sim** (and
optionally **Isaac Lab**) installed — Linux or Windows with an RTX GPU.
Only `slam_pose_eval_stub.py` runs anywhere with plain Python.

There are **two Python interpreters** to choose from:

| Interpreter | When to use | Typical location |
|---|---|---|
| `./python.sh` | **Isaac Sim scripts** (pure `isaacsim` imports, no `isaaclab`) | `<Isaac Sim install>/python.sh` |
| `./isaaclab.sh -p` | **Isaac Lab scripts** (`isaaclab` + `isaaclab_assets` imports) | `<Isaac Lab repo>/isaaclab.sh` |

Scripts that `from isaacsim import SimulationApp` use `./python.sh`.
Scripts that `from isaaclab.app import AppLauncher` use `./isaaclab.sh -p`.

> **Tip**: if you're unsure which interpreter a script needs, look at its
> first non-comment import line. The docstring at the top of each script
> also shows the exact run command.

### Step-by-step

Run these commands **from the Isaac Sim or Isaac Lab install directory**
(the scripts use relative paths inside the project, so pass the full or
relative path to the script file).

```bash
# ─── 0. (anywhere, plain Python) test the pose-metric stubs ───────────
python isaac_unitree_starter/slam_pose_eval_stub.py

# ─── 1. Validate your setup (Isaac Sim interpreter) ──────────────────
#    Runs 8 staged checks: sim launch, imports, Isaac Lab availability,
#    Nucleus assets, room scene, Go2 articulation, physics, movement.
#    Prints a PASS/FAIL summary; exit code 0 = all required checks pass.
./python.sh isaac_unitree_starter/validate_go2_setup.py --headless
#    (omit --headless to watch the robot in the GUI)

# ─── 2. Interactive room scene (Isaac Sim) ────────────────────────────
#    Opens the GUI with the 10m×10m room, obstacles, goal, and Go2.
./python.sh isaac_unitree_starter/create_room_scene.py

# ─── 3. Interactive empty scene (Isaac Sim, original starter) ─────────
./python.sh isaac_unitree_starter/create_empty_sim_scene.py

# ─── 4. Generate basic synthetic data from the room scene ────────────
#    Ego-camera traversal → RGB, depth, pose, goal in .npz chunks.
./python.sh isaac_unitree_starter/generate_room_scene_data.py --headless
#    Options: --num-frames 300  --output-dir data/room_scene_go2

# ─── 5. Generate full multi-sensor VLA data ──────────────────────────
#    Camera + depth + LiDAR + IMU + proprioception + actions + language.
./python.sh isaac_unitree_starter/generate_vla_sensor_data_go2.py --headless
#    Options: --num-episodes 3  --steps-per-episode 200
#             --output-dir data/vla_sensors_go2  --seed 42

# ─── 6. Short RL training run (Isaac Lab interpreter) ────────────────
./isaaclab.sh -p isaac_unitree_starter/train_rl_go2.py --headless

# ─── 7. Evaluation rollout (Isaac Lab) ───────────────────────────────
./isaaclab.sh -p isaac_unitree_starter/eval_go2.py --headless

# ─── 8. VLA-style episode data collection (Isaac Lab) ────────────────
./isaaclab.sh -p isaac_unitree_starter/collect_vla_data_go2.py --headless

# ─── 9. World-model transition data collection (Isaac Lab) ───────────
./isaaclab.sh -p isaac_unitree_starter/collect_world_model_data_go2.py --headless
```

### Recommended order for first-time setup

1. Run `validate_go2_setup.py` first — it tells you exactly what works and
   what doesn't on your install.
2. If the validation passes, run `create_room_scene.py` (GUI) to see the
   scene visually.
3. Then try `generate_room_scene_data.py` or
   `generate_vla_sensor_data_go2.py` to produce your first synthetic dataset.
4. If you also have Isaac Lab installed, continue with `train_rl_go2.py`
   and the remaining Isaac Lab scripts.

### Troubleshooting

- **"ModuleNotFoundError: No module named 'isaacsim'"** — you ran the
  script with the system Python instead of `./python.sh`.
- **"ModuleNotFoundError: No module named 'isaaclab'"** — you used
  `./python.sh` for an Isaac Lab script; use `./isaaclab.sh -p` instead.
- **"Could not resolve assets root path"** — Nucleus is not running or not
  reachable. Start the Nucleus local server from the Omniverse Launcher.
- **Go2 asset path not found** — the `GO2_RELATIVE_PATH` constant
  (`/Isaac/Robots/Unitree/Go2/go2.usd`) is version-dependent. Open the
  Nucleus asset browser in the Omniverse app and search for `Go2` to find
  the correct path, then update the constant in the script.
- **Isaac Lab import paths changed** — the `isaaclab_assets` module layout
  can differ between Isaac Lab versions. Check your version's docs for the
  correct import path for `UNITREE_GO2_CFG`.

If an import fails, the two most likely causes are (a) the script was not
run with the correct Isaac Sim / Isaac Lab Python interpreter, or (b) a
version-dependent module path (marked in the code comments) differs in
your installation.
