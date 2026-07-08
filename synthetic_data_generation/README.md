# Synthetic Data Generation for the Go2 Benchmark

Scripts to generate synthetic training data on the Unitree Go2 for each of
the four approach classes compared in the thesis benchmark (see
`PROJECT.md`). Every script supports two backends:

- **`--backend mock`** (default) — a deterministic kinematic point-robot
  stand-in for the Go2 (numpy only). Observation/action layout mirrors the
  Isaac Lab env in `isaac_unitree_starter/go2_lab_env.py` (obs=48, act=12).
  Runs anywhere (laptop, CI) — use it to develop data pipelines and models.
- **`--backend isaac`** — the real Isaac Lab Go2 environment. Must be
  launched through the Isaac Lab interpreter (`./isaaclab.sh -p ...`).

## Folder Structure

```
synthetic_data_generation/
├── README.md
├── common/                      # shared utilities (backend-agnostic)
│   ├── backends.py              # MockGo2Backend | IsaacGo2Backend (+factory)
│   ├── policies.py              # random | zero | scripted_goal (+factory)
│   ├── writer.py                # EpisodeWriter / TransitionWriter + manifest.json
│   └── config.py                # YAML config < CLI flag override handling
├── configs/                     # default YAML config per approach
│   ├── slam.yaml
│   ├── rl.yaml
│   ├── vla.yaml
│   └── world_model.yaml
├── slam/
│   └── generate_slam_data.py    # GT + drifting pose-estimate trajectories (ATE/RPE-ready)
├── rl/
│   └── generate_rl_data.py      # (obs, action, reward, done) rollouts (offline RL / BC)
├── vla/
│   └── generate_vla_data.py     # language-instruction-conditioned demonstrations
└── world_model/
    └── generate_world_model_data.py  # flat (s, a, s', r, done) transition chunks
```

## What each approach gets

| Approach    | Script                                        | Data format                                                                 |
|-------------|-----------------------------------------------|------------------------------------------------------------------------------|
| SLAM        | `slam/generate_slam_data.py`                  | Episodes with `gt_position` + drifting `est_position` — feed directly into the ATE/RPE metrics in `isaac_unitree_starter/slam_pose_eval_stub.py` |
| RL          | `rl/generate_rl_data.py`                      | Episode rollouts `(observation, action, reward, done, goal_pos)` for offline RL / behavior cloning / replay pre-fill |
| VLA         | `vla/generate_vla_data.py`                    | Demonstrations + per-env templated language `instruction` (OpenVLA/Octo-style `(instruction, trajectory)` pairs) |
| World Model | `world_model/generate_world_model_data.py`    | Chunked flat transitions `(observation, action, next_observation, reward, done)`; `done` marks reset boundaries |

All outputs are compressed `.npz` files under `data/synthetic/<approach>_go2/`
plus a `manifest.json` recording generation parameters, file names, and array
shapes.

## Usage

Mock backend (runs anywhere, no Isaac required):

```bash
python synthetic_data_generation/slam/generate_slam_data.py
python synthetic_data_generation/rl/generate_rl_data.py
python synthetic_data_generation/vla/generate_vla_data.py
python synthetic_data_generation/world_model/generate_world_model_data.py
```

With a YAML config (CLI flags still override the file):

```bash
python synthetic_data_generation/vla/generate_vla_data.py \
    --config synthetic_data_generation/configs/vla.yaml --num-episodes 50
```

Isaac Lab backend (from the Isaac Lab installation directory):

```bash
./isaaclab.sh -p /path/to/master_thesis_isaac/synthetic_data_generation/rl/generate_rl_data.py \
    --backend isaac --headless
```

## Policies (`--policy`)

- `random`        — uniform actions in [-1, 1]; broad state coverage (world model)
- `zero`          — zero actions + Gaussian noise (`--noise-std`)
- `scripted_goal` — walks toward the goal using the backend's goal direction;
  crude on the Isaac backend (joint bias), exact on the mock backend

## Extension points (marked with TODO in the code)

- `TODO(policy)`      — replace placeholder policies with a trained RL
  checkpoint (`isaac_unitree_starter/train_rl_go2.py`) for realistic gaits.
- `TODO(vla)`         — record camera `rgb` frames and paraphrase instructions.
- `TODO(slam)`        — record `rgb`/`depth` + intrinsics and run ORB-SLAM3
  instead of the drifting-pose simulation.
- `TODO(world_model)` — add image keys; the generic key layout already
  supports this without downstream changes.
- `TODO(isaac)`       — expose `reached_goal` from the Isaac env.