# Unified Synthetic Dataset and SOTA Model Pipelines for Unitree Go2 Machine Inspection

**Version date:** 30 July 2026  
**Primary task:** `Inspect Pipe 0-1`  
**Simulation:** NVIDIA Isaac Sim  
**Robot-learning environment:** NVIDIA Isaac Lab  
**Robot:** Unitree Go2  

## 1. Purpose and design rule

The project uses one synchronized multimodal episode dataset generated while a Unitree Go2 completes an industrial inspection task in Isaac Sim through an Isaac Lab environment.

The canonical task is:

> **Inspect Pipe 0-1.**

The task is considered complete when the Go2:

1. identifies the correct pipe instance;
2. navigates to the required inspection region without collision;
3. reaches one or more valid inspection viewpoints;
4. stops and captures inspection observations;
5. produces a condition result for Pipe 0-1;
6. terminates the episode explicitly.

The central dataset rule is:

> Record every modality and every label once in a canonical episode. Create model-specific views from that episode without recollecting the rollout.

A unified dataset does **not** require every model to consume every modality. It means that all models are trained, adapted, or evaluated from the same underlying scenes, trajectories, timestamps, instructions, actions, and outcomes.

## 2. SOTA selection policy

“SOTA” is benchmark- and date-dependent. The project therefore uses the following selection policy:

1. Select the strongest current model that directly fits the paradigm and task.
2. Require public weights or a sufficiently complete implementation before making it the primary reproducible model.
3. Freeze the exact model version and commit at the beginning of systematic experiments.
4. Record newer but unreleased models as candidates, not as completed baselines.
5. Do not label a custom extension as the published SOTA model. For example, a new LiDAR branch added to OneVLA must be called `OneVLA-LiDAR`, not OneVLA.

## 3. Shared system architecture

**Isaac Sim scene**  
→ industrial machine, Pipe 0-1, obstacles, defects and sensor simulation  
→ **Isaac Lab Go2 inspection environment**  
→ mission instruction and episode state machine  
→ one of the compared navigation/decision pipelines  
→ shared high-level command interface  
→ frozen low-level Go2 locomotion controller  
→ synchronized multimodal recorder  
→ canonical episode folder  
→ model-specific dataset exports.

### Shared high-level action interface

All compared decision approaches control the same frozen Go2 locomotion layer.

| Action field | Meaning | Recommended range or type |
|---|---|---|
| `cmd_vx_mps` | Forward/backward base velocity | continuous, `[-0.8, 0.8] m/s` |
| `cmd_vy_mps` | Lateral base velocity | continuous, `[-0.4, 0.4] m/s` |
| `cmd_yaw_rate_rps` | Yaw angular velocity | continuous, `[-1.0, 1.0] rad/s` |
| `inspect_trigger` | Request stable inspection capture | continuous probability or binary |
| `terminate_trigger` | Declare task completion/failure | binary |

For a discrete-action comparison, the continuous commands are converted to:

- `FORWARD`
- `TURN_LEFT`
- `TURN_RIGHT`
- `STOP_AND_INSPECT`
- `TERMINATE`

The continuous interface should remain the canonical stored action because it can be quantized later, while discrete actions cannot reconstruct the original velocity command.

## 4. Models and exact stack

| Paradigm | Primary model and stack | Model construction | Dataset view | Output | LiDAR use | Training/adaptation in this thesis |
|---|---|---|---|---|---|---|
| **Semantic SLAM and planning** | **FAST-LIVO2** + **Grounded SAM 2** using Grounding DINO 1.6 or DINO-X with SAM 2 + semantic instance registry + **nvblox** + **Nav2** | FAST-LIVO2 fuses synchronized camera, LiDAR and IMU measurements for pose estimation and mapping. Grounded SAM 2 grounds and tracks the text target `pipe`. The instance registry resolves the requested ID `0-1`. Depth/LiDAR and estimated pose place the target in the map. nvblox provides occupancy/traversability. Nav2 plans to validated inspection viewpoints. | ROS 2/rosbag-style camera, LiDAR, IMU, calibration and timestamps; semantic annotations are used to train or validate the target perception module. | Continuous base commands and inspection trigger | **Native and central** | FAST-LIVO2, nvblox and Nav2 are configured, not trained. The semantic detector/tracker may be fine-tuned or distilled using synthetic masks. Viewpoint ranking may be rule-based or learned. |
| **Vision-Language-Action** | **OneVLA** with its released checkpoint, unified navigation action head and a Go2 action adapter | Multi-view RGB history, language instruction and robot state enter OneVLA’s unified vision-language backbone. The flow-matching action head predicts a navigation action chunk. Only the navigation dimensions are mapped to the shared Go2 command interface. Closed-loop replanning executes a short prefix and then observes again. | LeRobot-style episodes containing RGB video, instruction text, robot state, timestamps and expert action chunks. | Navigation action chunk, inspect and termination decision | **Not native**. LiDAR is used only by an external safety shield unless a separately named `OneVLA-LiDAR` extension is implemented. | Fine-tune the released OneVLA checkpoint on successful and corrective Go2 demonstrations. Keep a clear distinction between pretrained parameters, adapters and newly trained action components. |
| **Reinforcement Learning** | **Asymmetric recurrent multimodal PPO in Isaac Lab/RSL-RL** | Policy: RGB encoder + LiDAR range encoder + instruction/target embedding + proprioception encoder → fusion transformer or GRU → Gaussian continuous actor. Critic: the same observations plus privileged ground-truth target pose, local occupancy and defect state during training only. | Interactive Isaac Lab transitions. The canonical recorder stores all on-policy episodes. Optional expert demonstrations initialize the policy by behavior cloning before PPO. | One continuous high-level action per decision step | **Native policy input** | Train the actor-critic in vectorized Isaac Lab environments. Privileged variables are restricted to the critic and curriculum logic, never the deployed actor. This is the strongest reproducible task-specific RL construction rather than a claim of one universal pretrained navigation checkpoint. |
| **Foundation World Model** | **V-JEPA 2-AC** with V-JEPA 2.1 visual features where compatible + action-conditioned predictor + MPPI/CEM planning | RGB/video clips are encoded into latent states. A Go2-specific action-conditioned predictor is post-trained to predict future latent states under candidate actions. A goal/inspection cost scores predicted futures. MPPI or CEM selects the best action sequence and executes only its first action before replanning. | Sequential RGB/video, actions, robot state, goal representation, rewards, termination and next observations. LiDAR can be retained for safety or introduced only as a clearly named multimodal extension. | First action of the optimized candidate sequence | Not native in the published model | Start from released V-JEPA 2-AC weights and post-train the action-conditioned predictor on Go2 navigation sequences. Train a task cost head or goal-latent distance model. |
| **Efficient World Model baseline** | **LeWorldModel (LeWM)** + CEM/MPC | LeWM learns an encoder and action-conditioned latent predictor end-to-end from pixels using next-embedding prediction and latent regularization. At inference, candidate action sequences are rolled forward in latent space and ranked against an inspection-goal latent. | HDF5 or Zarr sequences containing images, actions, next images, optional state, goal image/latent and termination. | First action of the best MPC sequence | Not native | Train LeWM from scratch on the unified synthetic episodes. This is the computationally efficient comparison against the much larger pretrained V-JEPA 2-AC pipeline. |
| **Static and temporal inspection** | **Grounded SAM 2** target isolation + **V-JEPA 2.1** dense/video encoder + task-specific anomaly heads | Grounded SAM 2 isolates and tracks Pipe 0-1. V-JEPA 2.1 encodes high-resolution inspection images or short videos. A static defect head classifies visible defects, while an action/camera-motion-aware temporal predictor detects unexpected changes, vibration, leakage progression or component motion. | Pipe crops, masks, full inspection video, camera pose, defect labels, defect masks, severity, normal sequences and temporal event onset. | Normal/abnormal, defect class, defect region, severity and confidence | LiDAR is normally unnecessary for close visual diagnosis but remains available for geometry and viewpoint validation. | Fine-tune small heads first while keeping the V-JEPA encoder frozen. Unfreeze upper encoder blocks only if synthetic-to-test generalization is inadequate. |
| **Shared low-level locomotion** | Frozen Go2 locomotion policy trained in Isaac Lab with RSL-RL | Desired base velocity and yaw rate → proprioceptive locomotion policy → 12 joint targets. | Terrain and locomotion observations, not the inspection dataset’s semantic labels. | 12 joint commands | Optional terrain rays | Train once, validate, freeze and reuse for every compared high-level approach. |

### Why the RL row is an architecture rather than a named checkpoint

There is no single universally accepted pretrained RL checkpoint that is simultaneously state-of-the-art for Go2, language-conditioned industrial instance inspection, RGB and LiDAR input, and the exact Isaac Lab task. The rigorous choice is therefore a current high-performance RL construction: asymmetric actor-critic training, recurrent multimodal fusion, privileged critic information, domain randomization and a frozen locomotion layer.

A diffusion navigation policy such as NavDP can be added as a separate **imitation/diffusion-policy baseline**, but it should not be mislabeled as reinforcement learning.

## 5. Unified instruction handling

The external instruction is identical for all approaches:

> **Inspect Pipe 0-1.**

The canonical task record stores both natural-language and symbolic forms:

| Field | Value example |
|---|---|
| `instruction_text` | `Inspect Pipe 0-1.` |
| `task_verb` | `inspect` |
| `target_class` | `pipe` |
| `target_instance_id` | `0-1` |
| `target_asset_path` | `/World/Factory/Machine_0/Pipe_0_1` |
| `required_components` | `[surface, joint_a, joint_b]` |
| `required_viewpoint_count` | `3` |
| `inspection_modalities` | `[rgb, depth, video]` |

Approach-specific interpretation:

- **Semantic SLAM:** the task parser queries the semantic instance registry.
- **OneVLA:** the original language instruction is part of the model input.
- **RL:** a learned instruction/target embedding is part of the actor observation.
- **V-JEPA 2-AC:** language is converted to a target or goal latent by a goal encoder or policy prior.
- **LeWM:** the instruction is converted to one or more goal images/latents because native LeWM is not language-conditioned.

Pipe 0-1 must be distinguishable from other pipe instances. The simulation should provide at least one of:

1. a visible instance label or plate;
2. a unique geometry/material configuration;
3. a stable parent-machine and topology relationship;
4. a semantic map registry linking `Pipe 0-1` to its world instance.

Without such information, asking a visual policy to distinguish identical unlabeled pipes is ill-posed.

## 6. Scene-to-dataset generation pipeline

### Phase A — Load and validate the scene

1. Load the selected factory or machine USD stage.
2. Spawn the Go2 at a valid starting pose.
3. Attach and calibrate the RGB-D camera, LiDAR and IMU.
4. Register every inspectable asset with a stable semantic class and instance ID.
5. Confirm that `/World/.../Pipe_0_1` exists and has inspection surfaces/viewpoints.
6. Confirm collision meshes, traversable surfaces and sensor visibility.

### Phase B — Randomize one episode

Randomize only variables included in the episode metadata:

- Go2 start pose;
- target pipe pose where physically valid;
- non-target machine positions;
- obstacle positions and motion;
- illumination intensity, direction and color temperature;
- material and texture variants;
- camera exposure and noise;
- LiDAR range noise, point dropout and intensity noise;
- IMU bias and white noise;
- defect type, location, severity and onset time;
- partial occlusions;
- distractor pipes and instance labels;
- instruction paraphrase.

The randomization seed must reproduce the complete episode.

### Phase C — Generate the task

Create the task object:

- instruction: `Inspect Pipe 0-1.`
- target: `pipe`, instance `0-1`;
- required inspection viewpoints;
- valid distance and angle ranges;
- minimum target visibility;
- defect-analysis requirements;
- maximum episode duration;
- collision and safety termination rules.

### Phase D — Execute the task to generate demonstrations

Use the semantic SLAM/planning pipeline as the initial expert demonstrator:

1. localize and map with FAST-LIVO2;
2. ground and track the target with Grounded SAM 2;
3. resolve the target to instance `0-1`;
4. generate candidate inspection viewpoints;
5. plan and navigate to the first viewpoint;
6. stabilize the Go2;
7. trigger inspection capture;
8. repeat for all required viewpoints;
9. classify or record the target condition;
10. terminate as success or failure.

The recorder runs throughout the rollout. It captures observations before and after each action, including failures, recovery behavior and inspection events.

Dataset diversity should include:

- successful expert trajectories;
- deliberately suboptimal but successful trajectories;
- recoveries from temporary occlusion or blocked paths;
- collisions and near-collisions;
- wrong-instance approaches;
- premature inspection triggers;
- timeouts;
- sensor-degraded episodes;
- defect and no-defect episodes.

### Phase E — Validate before committing the episode

Write the rollout first to a temporary folder. Commit it only when the following checks pass:

- all mandatory streams exist;
- timestamps are monotonic;
- calibration is present;
- frame and state counts are consistent;
- target ID is valid;
- action and next-observation pairs are aligned;
- termination reason is present;
- success labels match geometric completion rules;
- inspection frames satisfy or explicitly fail viewpoint requirements;
- files pass checksum and readability checks.

Then rename the temporary folder atomically to its final episode name.

## 7. Recommended timing and synchronization

| Process | Recommended rate | Canonical storage rule |
|---|---:|---|
| Physics simulation | 200 Hz | Do not store every full sensor observation; store physics-critical state or subsample |
| Low-level locomotion control | 50 Hz | Store desired and executed joint commands |
| Proprioception | 50 Hz | Store joint/base/contact state |
| IMU | 100 Hz | Store raw measurement and simulated bias/noise parameters |
| High-level decision policy | 5 Hz | Store observation timestamp, requested action and executed action |
| RGB and depth | 10 Hz | Store synchronized frame timestamps |
| LiDAR | 10 Hz | Store point cloud and range representation with scan timestamp |
| Semantic/instance annotations | 10 Hz | Align with camera frames |
| Inspection burst | 30 Hz for 2–5 s | Store high-quality short video around each inspection trigger |

Every record uses a common simulation clock. Sensor-specific timestamps must not be replaced by file-order assumptions.

## 8. Canonical variables to capture

### 8.1 Episode, task and provenance

| Variable | Type | Required | Meaning |
|---|---|---:|---|
| `dataset_version` | string | Yes | Dataset schema version |
| `episode_id` | string/integer | Yes | Globally unique episode identifier |
| `split` | enum | Yes | `train`, `validation`, `test` |
| `scene_id` | string | Yes | Source USD scene identifier |
| `scene_usd_path` | string | Yes | Loaded scene asset |
| `scene_seed` | integer | Yes | Complete deterministic randomization seed |
| `generator_commit` | string | Yes | Repository commit that generated the episode |
| `isaac_sim_version` | string | Yes | Simulator version |
| `isaac_lab_version` | string | Yes | Environment version |
| `robot_asset_version` | string | Yes | Go2 USD/URDF version |
| `policy_name` | string | Yes | Expert or learned policy used for collection |
| `policy_checkpoint` | string | Conditional | Checkpoint hash or identifier |
| `instruction_text` | string | Yes | Natural-language command |
| `instruction_paraphrase_id` | integer | Yes | Language augmentation variant |
| `task_verb` | string | Yes | `inspect` |
| `target_class` | string | Yes | `pipe` |
| `target_instance_id` | string | Yes | `0-1` |
| `target_prim_path` | string | Yes | Exact USD prim path |
| `required_viewpoints` | structured list | Yes | Required inspection poses or regions |
| `max_episode_time_s` | float | Yes | Timeout threshold |
| `episode_start_ns` | integer | Yes | Start on common simulation clock |
| `episode_end_ns` | integer | Yes | End on common simulation clock |

### 8.2 Sensor calibration and synchronization

| Variable | Shape/type | Required | Meaning |
|---|---|---:|---|
| `timestamp_ns` | scalar per sample | Yes | Common-clock timestamp |
| `frame_index` | integer | Yes | Monotonic observation index |
| `camera_intrinsics` | `3×3` | Yes | RGB/depth camera intrinsics |
| `camera_distortion` | vector | Yes | Distortion model and coefficients |
| `T_base_camera` | `4×4` | Yes | Camera extrinsic transform |
| `T_base_lidar` | `4×4` | Yes | LiDAR extrinsic transform |
| `T_base_imu` | `4×4` | Yes | IMU extrinsic transform |
| `sensor_time_offset_ns` | scalar per sensor | Yes | Simulated clock offset |
| `exposure_time_s` | float | Recommended | Camera exposure |
| `rolling_shutter_params` | structured | Optional | Rolling-shutter simulation settings |

### 8.3 Camera and rendered perception

| Variable | Shape/format | Recommended rate | Meaning |
|---|---|---:|---|
| `rgb_front` | `H×W×3 uint8` or MP4 | 10 Hz | Main visual observation |
| `rgb_left/right` | `H×W×3 uint8` or MP4 | 10 Hz | Optional multi-view input |
| `depth_front_m` | `H×W float32` | 10 Hz | Metric depth |
| `surface_normals` | `H×W×3 float16/32` | 10 Hz | Geometry supervision |
| `optical_flow` | `H×W×2 float16/32` | 10 Hz | Motion/temporal supervision |
| `semantic_segmentation` | `H×W uint16` | 10 Hz | Semantic class label per pixel |
| `instance_segmentation` | `H×W uint32` | 10 Hz | Stable instance label per pixel |
| `target_mask` | `H×W bool` | 10 Hz | Pipe 0-1 mask |
| `defect_mask` | `H×W bool` | Inspection frames | Defect-region ground truth |
| `bbox_2d_tight` | list of boxes | 10 Hz | Tight 2D boxes |
| `bbox_2d_loose` | list of boxes | Optional | Full projected object boxes |
| `keypoints_2d` | list | Inspection frames | Gauge/valve/inspection keypoints if used |
| `camera_pose_world` | `7D` or `4×4` | 10 Hz | Camera ground-truth pose |
| `render_settings` | structured | Per episode | Lighting, exposure and renderer settings |

### 8.4 LiDAR

Store both the raw point cloud and a policy-friendly fixed-shape representation.

| Variable | Shape/type | Recommended rate | Meaning |
|---|---|---:|---|
| `lidar_points_xyz` | `N×3 float32` | 10 Hz | Point coordinates in sensor frame |
| `lidar_intensity` | `N float32` | 10 Hz | Simulated return intensity |
| `lidar_ring` | `N uint16` | 10 Hz | Laser/ring index |
| `lidar_time_offset_s` | `N float32` | 10 Hz | Per-point time within scan |
| `lidar_valid` | `N bool` | 10 Hz | Valid-return flag |
| `lidar_range_image_m` | fixed `R×A float32` | 10 Hz | Neural-policy input representation |
| `lidar_min_range_m` | scalar | Per episode | Sensor configuration |
| `lidar_max_range_m` | scalar | Per episode | Sensor configuration |
| `lidar_noise_sigma_m` | scalar | Per episode | Applied range noise |
| `lidar_dropout_rate` | scalar | Per episode | Applied random dropout |
| `lidar_pose_world` | `7D` or `4×4` | 10 Hz | LiDAR ground-truth pose |

### 8.5 IMU and state estimation

| Variable | Shape/type | Recommended rate | Meaning |
|---|---|---:|---|
| `imu_accel_mps2` | `3D float32` | 100 Hz | Raw linear acceleration |
| `imu_gyro_rps` | `3D float32` | 100 Hz | Raw angular velocity |
| `imu_orientation_xyzw` | `4D float32` | 100 Hz | Optional orientation estimate/ground truth |
| `imu_accel_bias` | `3D float32` | Per episode/step | Simulated accelerometer bias |
| `imu_gyro_bias` | `3D float32` | Per episode/step | Simulated gyro bias |
| `imu_temperature` | float | Optional | Sensor-condition randomization |
| `slam_pose_world` | `7D` | At estimator rate | Estimated robot pose |
| `slam_covariance` | `6×6` | At estimator rate | Pose uncertainty |
| `slam_tracking_state` | enum | At estimator rate | Tracking/limited/lost |
| `map_revision_id` | integer | On update | Map version for reproducibility |

### 8.6 Go2 proprioception and contacts

| Variable | Shape/type | Recommended rate | Meaning |
|---|---|---:|---|
| `base_pose_world_gt` | `7D` | 50 Hz | Privileged ground-truth base pose |
| `base_linear_velocity` | `3D` | 50 Hz | Base velocity |
| `base_angular_velocity` | `3D` | 50 Hz | Base angular velocity |
| `projected_gravity` | `3D` | 50 Hz | Body orientation feature |
| `joint_position` | `12D` | 50 Hz | Go2 joint positions |
| `joint_velocity` | `12D` | 50 Hz | Go2 joint velocities |
| `joint_effort` | `12D` | 50 Hz | Joint effort/torque |
| `joint_position_target` | `12D` | 50 Hz | Executed low-level target |
| `foot_contact` | `4D bool` | 50 Hz | Contact state |
| `foot_contact_force` | `4×3` | 50 Hz | Contact force vectors |
| `body_collision_events` | event list | Event-based | Non-foot collisions |
| `stability_margin` | float | 50 Hz | Optional locomotion safety metric |

### 8.7 Actions and policy state

| Variable | Shape/type | Recommended rate | Meaning |
|---|---|---:|---|
| `action_requested` | `5D` | 5 Hz | Raw high-level policy output |
| `action_executed` | `5D` | 5 Hz | Command after safety filtering |
| `action_source` | enum | 5 Hz | expert, OneVLA, PPO, V-JEPA, LeWM, human |
| `action_chunk` | `K×5` | Conditional | Predicted future action chunk |
| `action_log_probability` | float | RL only | PPO policy log probability |
| `value_estimate` | float | RL only | Critic estimate |
| `policy_hidden_state` | vector/reference | Optional | Recurrent-policy state |
| `safety_override` | bool | 5 Hz | Whether LiDAR safety changed the action |
| `safety_override_reason` | enum | Conditional | Collision, instability, velocity limit |
| `previous_action` | `5D` | 5 Hz | Previous executed action |

### 8.8 Target, semantic map and inspection viewpoints

| Variable | Shape/type | Recommended rate | Meaning |
|---|---|---:|---|
| `target_pose_world_gt` | `7D` | 10 Hz/static | Pipe 0-1 ground-truth pose |
| `target_pose_world_est` | `7D` | On detection | Estimated target pose |
| `target_pose_covariance` | `6×6` | On detection | Estimation uncertainty |
| `target_relative_pose_gt` | `7D` | 10 Hz | Privileged relative target pose |
| `target_relative_pose_est` | `7D` | On detection | Policy-available estimate where applicable |
| `target_visible` | bool | 10 Hz | Any target pixels visible |
| `target_visible_fraction` | float | 10 Hz | Visible target surface/pixel fraction |
| `target_occlusion_fraction` | float | 10 Hz | Occluded fraction |
| `target_distance_m` | float | 10 Hz | Robot/camera to target |
| `target_bearing_rad` | float | 10 Hz | Target bearing |
| `grounding_confidence` | float | On detection | Target grounding score |
| `instance_match_correct` | bool | On detection | Whether Pipe 0-1, not another pipe, was selected |
| `viewpoint_id` | string | On approach | Required/candidate viewpoint identifier |
| `viewpoint_pose_world` | `7D` | Per viewpoint | Desired camera/base pose |
| `viewpoint_distance_error_m` | float | 10 Hz | Position error |
| `viewpoint_angle_error_rad` | float | 10 Hz | Orientation error |
| `viewpoint_visibility_score` | float | 10 Hz | Target coverage score |
| `viewpoint_quality_score` | float | 10 Hz | Combined inspection-view metric |
| `viewpoint_valid` | bool | 10 Hz | Meets all capture criteria |

### 8.9 Defect and inspection labels

| Variable | Type | Required | Meaning |
|---|---|---:|---|
| `condition_label` | enum | Yes | normal/abnormal |
| `defect_type` | enum | Yes | none, corrosion, leak, crack, deformation, loose_joint, blockage, other |
| `defect_instance_id` | string | Conditional | Unique defect identifier |
| `defect_pose_world` | `7D` | Conditional | Defect position/orientation |
| `defect_severity` | ordinal/float | Conditional | Severity label |
| `defect_start_time_ns` | integer | Temporal defect | Event onset |
| `defect_end_time_ns` | integer | Temporal defect | Event end |
| `inspection_triggered` | bool | Yes | Capture requested |
| `inspection_capture_id` | string | Conditional | Links frames/video to event |
| `inspection_result_pred` | structured | Evaluation | Predicted result |
| `inspection_result_gt` | structured | Yes | Ground truth result |
| `anomaly_score` | float | Evaluation | Model anomaly score |
| `defect_class_confidence` | float | Evaluation | Class confidence |
| `temporal_deviation_score` | float | Evaluation | Dynamic anomaly score |

### 8.10 Rewards, events and outcomes

| Variable | Type | Required | Meaning |
|---|---|---:|---|
| `reward_total` | float | RL/WM | Total transition reward |
| `reward_progress` | float | RL/WM | Target/viewpoint progress |
| `reward_visibility` | float | RL/WM | Target visibility reward |
| `reward_viewpoint` | float | RL/WM | Inspection-pose quality reward |
| `reward_inspection` | float | RL/WM | Correct capture reward |
| `penalty_collision` | float | RL/WM | Collision penalty |
| `penalty_time` | float | RL/WM | Time penalty |
| `collision` | bool | Yes | Collision occurred |
| `near_collision` | bool | Recommended | Safety-margin violation |
| `recovery_event` | event list | Recommended | Recovery from obstacle/lost target |
| `episode_success` | bool | Yes | Full task success |
| `target_found` | bool | Yes | Correct target located |
| `valid_viewpoint_reached` | bool | Yes | At least required viewpoint reached |
| `inspection_correct` | bool | Yes | Condition classification correct |
| `termination_reason` | enum | Yes | success, timeout, collision, instability, wrong_target, policy_terminate, simulator_error |
| `path_length_m` | float | Yes | Executed path length |
| `geodesic_reference_m` | float | Recommended | Shortest feasible path |
| `completion_time_s` | float | Yes | Task duration |
| `energy_proxy` | float | Recommended | Joint-effort-based energy estimate |

## 9. Canonical folder structure

```text
unified_go2_inspection_dataset/
├── dataset_manifest.json
├── schema/
│   ├── canonical_schema.json
│   ├── label_definitions.json
│   └── action_definition.json
├── calibration/
│   ├── camera_intrinsics.json
│   ├── sensor_extrinsics.json
│   └── sensor_models.json
├── scenes/
│   └── scene_registry.jsonl
├── tasks/
│   ├── tasks.jsonl
│   └── instruction_paraphrases.jsonl
├── episodes/
│   ├── train/
│   │   └── episode_000000/
│   ├── validation/
│   └── test/
└── exports/
    ├── fast_livo2_rosbag2/
    ├── onevla_lerobot/
    ├── ppo_transitions/
    ├── vjepa2_sequences/
    ├── lewm_hdf5/
    └── inspection_dataset/
```

### One canonical episode folder

```text
episode_000000/
├── episode.json
├── task.json
├── randomization.json
├── calibration.json
├── trajectory.parquet
├── rewards.parquet
├── events.parquet
├── sensors/
│   ├── rgb_front.mp4
│   ├── rgb_left.mp4
│   ├── rgb_right.mp4
│   ├── depth_front.zarr
│   ├── lidar_points.zarr
│   ├── lidar_range_image.zarr
│   ├── imu.parquet
│   └── proprioception.parquet
├── actions/
│   ├── high_level.parquet
│   ├── low_level.parquet
│   └── action_chunks.zarr
├── annotations/
│   ├── semantic_segmentation/
│   ├── instance_segmentation/
│   ├── target_masks/
│   ├── defect_masks/
│   ├── object_poses.parquet
│   ├── bounding_boxes_2d.parquet
│   ├── bounding_boxes_3d.parquet
│   └── viewpoint_labels.parquet
├── mapping/
│   ├── slam_trajectory.parquet
│   ├── slam_covariance.parquet
│   └── map_metadata.json
├── inspection/
│   ├── captures.jsonl
│   ├── capture_000.mp4
│   ├── capture_000_frames/
│   └── ground_truth_report.json
└── checksums.sha256
```

Recommended formats:

- **MP4:** compressed RGB streams and inspection clips;
- **Parquet:** timestamped tabular state, action, reward and event data;
- **Zarr:** large chunked depth, LiDAR, masks and action-sequence arrays;
- **PNG:** optional lossless masks and selected evidence frames;
- **JSON/JSONL:** metadata, tasks, calibration and reports;
- **rosbag2 export:** FAST-LIVO2 and ROS 2 replay;
- **LeRobot export:** OneVLA fine-tuning;
- **HDF5 export:** LeWM compatibility where required.

## 10. Model-specific exports from the same canonical episode

| Export | Included variables | Excluded or privileged variables |
|---|---|---|
| **FAST-LIVO2 rosbag2** | RGB, LiDAR, IMU, calibration, sensor timestamps | Ground-truth pose excluded from estimator input; retained separately for evaluation |
| **OneVLA LeRobot** | RGB view(s), instruction, robot state, action chunks, timestamps, episode/task metadata | LiDAR excluded from native OneVLA input; ground-truth target pose excluded |
| **PPO interactive transition view** | RGB features/frames, LiDAR range image, instruction embedding, proprioception, action, reward, next observation, done | Ground-truth target/map available only to privileged critic and training curriculum |
| **V-JEPA 2-AC sequence view** | RGB/video window, previous actions, candidate actions, next video/latent targets, robot state, goal representation | Exact map and target pose excluded unless explicitly used only for cost-label generation |
| **LeWM HDF5** | Current image sequence, action sequence, next image sequence, goal image/latent, termination | Language converted to goal representation; LiDAR excluded from native model |
| **Inspection view** | Target crops, masks, video, camera pose, condition label, defect type, defect mask, event onset | Navigation rewards and unrelated scene observations omitted |

## 11. Train/validation/test splitting

Split by scene and asset configuration, not by individual frames.

Recommended initial split:

- 70% training scenes;
- 15% validation scenes;
- 15% test scenes.

The test set should include held-out combinations of:

- factory layouts;
- pipe geometry and texture;
- target position;
- distractor arrangement;
- obstacle motion;
- lighting;
- defect location and severity;
- instruction paraphrase;
- sensor degradation.

Do not allow frames or trajectories from one randomized scene instance to appear across multiple splits.

## 12. Data-generation volumes

A reasonable staged target is:

| Stage | Episodes | Purpose |
|---|---:|---|
| Pipeline verification | 50–100 | Validate synchronization, labels and folders |
| Perception pilot | 1,000 | Validate target grounding, masks and defects |
| VLA demonstration set | 5,000–20,000 successful/corrective episodes | OneVLA adaptation |
| World Model set | 20,000–100,000 mixed episodes | Diverse transitions and failure dynamics |
| RL | Online vectorized rollouts, potentially millions of transitions | PPO interaction training |
| Final held-out benchmark | At least 1,000 episodes | Statistically meaningful comparison |

The exact volume should be justified by learning curves rather than treated as a fixed requirement.

## 13. Required dataset quality checks

Each generation run should report:

- missing-stream rate;
- timestamp skew between modalities;
- dropped camera frames;
- invalid LiDAR scans;
- target visibility distribution;
- defect-class balance;
- success/failure balance;
- path-length distribution;
- viewpoint-quality distribution;
- action distribution;
- scene and seed uniqueness;
- checksum failures;
- disk usage per modality;
- proportion of episodes rejected by validation.

## 14. Fair comparison rules

1. Every model is evaluated on the same held-out scene seeds and task definitions.
2. Every model uses the same frozen low-level locomotion controller.
3. Every model is constrained by the same high-level velocity and safety limits.
4. Native model inputs are documented explicitly.
5. Privileged data are never used by deployed policies.
6. A shared LiDAR safety shield may be evaluated in a separate safety-controlled track.
7. Native models and custom multimodal extensions are reported separately.
8. Navigation and inspection metrics are reported separately before computing end-to-end mission success.
9. The target instance must be identifiable from available evidence; otherwise the episode is invalid.
10. Model versions, weights, commits, preprocessing and action adapters are frozen and recorded.

## 15. Evaluation outputs

### Target grounding

- correct-instance grounding rate;
- wrong-pipe selection rate;
- time to first correct detection;
- target mask IoU;
- target pose error.

### Navigation

- task success rate;
- collision and near-collision rate;
- completion time;
- path length;
- SPL/path efficiency;
- recovery rate;
- localization ATE/RPE for the SLAM pipeline;
- target-loss duration;
- action latency.

### Inspection viewpoint

- valid-viewpoint success rate;
- camera-target distance error;
- camera-target angle error;
- visible target fraction;
- number of required surfaces captured;
- inspection-trigger precision.

### Inspection result

- normal/abnormal AUROC;
- defect classification macro-F1;
- defect-mask IoU;
- temporal-event detection delay;
- false-positive rate on normal pipes;
- confidence calibration.

### Efficiency

- training GPU-hours;
- number of environment transitions;
- inference latency;
- peak GPU memory;
- dataset storage;
- implementation and integration effort.

## 16. Final recommended comparison

### Primary four-paradigm comparison

1. **Semantic SLAM:** FAST-LIVO2 + Grounded SAM 2 + semantic registry + nvblox + Nav2.
2. **VLA:** OneVLA adapted to Go2 high-level navigation and inspection actions.
3. **RL:** asymmetric recurrent RGB-LiDAR PPO in Isaac Lab/RSL-RL.
4. **World Model:** V-JEPA 2-AC post-trained on Go2 action-conditioned sequences with MPPI/CEM planning.

### Required secondary baselines

- **LeWM:** efficient end-to-end world-model baseline;
- **NavDP:** optional diffusion-policy navigation baseline, reported outside the RL category;
- **Ground-truth expert:** privileged planner used only to establish an upper bound and generate demonstrations;
- **random/reactive baseline:** minimum-performance reference.

### Shared inspection system

- Grounded SAM 2 isolates Pipe 0-1;
- V-JEPA 2.1 encodes static and temporal inspection observations;
- small defect and temporal-deviation heads produce the final inspection result.

## 17. End-to-end generation summary

**Load USD scene**  
→ register Pipe 0-1  
→ spawn Go2 and sensor rig  
→ apply reproducible randomization  
→ create instruction `Inspect Pipe 0-1`  
→ execute the task with the expert or selected policy  
→ capture synchronized RGB, depth, LiDAR, IMU, state, action, semantic and defect data  
→ trigger high-rate inspection capture at valid viewpoints  
→ store success, failure and event labels  
→ validate the episode  
→ atomically write one canonical episode folder  
→ export the same episode into FAST-LIVO2, OneVLA, PPO, V-JEPA 2-AC, LeWM and inspection-specific formats.

## 18. Implementation-freeze references

At the beginning of implementation, record the exact commit, checkpoint and license for:

- NVIDIA Isaac Sim and Isaac Lab;
- FAST-LIVO2;
- Grounded SAM 2, Grounding DINO/DINO-X and SAM 2;
- OneVLA;
- RSL-RL;
- V-JEPA 2 and V-JEPA 2.1;
- LeWorldModel;
- NavDP if included;
- nvblox and Nav2.

The literature and model selection should be rechecked immediately before the experimental freeze because newer model releases may change which implementation is the strongest reproducible option.
