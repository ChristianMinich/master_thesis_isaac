"""SLAM-style / pose-based evaluation stubs for the Go2 navigation benchmark.

This module does NOT run a real SLAM system yet. It provides:

1. Trajectory-metric functions (ATE, RPE) that already work on pose arrays.
   These are the standard SLAM evaluation metrics and can be used as soon as
   an estimated trajectory exists (from visual odometry, ORB-SLAM3, or a
   noisy-pose baseline).
2. A `NoisyPoseEstimator` baseline that simulates a drifting pose estimate
   from ground truth. This lets the whole evaluation pipeline run before a
   real SLAM system is integrated.
3. An `ORBSlam3Stub` placeholder showing where a real SLAM system would be
   connected (fed with simulated camera images from Isaac Sim).

Runs with plain Python (no Isaac Sim required) so metrics can be developed
and tested independently of the simulator:
    python isaac_unitree_starter/slam_pose_eval_stub.py
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Trajectory metrics (these are real, usable now)
# ---------------------------------------------------------------------------


def absolute_trajectory_error(gt_xyz: np.ndarray, est_xyz: np.ndarray) -> float:
    """Absolute Trajectory Error (ATE): RMSE of position differences.

    Args:
        gt_xyz:  (T, 3) ground-truth positions.
        est_xyz: (T, 3) estimated positions (already aligned / same frame).

    Note: full ATE typically includes an SE(3)/Sim(3) alignment step
    (e.g. Umeyama). For a simulation with a shared world frame, direct
    comparison is acceptable as a starting point.
    """
    assert gt_xyz.shape == est_xyz.shape, "trajectories must have equal length"
    errors = np.linalg.norm(gt_xyz - est_xyz, axis=-1)
    return float(np.sqrt(np.mean(errors**2)))


def relative_pose_error(gt_xyz: np.ndarray, est_xyz: np.ndarray, delta: int = 10) -> float:
    """Relative Pose Error (RPE) on translation: RMSE of relative motion error.

    Compares the motion over `delta` steps in the ground-truth trajectory
    against the same motion in the estimated trajectory. Captures drift
    independent of a global offset.
    """
    assert gt_xyz.shape == est_xyz.shape
    gt_rel = gt_xyz[delta:] - gt_xyz[:-delta]
    est_rel = est_xyz[delta:] - est_xyz[:-delta]
    errors = np.linalg.norm(gt_rel - est_rel, axis=-1)
    return float(np.sqrt(np.mean(errors**2)))


def final_drift(gt_xyz: np.ndarray, est_xyz: np.ndarray) -> float:
    """Distance between estimated and true final position."""
    return float(np.linalg.norm(gt_xyz[-1] - est_xyz[-1]))


# ---------------------------------------------------------------------------
# Baseline pose estimator: ground truth + accumulated noise (drift)
# ---------------------------------------------------------------------------


class NoisyPoseEstimator:
    """Simulates a drifting localization system from ground-truth poses.

    Useful as a controllable stand-in for SLAM/odometry: the drift level is
    a parameter, so localization-degradation perturbations (as planned in
    the thesis expose) can be injected deliberately.
    """

    def __init__(self, noise_std: float = 0.005, seed: int = 0):
        self.noise_std = noise_std
        self._rng = np.random.default_rng(seed)
        self._drift = np.zeros(3)

    def reset(self):
        self._drift = np.zeros(3)

    def estimate(self, gt_position: np.ndarray) -> np.ndarray:
        """Return a noisy pose estimate. Drift accumulates over time."""
        self._drift += self._rng.normal(0.0, self.noise_std, size=3)
        return gt_position + self._drift


# ---------------------------------------------------------------------------
# Placeholder for a real SLAM integration
# ---------------------------------------------------------------------------


class ORBSlam3Stub:
    """Placeholder for a real visual(-inertial) SLAM system, e.g. ORB-SLAM3.

    Planned integration (later thesis phase):
      1. Add an RGB(-D) camera to the Go2 in the Isaac Lab environment
         (isaaclab.sensors.CameraCfg) and record images + intrinsics.
      2. Feed images (and optionally IMU) into ORB-SLAM3 via its Python
         bindings or by writing images to disk and running it offline.
      3. Convert the SLAM trajectory into the world frame and evaluate it
         with `absolute_trajectory_error` / `relative_pose_error` above.
    """

    def __init__(self):
        raise NotImplementedError(
            "Real SLAM integration is a later thesis step. "
            "Use NoisyPoseEstimator as the pose baseline for now."
        )


# ---------------------------------------------------------------------------
# Demo: evaluate the noisy-pose baseline on a synthetic trajectory
# ---------------------------------------------------------------------------


def _demo():
    # synthetic ground-truth trajectory: robot walks an arc toward a goal
    steps = 400
    t = np.linspace(0.0, 1.0, steps)
    gt = np.stack(
        [8.0 * t, 1.5 * np.sin(np.pi * t), 0.35 * np.ones_like(t)],
        axis=-1,
    )  # (T, 3)

    estimator = NoisyPoseEstimator(noise_std=0.01, seed=42)
    est = np.stack([estimator.estimate(p) for p in gt], axis=0)

    print("=" * 50)
    print("Pose-based evaluation (noisy-pose baseline demo)")
    print("=" * 50)
    print(f"  ATE (RMSE)   : {absolute_trajectory_error(gt, est):.4f} m")
    print(f"  RPE (d=10)   : {relative_pose_error(gt, est, delta=10):.4f} m")
    print(f"  final drift  : {final_drift(gt, est):.4f} m")
    print("=" * 50)
    print("Replace the synthetic trajectory with logged robot poses from the")
    print("Isaac Lab environment, and NoisyPoseEstimator with a real SLAM /")
    print("visual-odometry system in a later thesis phase.")


if __name__ == "__main__":
    _demo()