"""Inspection task definition, success criteria and outcome.

Specification reference: section 5 (task record fields), section 6 Phase C
("Generate the task"), section 8.8 (viewpoints) and section 8.10 (outcomes).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from navbench.domain.semantics import ParsedInstruction, SemanticInstance
from navbench.domain.types import Pose


class TerminationReason(Enum):
    """Allowed values of ``termination_reason`` (spec section 8.10)."""

    SUCCESS = "success"
    TIMEOUT = "timeout"
    COLLISION = "collision"
    INSTABILITY = "instability"
    WRONG_TARGET = "wrong_target"
    POLICY_TERMINATE = "policy_terminate"
    SIMULATOR_ERROR = "simulator_error"


class DefectType(Enum):
    """Allowed values of ``defect_type`` (spec section 8.9)."""

    NONE = "none"
    CORROSION = "corrosion"
    LEAK = "leak"
    CRACK = "crack"
    DEFORMATION = "deformation"
    LOOSE_JOINT = "loose_joint"
    BLOCKAGE = "blockage"
    OTHER = "other"


class Split(Enum):
    """Dataset split (spec section 8.1 / 11)."""

    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


@dataclass(frozen=True)
class ViewpointRequirement:
    """One required inspection viewpoint (spec section 5 ``required_viewpoints``).

    A viewpoint is *valid* when the base is inside the distance band, the
    bearing error is within tolerance, the target is visible above
    ``min_visible_fraction`` and the line of sight is unobstructed.
    """

    viewpoint_id: str
    component: str
    approach_bearing_rad: float
    min_distance_m: float = 1.0
    max_distance_m: float = 2.5
    max_angle_error_rad: float = 0.5
    min_visible_fraction: float = 0.02
    hold_frames: int = 5

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable mapping."""
        return {
            "viewpoint_id": self.viewpoint_id,
            "component": self.component,
            "approach_bearing_rad": self.approach_bearing_rad,
            "min_distance_m": self.min_distance_m,
            "max_distance_m": self.max_distance_m,
            "max_angle_error_rad": self.max_angle_error_rad,
            "min_visible_fraction": self.min_visible_fraction,
            "hold_frames": self.hold_frames,
        }


@dataclass(frozen=True)
class SuccessCriteria:
    """Geometric completion rules checked by the evaluator (spec section 6 C/E)."""

    required_viewpoint_count: int = 3
    min_distance_m: float = 1.0
    max_distance_m: float = 2.5
    max_angle_error_rad: float = 0.5
    min_visible_fraction: float = 0.02
    required_hold_frames: int = 5
    max_episode_time_s: float = 120.0
    collision_terminates: bool = True
    max_collisions: int = 0
    near_collision_margin_m: float = 0.35

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable mapping."""
        return {
            "required_viewpoint_count": self.required_viewpoint_count,
            "min_distance_m": self.min_distance_m,
            "max_distance_m": self.max_distance_m,
            "max_angle_error_rad": self.max_angle_error_rad,
            "min_visible_fraction": self.min_visible_fraction,
            "required_hold_frames": self.required_hold_frames,
            "max_episode_time_s": self.max_episode_time_s,
            "collision_terminates": self.collision_terminates,
            "max_collisions": self.max_collisions,
            "near_collision_margin_m": self.near_collision_margin_m,
        }


@dataclass
class InspectionTask:
    """The resolved task object written to ``task.json``.

    Holds both the natural-language and the symbolic form (spec section 5), the
    resolved target instance, the required viewpoints and the success criteria.
    """

    instruction: ParsedInstruction
    instruction_paraphrase_id: int
    target: SemanticInstance
    required_components: tuple[str, ...]
    required_viewpoints: tuple[ViewpointRequirement, ...]
    success_criteria: SuccessCriteria
    distractors: tuple[SemanticInstance, ...] = ()
    inspection_modalities: tuple[str, ...] = ("rgb", "depth", "video")

    @property
    def target_prim_path(self) -> str:
        """Exact USD prim path of the target (spec ``target_prim_path``)."""
        return self.target.prim_path

    @property
    def required_viewpoint_count(self) -> int:
        """Number of viewpoints that must be captured."""
        return self.success_criteria.required_viewpoint_count

    def ground_truth_report(self) -> dict[str, Any]:
        """Return ``inspection_result_gt`` (spec section 8.9)."""
        return {
            "instance_name": self.target.instance_name,
            "instance_uid": self.target.instance_uid,
            "condition_label": self.target.condition_label,
            "defect_type": self.target.defect_type,
            "defects": [d.to_dict() for d in self.target.defects],
            "required_components": list(self.required_components),
        }

    def to_dict(self) -> dict[str, Any]:
        """Return the ``task.json`` payload."""
        return {
            "instruction_text": self.instruction.instruction_text,
            "instruction_paraphrase_id": self.instruction_paraphrase_id,
            "task_verb": self.instruction.task_verb,
            "target_class": self.instruction.target_class,
            "target_instance_id": self.instruction.target_instance_id,
            "target_instance_uid": self.target.instance_uid,
            "target_instance_name": self.target.instance_name,
            "target_prim_path": self.target.prim_path,
            "target_pose_world_gt": [float(v) for v in self.target.pose.as_array()],
            "target_bbox_world": self.target.bbox.as_list(),
            "target_evidence": list(self.target.evidence),
            "required_components": list(self.required_components),
            "required_viewpoint_count": self.required_viewpoint_count,
            "required_viewpoints": [v.to_dict() for v in self.required_viewpoints],
            "inspection_modalities": list(self.inspection_modalities),
            "success_criteria": self.success_criteria.to_dict(),
            "max_episode_time_s": self.success_criteria.max_episode_time_s,
            "distractors": [
                {
                    "instance_uid": d.instance_uid,
                    "instance_id": d.instance_id,
                    "instance_name": d.instance_name,
                    "prim_path": d.prim_path,
                }
                for d in self.distractors
            ],
            "inspection_result_gt": self.ground_truth_report(),
        }


@dataclass
class InspectionCapture:
    """One inspection burst triggered at a viewpoint (spec section 8.9)."""

    inspection_capture_id: str
    viewpoint_id: str
    component: str
    start_time_ns: int
    end_time_ns: int
    start_frame_index: int
    end_frame_index: int
    camera_pose_world: Pose
    target_distance_m: float
    target_bearing_rad: float
    target_visible_fraction: float
    viewpoint_valid: bool
    viewpoint_quality_score: float
    burst_frame_count: int
    condition_label_gt: str
    defect_type_gt: str

    def to_dict(self) -> dict[str, Any]:
        """Return the ``inspection/captures.jsonl`` record."""
        return {
            "inspection_capture_id": self.inspection_capture_id,
            "viewpoint_id": self.viewpoint_id,
            "component": self.component,
            "start_time_ns": self.start_time_ns,
            "end_time_ns": self.end_time_ns,
            "start_frame_index": self.start_frame_index,
            "end_frame_index": self.end_frame_index,
            "camera_pose_world": [float(v) for v in self.camera_pose_world.as_array()],
            "target_distance_m": self.target_distance_m,
            "target_bearing_rad": self.target_bearing_rad,
            "target_visible_fraction": self.target_visible_fraction,
            "viewpoint_valid": self.viewpoint_valid,
            "viewpoint_quality_score": self.viewpoint_quality_score,
            "burst_frame_count": self.burst_frame_count,
            "condition_label_gt": self.condition_label_gt,
            "defect_type_gt": self.defect_type_gt,
        }


@dataclass
class EpisodeOutcome:
    """Final labels of an episode (spec section 8.10)."""

    episode_success: bool
    target_found: bool
    valid_viewpoint_reached: bool
    inspection_correct: bool
    termination_reason: TerminationReason
    collision: bool
    near_collision: bool
    collisions: int
    path_length_m: float
    geodesic_reference_m: float
    completion_time_s: float
    energy_proxy: float
    viewpoints_captured: int
    viewpoints_required: int
    visibility_frames: int
    time_to_first_detection_s: float | None
    instance_match_correct: bool
    condition_label_gt: str
    condition_label_pred: str
    recovery_events: list[dict[str, Any]] = field(default_factory=list)

    @property
    def spl(self) -> float:
        """Success weighted by path length (spec section 15 navigation)."""
        if not self.episode_success or self.path_length_m <= 0.0:
            return 0.0
        return float(
            self.geodesic_reference_m / max(self.path_length_m, self.geodesic_reference_m)
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the outcome block of ``episode.json``."""
        return {
            "episode_success": self.episode_success,
            "target_found": self.target_found,
            "valid_viewpoint_reached": self.valid_viewpoint_reached,
            "inspection_correct": self.inspection_correct,
            "termination_reason": self.termination_reason.value,
            "collision": self.collision,
            "near_collision": self.near_collision,
            "collisions": self.collisions,
            "path_length_m": self.path_length_m,
            "geodesic_reference_m": self.geodesic_reference_m,
            "spl": self.spl,
            "completion_time_s": self.completion_time_s,
            "energy_proxy": self.energy_proxy,
            "viewpoints_captured": self.viewpoints_captured,
            "viewpoints_required": self.viewpoints_required,
            "visibility_frames": self.visibility_frames,
            "time_to_first_detection_s": self.time_to_first_detection_s,
            "instance_match_correct": self.instance_match_correct,
            "condition_label_gt": self.condition_label_gt,
            "condition_label_pred": self.condition_label_pred,
            "recovery_events": list(self.recovery_events),
        }