"""Semantic instance registry and instruction handling.

Specification reference: section 5 ("Unified instruction handling") and
section 6 Phase A step 4 ("Register every inspectable asset with a stable
semantic class and instance ID").

The registry is the ground truth that maps the language instruction
``"Inspect Pipe 0-1."`` to exactly one USD prim. Per spec section 5, an
instance must be distinguishable by at least one of: a visible label plate, a
unique geometry/material configuration, a stable parent-machine topology, or
the registry entry itself. Every registered instance therefore carries an
``evidence`` list, and the task manager rejects targets without evidence
(spec section 14.9: "The target instance must be identifiable from available
evidence; otherwise the episode is invalid").
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from navbench.domain.types import AABB, Pose

INSTRUCTION_TEMPLATES: tuple[str, ...] = (
    "Inspect {object}.",
    "Check {object}.",
    "Go to {object} and inspect it.",
    "Navigate to {object} and report its condition.",
    "Perform a visual inspection of {object}.",
)
"""Instruction paraphrases; the index is recorded as ``instruction_paraphrase_id``."""


class EvidenceKind:
    """Allowed kinds of instance-identifying evidence (spec section 5)."""

    LABEL_PLATE = "label_plate"
    UNIQUE_GEOMETRY = "unique_geometry"
    UNIQUE_MATERIAL = "unique_material"
    PARENT_TOPOLOGY = "parent_topology"
    REGISTRY = "semantic_registry"


@dataclass(frozen=True)
class Defect:
    """A ground-truth defect attached to an instance (spec section 8.9)."""

    defect_instance_id: str
    defect_type: str
    severity: float
    pose_world: Pose
    start_time_ns: int | None = None
    end_time_ns: int | None = None
    surface: str = "surface"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable mapping."""
        return {
            "defect_instance_id": self.defect_instance_id,
            "defect_type": self.defect_type,
            "defect_severity": float(self.severity),
            "defect_pose_world": [float(v) for v in self.pose_world.as_array()],
            "defect_start_time_ns": self.start_time_ns,
            "defect_end_time_ns": self.end_time_ns,
            "surface": self.surface,
        }


@dataclass
class SemanticInstance:
    """One registered semantic object in the scene.

    Attributes:
        instance_uid: Dense integer id used in instance-segmentation images.
        instance_id: Human/task-level id such as ``"0-1"`` for Pipe 0-1.
        semantic_class: Class name such as ``"pipe"``, ``"valve"``, ``"crate"``.
        instance_name: Stable unique name such as ``"Pipe_0_1"``.
        prim_path: Exact USD prim path.
        pose: Ground-truth world pose.
        bbox: World-axis-aligned bounding box.
        parent_machine: Owning machine name, used for topology evidence.
        inspectable: Whether the instance may be an inspection target.
        evidence: Identifying evidence kinds (spec section 5).
        defects: Ground-truth defects attached to this instance.
        required_components: Surfaces that must be captured to complete an
            inspection of this instance.
    """

    instance_uid: int
    instance_id: str
    semantic_class: str
    instance_name: str
    prim_path: str
    pose: Pose
    bbox: AABB
    parent_machine: str = ""
    inspectable: bool = False
    evidence: tuple[str, ...] = ()
    defects: tuple[Defect, ...] = ()
    required_components: tuple[str, ...] = ()
    material: str = "steel"
    radius_m: float = 0.0
    height_m: float = 0.0

    @property
    def semantic_uid(self) -> int:
        """Class id used in semantic-segmentation images."""
        return SEMANTIC_CLASS_IDS.get(self.semantic_class, 0)

    @property
    def display_name(self) -> str:
        """Human-readable name used in instructions, e.g. ``"Pipe 0-1"``."""
        return humanize_instance(self.semantic_class, self.instance_id)

    @property
    def condition_label(self) -> str:
        """``"abnormal"`` when the instance carries defects, else ``"normal"``."""
        return "abnormal" if self.defects else "normal"

    @property
    def defect_type(self) -> str:
        """Primary defect type, or ``"none"`` (spec section 8.9)."""
        if not self.defects:
            return "none"
        return max(self.defects, key=lambda d: d.severity).defect_type

    def to_registry_dict(self) -> dict[str, Any]:
        """Return the registry record written to ``scene_registry.jsonl``."""
        return {
            "instance_uid": self.instance_uid,
            "instance_id": self.instance_id,
            "instance_name": self.instance_name,
            "semantic_class": self.semantic_class,
            "semantic_uid": self.semantic_uid,
            "prim_path": self.prim_path,
            "parent_machine": self.parent_machine,
            "pose_world": [float(v) for v in self.pose.as_array()],
            "bbox_world": self.bbox.as_list(),
            "inspectable": self.inspectable,
            "evidence": list(self.evidence),
            "material": self.material,
            "required_components": list(self.required_components),
            "defects": [d.to_dict() for d in self.defects],
        }


SEMANTIC_CLASS_IDS: dict[str, int] = {
    "background": 0,
    "floor": 1,
    "wall": 2,
    "machine": 3,
    "pipe": 4,
    "valve": 5,
    "pump": 6,
    "tank": 7,
    "crate": 8,
    "barrier": 9,
    "forklift": 10,
    "shelf": 11,
    "label_plate": 12,
    "robot": 13,
}
"""Stable class ids used in ``semantic_segmentation`` (uint16, spec 8.3)."""


def humanize_instance(semantic_class: str, instance_id: str) -> str:
    """Return the human-readable object name used in instructions.

    ``("pipe", "0-1")`` becomes ``"Pipe 0-1"`` (spec section 7.2 / 5).
    """
    return f"{semantic_class.replace('_', ' ').title()} {instance_id}"


_INSTRUCTION_RE = re.compile(
    r"(?P<verb>inspect|check|navigate to|go to|perform a visual inspection of)\s+"
    r"(?P<object>[A-Za-z_ ]+?)\s*(?P<id>[0-9]+(?:-[0-9]+)*)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ParsedInstruction:
    """Symbolic form of an instruction (spec section 5 table)."""

    instruction_text: str
    task_verb: str
    target_class: str
    target_instance_id: str

    def to_dict(self) -> dict[str, str]:
        """Return a JSON-serializable mapping."""
        return {
            "instruction_text": self.instruction_text,
            "task_verb": self.task_verb,
            "target_class": self.target_class,
            "target_instance_id": self.target_instance_id,
        }


def parse_instruction(text: str) -> ParsedInstruction:
    """Parse ``"Inspect Pipe 0-1."`` into its symbolic fields.

    Raises:
        ValueError: if the instruction does not name a class and an instance id.
    """
    match = _INSTRUCTION_RE.search(text)
    if match is None:
        raise ValueError(f"cannot parse instruction: {text!r}")
    verb = match.group("verb").lower()
    verb = "inspect" if "inspect" in verb or verb == "check" else "navigate"
    target_class = match.group("object").strip().lower().replace(" ", "_")
    return ParsedInstruction(
        instruction_text=text,
        task_verb=verb,
        target_class=target_class,
        target_instance_id=match.group("id"),
    )


def render_instruction(
    semantic_class: str, instance_id: str, paraphrase_id: int = 0
) -> tuple[str, int]:
    """Render an instruction from a template.

    Returns:
        The instruction text and the ``instruction_paraphrase_id`` actually used.
    """
    index = int(paraphrase_id) % len(INSTRUCTION_TEMPLATES)
    text = INSTRUCTION_TEMPLATES[index].format(
        object=humanize_instance(semantic_class, instance_id)
    )
    return text, index


class InstanceRegistry:
    """Registry of every semantic instance in a scene.

    Provides the resolution step required by spec section 6 Phase D step 3
    ("resolve the target to instance ``0-1``").
    """

    def __init__(self, instances: Iterable[SemanticInstance] = ()) -> None:
        self._by_uid: dict[int, SemanticInstance] = {}
        self._by_name: dict[str, SemanticInstance] = {}
        for instance in instances:
            self.add(instance)

    def add(self, instance: SemanticInstance) -> SemanticInstance:
        """Register an instance.

        Raises:
            ValueError: on duplicate ``instance_uid`` or ``instance_name``.
        """
        if instance.instance_uid in self._by_uid:
            raise ValueError(f"duplicate instance_uid {instance.instance_uid}")
        if instance.instance_name in self._by_name:
            raise ValueError(f"duplicate instance_name {instance.instance_name!r}")
        self._by_uid[instance.instance_uid] = instance
        self._by_name[instance.instance_name] = instance
        return instance

    def __len__(self) -> int:
        return len(self._by_uid)

    def __iter__(self):
        return iter(self._by_uid.values())

    @property
    def instances(self) -> list[SemanticInstance]:
        """All registered instances, ordered by ``instance_uid``."""
        return [self._by_uid[uid] for uid in sorted(self._by_uid)]

    def by_uid(self, instance_uid: int) -> SemanticInstance:
        """Return the instance with the given dense uid."""
        return self._by_uid[instance_uid]

    def by_name(self, instance_name: str) -> SemanticInstance:
        """Return the instance with the given stable name."""
        return self._by_name[instance_name]

    def by_class(self, semantic_class: str) -> list[SemanticInstance]:
        """Return all instances of a semantic class, ordered by uid."""
        return [i for i in self.instances if i.semantic_class == semantic_class]

    def inspectable(self) -> list[SemanticInstance]:
        """Return all instances that may serve as inspection targets."""
        return [i for i in self.instances if i.inspectable]

    def resolve(self, semantic_class: str, instance_id: str) -> SemanticInstance:
        """Resolve ``(class, instance_id)`` to exactly one instance.

        Raises:
            LookupError: if no instance matches, or if the match is ambiguous
                (which would make the episode invalid under spec section 14.9).
        """
        matches = [
            i
            for i in self.instances
            if i.semantic_class == semantic_class and i.instance_id == instance_id
        ]
        if not matches:
            raise LookupError(f"no instance {semantic_class!r} {instance_id!r} in registry")
        if len(matches) > 1:
            raise LookupError(
                f"ambiguous target {semantic_class!r} {instance_id!r}: "
                f"{[m.instance_name for m in matches]}"
            )
        target = matches[0]
        if not target.evidence:
            raise LookupError(
                f"target {target.instance_name!r} has no identifying evidence; "
                "episode would be ill-posed (spec section 5)"
            )
        return target

    def resolve_instruction(self, text: str) -> tuple[ParsedInstruction, SemanticInstance]:
        """Parse an instruction and resolve it to a registered instance."""
        parsed = parse_instruction(text)
        return parsed, self.resolve(parsed.target_class, parsed.target_instance_id)

    def distractors(self, target: SemanticInstance) -> list[SemanticInstance]:
        """Return same-class instances that are not the target (spec 5)."""
        return [
            i
            for i in self.instances
            if i.semantic_class == target.semantic_class and i.instance_uid != target.instance_uid
        ]

    def to_records(self) -> list[dict[str, Any]]:
        """Return all registry records for ``scene_registry.jsonl``."""
        return [i.to_registry_dict() for i in self.instances]


def class_id_table() -> dict[str, int]:
    """Return a copy of the semantic class-id table for ``label_definitions.json``."""
    return dict(SEMANTIC_CLASS_IDS)


def uid_lookup_array(registry: InstanceRegistry) -> Sequence[int]:
    """Return semantic uids indexed by instance uid, for fast image mapping."""
    if len(registry) == 0:
        return []
    size = max(i.instance_uid for i in registry) + 1
    table = [0] * size
    for instance in registry:
        table[instance.instance_uid] = instance.semantic_uid
    return table


@dataclass
class SceneGraph:
    """Parent/child topology of the scene, exported for CLIP-Nav style models."""

    nodes: list[dict[str, Any]] = field(default_factory=list)
    edges: list[dict[str, Any]] = field(default_factory=list)

    @staticmethod
    def from_registry(registry: InstanceRegistry, scene_id: str) -> SceneGraph:
        """Build a scene graph from the instance registry."""
        graph = SceneGraph()
        graph.nodes.append({"id": scene_id, "type": "scene", "name": scene_id})
        machines = sorted({i.parent_machine for i in registry if i.parent_machine})
        for machine in machines:
            graph.nodes.append({"id": machine, "type": "machine", "name": machine})
            graph.edges.append({"source": scene_id, "target": machine, "relation": "contains"})
        for instance in registry:
            graph.nodes.append(
                {
                    "id": instance.instance_name,
                    "type": instance.semantic_class,
                    "name": instance.display_name,
                    "instance_uid": instance.instance_uid,
                    "instance_id": instance.instance_id,
                    "prim_path": instance.prim_path,
                    "position_world": [float(v) for v in instance.pose.position],
                    "inspectable": instance.inspectable,
                }
            )
            parent = instance.parent_machine or scene_id
            graph.edges.append(
                {"source": parent, "target": instance.instance_name, "relation": "contains"}
            )
        return graph

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable mapping."""
        return {"nodes": self.nodes, "edges": self.edges}