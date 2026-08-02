"""Provenance capture for reproducible episodes.

Specification reference: section 8.1 (``generator_commit``, ``isaac_sim_version``,
``isaac_lab_version``, ``robot_asset_version``, ``policy_name``,
``policy_checkpoint``) and section 18 ("Implementation-freeze references").

The pipeline never guesses a version. Values that cannot be determined are
recorded as ``"unknown"`` so that a missing freeze reference is visible in the
dataset instead of silently absent.
"""

from __future__ import annotations

import hashlib
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

UNKNOWN = "unknown"


def git_commit(repo_root: Path | None = None, short: bool = False) -> str:
    """Return the current git commit hash of the generating repository.

    Returns ``"unknown"`` when git is unavailable or the directory is not a
    repository, and appends ``"-dirty"`` when the working tree has
    uncommitted changes (an episode generated from a dirty tree is not
    exactly reproducible from the commit alone).
    """
    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[3]
    args = ["git", "rev-parse", "--short", "HEAD"] if short else ["git", "rev-parse", "HEAD"]
    try:
        commit = subprocess.run(
            args,
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return UNKNOWN
    if not commit:
        return UNKNOWN
    return f"{commit}-dirty" if dirty else commit


def isaac_sim_version() -> str:
    """Return the installed Isaac Sim version, or ``"unknown"``.

    Isaac Sim exposes its version through ``isaacsim.core.version.get_version``
    in recent releases; older layouts expose ``omni.isaac.version``. Both are
    probed without raising, because the pipeline must also run on machines
    without Isaac installed (synthetic backend).
    """
    for module_name in ("isaacsim.core.version", "omni.isaac.version"):
        try:  # pragma: no cover - requires Isaac Sim
            module = __import__(module_name, fromlist=["get_version"])
            version = module.get_version()
        except Exception:
            continue
        if isinstance(version, (tuple, list)) and version:
            return str(version[0])
        if version:
            return str(version)
    return UNKNOWN


def isaac_lab_version() -> str:
    """Return the installed Isaac Lab version, or ``"unknown"``."""
    for module_name in ("isaaclab", "omni.isaac.lab"):
        try:  # pragma: no cover - requires Isaac Lab
            module = __import__(module_name)
        except Exception:
            continue
        version = getattr(module, "__version__", None)
        if version:
            return str(version)
    return UNKNOWN


def file_version(path: Path | str) -> str:
    """Return a content hash for an asset file, used as ``robot_asset_version``.

    A content hash is preferred over a declared version string because USD
    assets are frequently edited in place during a thesis project.
    """
    p = Path(path)
    if not p.is_file():
        return UNKNOWN
    digest = hashlib.sha256()
    with p.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()[:16]}"


@dataclass
class Provenance:
    """Everything needed to trace an episode back to the code that made it."""

    dataset_version: str
    generator_commit: str
    isaac_sim_version: str
    isaac_lab_version: str
    robot_asset_version: str
    robot_asset_path: str
    simulator_backend: str
    policy_name: str
    policy_checkpoint: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable mapping."""
        return asdict(self)


def collect_provenance(
    *,
    dataset_version: str,
    robot_asset_path: Path | str,
    simulator_backend: str,
    policy_name: str,
    policy_checkpoint: str | None = None,
    repo_root: Path | None = None,
    extra: dict[str, Any] | None = None,
) -> Provenance:
    """Collect all provenance fields required by spec section 8.1."""
    return Provenance(
        dataset_version=dataset_version,
        generator_commit=git_commit(repo_root),
        isaac_sim_version=isaac_sim_version(),
        isaac_lab_version=isaac_lab_version(),
        robot_asset_version=file_version(robot_asset_path),
        robot_asset_path=str(robot_asset_path),
        simulator_backend=simulator_backend,
        policy_name=policy_name,
        policy_checkpoint=policy_checkpoint,
        extra=dict(extra or {}),
    )