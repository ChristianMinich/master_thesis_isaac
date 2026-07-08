"""Dataset writers for synthetic Go2 data.

Two writers cover the formats used by the four approaches:

- ``EpisodeWriter``    — one ``.npz`` file per episode (SLAM, VLA, RL rollouts)
- ``TransitionWriter`` — chunked ``.npz`` files of flat transitions (world model)

Both write a ``manifest.json`` next to the data so downstream training code
can discover files, shapes, and generation parameters without re-reading
every array.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np


def _to_arrays(buffers: dict[str, list[np.ndarray]]) -> dict[str, np.ndarray]:
    return {key: np.stack(vals, axis=0) for key, vals in buffers.items() if vals}


class ManifestMixin:
    """Shared manifest bookkeeping."""

    output_dir: Path

    def _init_manifest(self, kind: str, metadata: dict[str, Any]) -> None:
        self._manifest: dict[str, Any] = {
            "kind": kind,
            "created_unix_time": time.time(),
            "metadata": metadata,
            "files": [],
        }

    def _register_file(self, path: Path, arrays: dict[str, np.ndarray], **extra: Any) -> None:
        entry: dict[str, Any] = {
            "file": path.name,
            "arrays": {key: list(arr.shape) for key, arr in arrays.items()},
        }
        entry.update(extra)
        self._manifest["files"].append(entry)

    def finalize(self) -> Path:
        """Write ``manifest.json`` and return its path."""
        manifest_path = self.output_dir / "manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as fh:
            json.dump(self._manifest, fh, indent=2)
        return manifest_path


class EpisodeWriter(ManifestMixin):
    """Saves one compressed ``.npz`` file per episode.

    Arrays are stacked to shape ``(T, num_envs, ...)``. Keys are free-form so
    each approach can store what it needs (rgb, depth, language, poses, ...).
    """

    def __init__(self, output_dir: str | Path, metadata: dict[str, Any] | None = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._episode_index = 0
        self._init_manifest("episodes", metadata or {})

    def write_episode(self, buffers: dict[str, list[np.ndarray]], **extra: Any) -> Path:
        arrays = _to_arrays(buffers)
        path = self.output_dir / f"episode_{self._episode_index:05d}.npz"
        np.savez_compressed(path, **arrays)
        self._register_file(path, arrays, episode_index=self._episode_index, **extra)
        self._episode_index += 1
        return path


class TransitionWriter(ManifestMixin):
    """Buffers flat transitions and flushes chunked ``.npz`` files.

    Suitable for world-model / off-policy training data where episode
    boundaries are encoded via a ``done`` flag rather than file boundaries.
    """

    def __init__(
        self,
        output_dir: str | Path,
        chunk_size: int = 1000,
        metadata: dict[str, Any] | None = None,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.chunk_size = chunk_size
        self._buffers: dict[str, list[np.ndarray]] = {}
        self._count = 0
        self._chunk_index = 0
        self._init_manifest("transitions", metadata or {})

    def add(self, transition: dict[str, np.ndarray]) -> None:
        """Add one batched transition; each value has shape ``(num_envs, ...)``."""
        for key, value in transition.items():
            self._buffers.setdefault(key, []).append(np.asarray(value))
        first_key = next(iter(transition))
        self._count += int(np.asarray(transition[first_key]).shape[0])
        if self._count >= self.chunk_size:
            self.flush()

    def flush(self) -> Path | None:
        if not self._buffers or not next(iter(self._buffers.values())):
            return None
        arrays = _to_arrays(self._buffers)
        path = self.output_dir / f"transitions_{self._chunk_index:05d}.npz"
        np.savez_compressed(path, **arrays)
        self._register_file(path, arrays, chunk_index=self._chunk_index)
        self._buffers = {}
        self._count = 0
        self._chunk_index += 1
        return path

    def finalize(self) -> Path:
        self.flush()
        return super().finalize()