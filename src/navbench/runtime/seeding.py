"""Deterministic seed derivation.

Specification reference: section 6 Phase B ("The randomization seed must
reproduce the complete episode") and section 8.1 (``scene_seed``).

Design rule: one integer ``scene_seed`` per episode is the *only* entropy
source. Every consumer (scene layout, lighting, materials, clutter, defects,
sensor noise, instruction paraphrase, expert tie-breaking) receives its own
sub-seed derived from that master seed by hashing a stable namespace string.

Consequences:

* Two consumers never share a random stream, so adding a new randomization
  domain cannot shift the values drawn by existing ones.
* The derivation is pure SHA-256 arithmetic, so it is stable across Python
  versions, platforms, and process restarts (unlike :func:`hash`).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np

_MASK64 = (1 << 64) - 1


def derive_seed(master_seed: int, *components: object) -> int:
    """Derive a stable 64-bit sub-seed from a master seed and a namespace path.

    Args:
        master_seed: The episode's ``scene_seed``.
        *components: Namespace components, e.g. ``"lighting"`` or
            ``("sensor_noise", "lidar")``. Each is converted with :func:`str`.

    Returns:
        A deterministic seed in ``[0, 2**64)``.

    Example:
        >>> derive_seed(123, "lighting") == derive_seed(123, "lighting")
        True
        >>> derive_seed(123, "lighting") != derive_seed(123, "materials")
        True
    """
    payload = "|".join([str(int(master_seed)), *(str(c) for c in components)])
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & _MASK64


def derive_episode_seed(master_seed: int, scene_id: str, episode_index: int) -> int:
    """Derive the per-episode ``scene_seed`` of a generation run.

    A run has one ``master_seed``; each episode of each scene gets a unique,
    reproducible ``scene_seed`` that alone regenerates the whole episode.
    """
    return derive_seed(master_seed, "episode", scene_id, episode_index)


@dataclass(frozen=True)
class SeedBundle:
    """Namespaced random-number-generator factory for one episode.

    All randomization in the pipeline goes through a :class:`SeedBundle` so that
    the recorded ``scene_seed`` is provably sufficient to reproduce the episode.

    Args:
        scene_seed: The episode master seed recorded in ``episode.json`` and
            ``randomization.json``.
    """

    scene_seed: int

    def seed_for(self, *components: object) -> int:
        """Return the derived integer seed for a namespace."""
        return derive_seed(self.scene_seed, *components)

    def rng(self, *components: object) -> np.random.Generator:
        """Return a fresh :class:`numpy.random.Generator` for a namespace.

        NumPy is imported lazily so that pure-metadata code paths (config
        loading, validation of an existing folder) do not require it.
        """
        import numpy as np

        return np.random.default_rng(self.seed_for(*components))

    def child(self, *components: object) -> SeedBundle:
        """Return a sub-bundle whose namespaces are all nested under a prefix."""
        return SeedBundle(self.seed_for(*components))


def seed_everything(seed: int) -> None:
    """Seed the global RNGs of Python, NumPy and torch when available.

    Per-stream RNGs from :class:`SeedBundle` are always preferred. This helper
    only pins global state for third-party code that ignores explicit RNGs
    (for example some model implementations used by the exporters' consumers).
    """
    import random

    random.seed(seed)

    try:
        import numpy as np
    except ImportError:  # pragma: no cover - numpy is a hard dep in practice
        pass
    else:
        np.random.seed(seed % (2**32))

    try:  # pragma: no cover - torch is optional
        import torch
    except ImportError:
        pass
    else:
        torch.manual_seed(seed % (2**63))