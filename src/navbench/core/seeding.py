"""Reproducible seeding utilities.

Every benchmark run has one master seed; per-episode seeds are derived
deterministically so results are reproducible regardless of execution order.
"""

from __future__ import annotations

import hashlib
import random


def derive_seed(master_seed: int, *components: str | int) -> int:
    """Derive a stable sub-seed from a master seed and identifying components.

    Uses SHA-256 (not Python's ``hash``, which is salted per process) so the
    derivation is stable across runs, machines, and Python versions.
    """
    payload = ":".join([str(master_seed), *map(str, components)])
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**31 - 1)


def seed_everything(seed: int) -> None:
    """Seed all random sources available in the current environment.

    Core only depends on the stdlib; numpy/torch are seeded opportunistically
    if installed (plugins that use them get reproducibility for free).
    """
    random.seed(seed)
    try:  # pragma: no cover - optional dependency
        import numpy as np

        np.random.seed(seed % (2**32 - 1))
    except ImportError:
        pass
    try:  # pragma: no cover - optional dependency
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass