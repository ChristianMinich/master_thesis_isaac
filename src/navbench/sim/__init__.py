"""Simulator adapters implementing the ``SimulatorAdapter`` port.

- ``MockSimulatorAdapter``: deterministic kinematic point-robot world.
  Used for tests, CI, and development on machines without Isaac.
- ``IsaacSimAdapter``: skeleton adapter for NVIDIA Isaac Sim (TODO wiring).
- ``IsaacLabAdapter``: skeleton adapter for NVIDIA Isaac Lab (TODO wiring).

``make_simulator`` is the config-driven factory injected into the
``BenchmarkRunner`` by entry-point scripts. Isaac modules are imported lazily
inside the adapters so that this package imports cleanly without Isaac.
"""

from __future__ import annotations

from typing import Any, Mapping

from navbench.core.simulator import SimulatorAdapter
from navbench.sim.mock import MockSimulatorAdapter


def make_simulator(config: Mapping[str, Any]) -> SimulatorAdapter:
    """Build a SimulatorAdapter from a config's ``simulator`` section.

    ``config["type"]`` selects the backend: mock | isaac_sim | isaac_lab.
    """
    sim_type = str(config.get("type", "mock"))
    if sim_type == "mock":
        return MockSimulatorAdapter(
            step_size=float(config.get("step_size", 0.1)),
            noise_std=float(config.get("noise_std", 0.0)),
        )
    if sim_type == "isaac_sim":
        from navbench.sim.isaac_sim import IsaacSimAdapter

        return IsaacSimAdapter(config)
    if sim_type == "isaac_lab":
        from navbench.sim.isaac_lab import IsaacLabAdapter

        return IsaacLabAdapter(config)
    raise ValueError(f"unknown simulator type: {sim_type!r} (mock | isaac_sim | isaac_lab)")


__all__ = ["MockSimulatorAdapter", "make_simulator"]