"""CommandSource protocol: unifies keyboard and autonomous drivers behind one call site.

Anything that can produce a [v_x, v_y, w_z] command tensor each frame -- keyboard
input, the random-walk generator, a future scripted-path or RL-inference driver --
implements `update(dt)`. `main.py`'s loop calls it polymorphically instead of
branching on which driver is active.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class CommandSource(Protocol):
    def update(self, dt: float):
        """Return the current [v_x, v_y, w_z] command tensor for this frame."""
        ...
