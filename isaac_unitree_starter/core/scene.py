"""Scene abstraction: the swappable environment a robot is dropped into.

A Scene owns only static/environment authoring (ground + geometry) -- it knows
nothing about robots. `main.py` creates the one shared PhysicsScene prim itself
(identical for every scene) before calling `Scene.build()`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class SceneInfo:
    """Result of building a scene: what callers need to spawn a robot into it."""

    assets_root_path: str
    default_spawn_position: list[float]
    # (half_x, half_y) if this scene has a bounded floor plan a driver should stay
    # within (e.g. RandomWalkController's wall-avoidance), or None if unbounded.
    bounds_half_extent: tuple[float, float] | None = None


class Scene(ABC):
    """A swappable environment: ground + static geometry, authored onto the current stage."""

    name: str

    @abstractmethod
    def build(self) -> SceneInfo:
        """Author this scene's ground + static geometry onto the current USD stage.

        Returns:
            SceneInfo with the resolved assets root path and a suggested spawn position.
        """
        raise NotImplementedError
