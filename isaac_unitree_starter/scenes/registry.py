"""Scene registry: name -> Scene class, so a scene can be picked by a plain string
(e.g. a --scene CLI arg) instead of importing a specific class.
"""

from __future__ import annotations

from core.scene import Scene
from scenes.empty_scene import EmptyScene
from scenes.room_scene import RoomScene

SCENE_REGISTRY: dict[str, type[Scene]] = {
    RoomScene.name: RoomScene,
    EmptyScene.name: EmptyScene,
}


def get_scene(name: str) -> Scene:
    """Instantiate the registered Scene for the given name.

    Args:
        name: Registry key (see SCENE_REGISTRY, e.g. "room" or "empty").

    Raises:
        ValueError: If no scene is registered under that name.
    """
    scene_cls = SCENE_REGISTRY.get(name)
    if scene_cls is None:
        available = ", ".join(sorted(SCENE_REGISTRY))
        raise ValueError(f"Unknown scene {name!r}. Available scenes: {available}")
    return scene_cls()
