"""ApproachPlugin port and PluginRegistry.

This is the extension seam of the whole benchmark:

- An ApproachPlugin is a factory bundle that knows how to build the Agent and
  Trainer for one navigation approach from its config dict.
- The PluginRegistry maps approach names -> plugins. Plugins register
  themselves at import time via the ``@register_approach`` decorator, or are
  discovered dynamically from a dotted module path given in config
  (``plugin_module``), so adding a new approach never requires editing core.
"""

from __future__ import annotations

import importlib
import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, Mapping, TypeVar

from navbench.core.agent import Agent
from navbench.core.trainer import Trainer

logger = logging.getLogger(__name__)


class ApproachPlugin(ABC):
    """Port: factory bundle for one navigation approach.

    Implementations live in ``navbench/plugins/<name>/`` (or any external
    package) and must be self-contained: core never imports concrete
    approaches directly.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique approach identifier used in configs (e.g. "rl_ppo")."""

    @abstractmethod
    def build_agent(self, config: Mapping[str, Any]) -> Agent:
        """Construct the approach's Agent from its config section."""

    @abstractmethod
    def build_trainer(self, config: Mapping[str, Any]) -> Trainer:
        """Construct the approach's Trainer (NoOpTrainer if not trainable)."""

    def describe(self) -> Mapping[str, Any]:
        """Optional metadata (paper reference, capabilities, modalities)."""
        return {}


class PluginRegistry:
    """Registry mapping approach names to ApproachPlugin instances.

    Supports two registration paths:

    1. Static: ``@register_approach`` decorator on plugin classes, applied
       when the plugin module is imported.
    2. Dynamic: ``load_from_module("my_pkg.my_plugin")`` imports a module and
       collects everything it registered — this is what lets a *new paper's
       approach* live entirely outside this repository.
    """

    _global: "PluginRegistry | None" = None

    def __init__(self) -> None:
        self._plugins: dict[str, ApproachPlugin] = {}

    # -- global default registry ------------------------------------------
    @classmethod
    def default(cls) -> "PluginRegistry":
        if cls._global is None:
            cls._global = cls()
        return cls._global

    # -- registration -------------------------------------------------------
    def register(self, plugin: ApproachPlugin) -> None:
        if plugin.name in self._plugins:
            raise ValueError(f"approach {plugin.name!r} is already registered")
        self._plugins[plugin.name] = plugin
        logger.info("plugin.registered", extra={"approach": plugin.name})

    def load_from_module(self, module_path: str) -> None:
        """Import a module so its ``@register_approach`` decorators run."""
        importlib.import_module(module_path)

    # -- lookup ---------------------------------------------------------------
    def get(self, name: str) -> ApproachPlugin:
        try:
            return self._plugins[name]
        except KeyError as exc:
            raise KeyError(
                f"unknown approach {name!r}; registered: {sorted(self._plugins)}"
            ) from exc

    def names(self) -> list[str]:
        return sorted(self._plugins)

    def __contains__(self, name: str) -> bool:
        return name in self._plugins


P = TypeVar("P", bound=ApproachPlugin)


def register_approach(
    registry: PluginRegistry | None = None,
) -> Callable[[type[P]], type[P]]:
    """Class decorator: instantiate and register a plugin class.

    Usage::

        @register_approach()
        class MyPlugin(ApproachPlugin):
            ...
    """

    def decorator(cls: type[P]) -> type[P]:
        target = registry if registry is not None else PluginRegistry.default()
        target.register(cls())
        return cls

    return decorator