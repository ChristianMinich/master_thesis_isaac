"""navbench: simulation-based benchmark for comparing robot-navigation approaches.

Ports-and-adapters architecture:

- ``navbench.core``    -- interfaces (ports) + benchmark orchestration. No Isaac imports.
- ``navbench.sim``     -- simulator adapters (mock, Isaac Sim, Isaac Lab).
- ``navbench.plugins`` -- concrete approach plugins (SLAM, RL, VLA, WM, hybrid).
"""

from navbench.core.plugin import ApproachPlugin, PluginRegistry

__all__ = ["ApproachPlugin", "PluginRegistry"]
__version__ = "0.1.0"