"""Config handling for the synthetic data generation scripts.

Every generator script defines its parameters as ``argparse`` defaults.
A YAML config file (``--config path.yaml``) can override those defaults,
and explicit CLI flags override the YAML values in turn:

    CLI flag  >  YAML config  >  built-in default
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML mapping; returns {} for an empty file."""
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file {path} must contain a YAML mapping.")
    return data


def parse_args_with_config(parser: argparse.ArgumentParser) -> argparse.Namespace:
    """Parse args in two phases so a YAML config can override defaults.

    Phase 1 only extracts ``--config``; its values become the new parser
    defaults. Phase 2 parses the full command line, so explicitly passed
    flags still win over the YAML file.
    """
    prelim, _ = parser.parse_known_args()
    config_path = getattr(prelim, "config", None)
    if config_path:
        config = load_yaml(config_path)
        known_dests = {action.dest for action in parser._actions}
        unknown = set(config) - known_dests
        if unknown:
            raise ValueError(
                f"Unknown keys in config {config_path}: {sorted(unknown)}. "
                f"Valid keys: {sorted(known_dests - {'help'})}"
            )
        parser.set_defaults(**config)
    return parser.parse_args()