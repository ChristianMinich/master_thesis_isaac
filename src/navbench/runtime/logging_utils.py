"""Structured JSON-lines logging for the generation pipeline.

Specification reference: section 13 ("Required dataset quality checks") — a
generation run must be able to report per-episode outcomes in a machine
parseable form, so log records are emitted as one JSON object per line.

Usage::

    configure_logging(level="INFO", log_file=Path("runs/gen/run.jsonl"))
    log = get_logger(__name__)
    log.info("episode.committed", extra={"event": {"episode_id": "episode_000000"}})
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

LOGGER_NAME = "navbench"


class JsonLineFormatter(logging.Formatter):
    """Format each record as a single-line JSON object.

    Anything passed through ``extra={"event": {...}}`` is merged into the
    record so downstream tooling can filter on structured fields instead of
    parsing free text.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, datefmt="%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        event = getattr(record, "event", None)
        if isinstance(event, dict):
            payload.update(event)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, sort_keys=True, default=str)


def configure_logging(level: str = "INFO", log_file: Path | str | None = None) -> logging.Logger:
    """Configure the ``navbench`` logger with JSON-lines output.

    Args:
        level: Logging level name.
        log_file: Optional path receiving a copy of every record. Parent
            directories are created.

    Returns:
        The configured root logger of the package.
    """
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.handlers.clear()
    logger.propagate = False

    stream = logging.StreamHandler(sys.stderr)
    stream.setFormatter(JsonLineFormatter())
    logger.addHandler(stream)

    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setFormatter(JsonLineFormatter())
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a child logger of the package logger."""
    if not name or name == LOGGER_NAME:
        return logging.getLogger(LOGGER_NAME)
    return logging.getLogger(f"{LOGGER_NAME}.{name.split('.')[-1]}")