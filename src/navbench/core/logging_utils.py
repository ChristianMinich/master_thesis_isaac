"""Structured logging setup.

Emits JSON lines (one event per line) so benchmark runs can be parsed,
aggregated, and compared programmatically. Standard LogRecord attributes are
excluded; anything passed via ``logger.info(msg, extra={...})`` is included.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Attributes present on every LogRecord that we do not treat as user payload.
_STANDARD_ATTRS = frozenset(
    logging.LogRecord("", 0, "", 0, "", (), None).__dict__.keys()
) | {"message", "asctime", "taskName"}


class JsonLineFormatter(logging.Formatter):
    """Formats each log record as a single JSON object."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _STANDARD_ATTRS and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(
    level: int = logging.INFO,
    log_file: Path | None = None,
    stream: bool = True,
) -> None:
    """Configure the root ``navbench`` logger with structured JSON output.

    Idempotent: existing navbench handlers are replaced, not duplicated.
    """
    logger = logging.getLogger("navbench")
    logger.setLevel(level)
    logger.handlers.clear()
    logger.propagate = False

    formatter = JsonLineFormatter()
    if stream:
        stream_handler = logging.StreamHandler(sys.stderr)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)