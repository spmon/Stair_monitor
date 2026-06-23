from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger

_LOGGING_CONFIGURED = False


def _is_violation_record(record: dict[str, object]) -> bool:
    extra = record.get("extra")
    if not isinstance(extra, dict):
        return False
    return extra.get("event") == "violation"


def setup_app_logging(log_dir: str = "logs", level: str = "INFO") -> None:
    global _LOGGING_CONFIGURED
    if _LOGGING_CONFIGURED:
        return

    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    logger.remove()
    logger.add(
        sys.stderr,
        level=level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
    )
    logger.add(
        log_path / "runtime_{time:YYYYMMDD_HHmmss}.log",
        level=level,
        format=(
            "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | "
            "{name}:{function}:{line} | {message}"
        ),
    )
    logger.add(
        log_path / "violations_{time:YYYYMMDD_HHmmss}.log",
        level="WARNING",
        format=(
            "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | "
            "{name}:{function}:{line} | {message}"
        ),
        filter=_is_violation_record,
    )
    _LOGGING_CONFIGURED = True
