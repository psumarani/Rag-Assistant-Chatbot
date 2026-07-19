"""Centralized logging configuration for the RAG Assistant.

Provides a single `get_logger()` factory so every module logs to the
same rotating file (under `logs/`) and the console, with a consistent
format. Modules should never call `logging.basicConfig` themselves.
"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler

from config.constants import LOG_DATE_FORMAT, LOG_FILE_NAME, LOG_FORMAT, LOGS_DIR

_MAX_BYTES = 5 * 1024 * 1024  # 5 MB per log file
_BACKUP_COUNT = 3

_configured = False


def _configure_root_logger() -> None:
    """Attach a rotating file handler and a console handler exactly once."""
    global _configured
    if _configured:
        return

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOGS_DIR / LOG_FILE_NAME

    formatter = logging.Formatter(fmt=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)

    file_handler = RotatingFileHandler(
        filename=log_path,
        maxBytes=_MAX_BYTES,
        backupCount=_BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)

    root_logger = logging.getLogger("rag_assistant")
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    root_logger.propagate = False

    _configured = True


def get_logger(module_name: str) -> logging.Logger:
    """Return a namespaced logger that writes to the shared log file.

    Args:
        module_name: Typically `__name__` of the calling module.

    Returns:
        A configured `logging.Logger` instance scoped under
        `rag_assistant.<module_name>`.
    """
    _configure_root_logger()
    return logging.getLogger(f"rag_assistant.{module_name}")
