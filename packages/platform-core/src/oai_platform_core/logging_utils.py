"""
oai_platform_core.logging_utils — shared logging helpers.

Provides a single :func:`get_logger` factory that all OAI packages can
import instead of duplicating the rotating-file + console setup.

Usage
-----
    from oai_platform_core.logging_utils import get_logger

    logger = get_logger()                   # name="app", writes to <tmp>/app.log
    logger = get_logger("my-service")       # name="my-service", writes to <tmp>/my-service.log
    logger = get_logger("svc", level=logging.DEBUG, log_file="/var/log/svc.log")
"""

from __future__ import annotations

import logging
import os
import tempfile
from logging.handlers import RotatingFileHandler
from typing import Optional

__all__ = ["get_logger"]


def get_logger(
    name: str = "app",
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    max_bytes: int = 1024 * 1024,  # 1 MB per file
    backup_count: int = 5,
) -> logging.Logger:
    """Return a named :class:`logging.Logger` with rotating-file and console handlers.

    Re-entrant: if the named logger already has handlers attached it is
    returned as-is, so calling ``get_logger()`` multiple times with the
    same *name* never multiplies handlers.

    Args:
        name:         Logger name (also used to derive the default log
                      file name when *log_file* is ``None``).
        level:        Root level for the logger (default ``INFO``).
        log_file:     Explicit path for the rotating log file.  When
                      omitted the file is written to
                      ``<system-tmp-dir>/<name>.log``.
        max_bytes:    Maximum size of each log file before rotation
                      (default 1 MiB).
        backup_count: Number of rotated backup files to keep (default 5).

    Returns:
        Configured :class:`logging.Logger` instance.
    """
    logger = logging.getLogger(name)

    # Idempotent — avoid adding duplicate handlers on repeated calls.
    if logger.handlers:
        return logger

    logger.setLevel(level)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Rotating file handler
    if log_file is None:
        log_file = os.path.join(tempfile.gettempdir(), f"{name}.log")

    fh = RotatingFileHandler(log_file, maxBytes=max_bytes, backupCount=backup_count)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # Console (stream) handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger
