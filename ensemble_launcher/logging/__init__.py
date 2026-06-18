"""Centralised logging helpers for ensemble_launcher."""

import logging
import os
from typing import Optional


def get_log_dir(subdir: Optional[str] = None) -> str:
    """Return the resolved log directory.

    Checks ``EL_LOGDIR`` env var first, then defaults to ``<cwd>/logs``.
    """
    base = os.environ.get("EL_LOGDIR", os.path.join(os.getcwd(), "logs"))
    return os.path.join(base, subdir) if subdir else base


def setup_logger(
    name: str,
    node_id: Optional[str] = None,
    level: int = logging.INFO,
    log_to_file: bool = True,
    subdir: Optional[str] = None,
) -> logging.Logger:
    """Create and configure a logger for an orchestrator node.

    Args:
        name:        Logger name (typically ``__name__`` of the calling module).
        node_id:     Optional node identifier appended to the logger name.
                     When provided the logger is named ``{name}.{node_id}``.
        level:       Logging level (default ``logging.INFO``).
        log_to_file: When ``True`` (default), a ``FileHandler`` is attached
                     using the directory resolved by :func:`get_log_dir`.
                     When ``False`` a ``NullHandler`` is used instead.
        subdir:      Optional subdirectory under the log root
                     (e.g. ``"actors"``, ``"connections"``).

    Returns:
        A configured :class:`logging.Logger` instance.
    """
    logger_name = f"{name}.{node_id}" if node_id else name
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)

    if log_to_file:
        log_dir = get_log_dir(subdir)
        os.makedirs(log_dir, exist_ok=True)
        file_stem = node_id if node_id else name.split(".")[-1]
        # Replace characters that are invalid in file names on some systems
        safe_stem = file_stem.replace(":", "-")
        log_file = os.path.join(log_dir, f"{safe_stem}.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(file_handler)
        # Don't propagate to root — output goes to file only
        logger.propagate = False
    else:
        # No file handler: suppress all output (library best practice)
        logger.addHandler(logging.NullHandler())

    return logger
