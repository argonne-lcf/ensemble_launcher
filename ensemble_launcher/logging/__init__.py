"""Centralised logging helpers for ensemble_launcher."""

import logging
import os
from typing import Optional


def setup_logger(
    name: str,
    node_id: Optional[str] = None,
    log_dir: Optional[str] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """Create and configure a logger for an orchestrator node.

    Args:
        name:    Logger name (typically ``__name__`` of the calling module).
        node_id: Optional node identifier appended to the logger name.
                 When provided the logger is named ``{name}.{node_id}``.
        log_dir: If provided, a ``FileHandler`` writing to
                 ``{log_dir}/{node_id}.log`` (or ``{log_dir}/{name}.log``
                 when *node_id* is ``None``) is attached.  The directory is
                 created if it does not exist.  When ``None`` the logger uses
                 whatever root handlers are already configured.
        level:   Logging level (default ``logging.INFO``).

    Returns:
        A configured :class:`logging.Logger` instance.
    """
    logger_name = f"{name}.{node_id}" if node_id else name
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)

    if log_dir is not None:
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
