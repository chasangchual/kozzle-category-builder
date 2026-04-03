"""Utilities and helper functions."""

import logging
from pathlib import Path
from typing import Any


def setup_logging(level: int = logging.INFO) -> None:
    """Configure logging for the application.

    Args:
        level: Logging level (default: INFO).
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance.

    Args:
        name: Logger name.

    Returns:
        Configured logger instance.
    """
    return logging.getLogger(name)


def ensure_directory(path: Path | str) -> Path:
    """Ensure a directory exists, creating it if necessary.

    Args:
        path: Directory path.

    Returns:
        Path object for the directory.
    """
    dir_path = Path(path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def batch_list(items: list[Any], batch_size: int) -> list[list[Any]]:
    """Split a list into batches.

    Args:
        items: List of items to batch.
        batch_size: Size of each batch.

    Returns:
        List of batches.
    """
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]
