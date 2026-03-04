"""Logging utilities."""

import logging
import sys
from pathlib import Path
from typing import Optional
from contextlib import contextmanager

from ..core.types import LogLevel


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def setup_logger(
    name: str,
    level: LogLevel = LogLevel.INFO,
    log_file: Optional[Path] = None,
) -> logging.Logger:
    logger = logging.getLogger(name)
    level_map = {
        LogLevel.DEBUG: logging.DEBUG,
        LogLevel.INFO: logging.INFO,
        LogLevel.WARNING: logging.WARNING,
        LogLevel.ERROR: logging.ERROR,
        LogLevel.CRITICAL: logging.CRITICAL,
    }
    logger.setLevel(level_map.get(level, logging.INFO))

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
        logger.addHandler(handler)

    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
        logger.addHandler(fh)

    return logger


@contextmanager
def LogProgress(logger: logging.Logger, message: str):
    logger.info(f"Starting: {message}")
    try:
        yield
    finally:
        logger.info(f"Done: {message}")


def log_model_info(logger: logging.Logger, model) -> None:
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {total:,}")