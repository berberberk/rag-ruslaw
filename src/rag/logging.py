from __future__ import annotations

import logging
import os
from pathlib import Path


def setup_logging() -> None:
    """
    Настройка логирования для всего проекта.

    Parameters
    ----------
    None
        Используются переменные окружения LOG_LEVEL и LOG_FILE

    Returns
    -------
    None
        Логирование настроено; повторные вызовы безопасны
    """
    level_str = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_str, logging.INFO)

    root = logging.getLogger()
    root.setLevel(level)

    fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)

    has_stream = any(
        isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        for h in root.handlers
    )
    if not has_stream:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        root.addHandler(stream_handler)

    log_file = os.getenv("LOG_FILE")
    if log_file:
        log_path = Path(log_file)
        has_file = any(
            isinstance(h, logging.FileHandler) and Path(getattr(h, "baseFilename", "")) == log_path
            for h in root.handlers
        )
        if not has_file:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_path, encoding="utf-8")
            file_handler.setFormatter(formatter)
            root.addHandler(file_handler)
