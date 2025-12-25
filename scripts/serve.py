from __future__ import annotations

import logging

from rag.logging import setup_logging

logger = logging.getLogger(__name__)


def main() -> None:
    """
    Точка входа для запуска FastAPI сервера (заглушка).
    """
    setup_logging()
    logger.info("Запуск FastAPI сервера (реализация маршрутов в rag.api)")


if __name__ == "__main__":
    main()
