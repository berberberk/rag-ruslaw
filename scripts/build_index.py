from __future__ import annotations

import logging

from rag.logging import setup_logging

logger = logging.getLogger(__name__)


def main() -> None:
    """
    Точка входа для сборки индексов (пока заглушка).
    """
    setup_logging()
    logger.info("Старт сборки индексов (реализация будет добавлена в Phase 3+)")


if __name__ == "__main__":
    main()
