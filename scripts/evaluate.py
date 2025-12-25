from __future__ import annotations

import logging

from rag.logging import setup_logging

logger = logging.getLogger(__name__)


def main() -> None:
    """
    Точка входа для запуска оценки (пока заглушка).
    """
    setup_logging()
    logger.info("Старт eval пайплайна (реализация будет добавлена в Phase 7)")


if __name__ == "__main__":
    main()
