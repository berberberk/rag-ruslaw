from __future__ import annotations

import logging

from rag.logging import setup_logging

logger = logging.getLogger(__name__)


def main() -> None:
    """
    Точка входа для пайплайна ingest (пока заглушка).
    """
    setup_logging()
    logger.info("Старт ingest-пайплайна (реализация будет добавлена в Phase 2)")


if __name__ == "__main__":
    main()
