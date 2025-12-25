from __future__ import annotations

import logging

from rag.logging import setup_logging

setup_logging()
logger = logging.getLogger(__name__)
logger.info("Старт Streamlit UI")
