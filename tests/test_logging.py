import logging
from pathlib import Path

from rag.logging import setup_logging


def test_setup_logging_idempotent_and_no_duplicate_handlers(tmp_path: Path, monkeypatch):
    log_file = tmp_path / "test.log"
    monkeypatch.setenv("LOG_FILE", str(log_file))
    monkeypatch.setenv("LOG_LEVEL", "INFO")

    setup_logging()
    setup_logging()

    root = logging.getLogger()
    stream_handlers = [
        h
        for h in root.handlers
        if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
    ]
    file_handlers = [
        h
        for h in root.handlers
        if isinstance(h, logging.FileHandler) and Path(getattr(h, "baseFilename", "")) == log_file
    ]
    # хотя бы один stream-обработчик и ровно один файловый (наш)
    assert len(stream_handlers) >= 1
    assert len(file_handlers) == 1

    logging.getLogger("rag.test").info("hello to file")
    assert log_file.exists()
    content = log_file.read_text(encoding="utf-8")
    assert "hello to file" in content
