from __future__ import annotations

import gzip
import json
import logging
from pathlib import Path
from typing import Any

from rag.ingest.schema import Document
from rag.logging import setup_logging

logger = logging.getLogger(__name__)


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """
    Загрузка локального JSONL-файла с RusLawOD-совместимыми записями.

    Parameters
    ----------
    path : str | Path
        Путь до JSONL файла

    Returns
    -------
    List[Dict[str, Any]]
        Список записей, по одной на строку
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"JSONL not found: {p}")

    rows: list[dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_documents_from_slice(path: Path) -> list[Document]:
    """
    Загрузка нормализованного gzip JSONL среза в список Document.

    Parameters
    ----------
    path : Path
        Путь до файла .jsonl.gz

    Returns
    -------
    list[Document]
        Список документов
    """
    setup_logging()
    if not path.exists():
        raise FileNotFoundError(f"Slice not found: {path}")

    rows_read = 0
    docs: list[Document] = []
    empty_text = 0

    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows_read += 1
            raw = json.loads(line)
            if "doc_id" in raw and "text" in raw:
                text = str(raw.get("text") or "").strip()
                if not text:
                    empty_text += 1
                    continue
                doc_id = str(raw.get("doc_id"))
                metadata = dict(raw.get("metadata") or {})
                title = str(raw.get("title") or "")
                docs.append(Document(doc_id=doc_id, title=title, text=text, metadata=metadata))
            else:
                continue

    logger.info(
        "Загрузка среза: %s, строк=%s, документов=%s, пустой текст=%s",
        path,
        rows_read,
        len(docs),
        empty_text,
    )
    return docs
