from __future__ import annotations

import logging
from collections.abc import Iterable

from rag.ingest.schema import Chunk, Document
from rag.logging import setup_logging

logger = logging.getLogger(__name__)
PROGRESS_DOCS = 50


def chunk_document(doc: Document, *, chunk_size: int, overlap: int) -> list[Chunk]:
    """
    Разбиение текста документа на перекрывающиеся чанки по символам.

    Parameters
    ----------
    doc : Document
        Нормализованный документ
    chunk_size : int
        Размер чанка в символах (должен быть > 0)
    overlap : int
        Перекрытие между соседними чанками (0 <= overlap < chunk_size)

    Returns
    -------
    list[Chunk]
        Упорядоченный список чанков
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size должен быть положительным")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap должен быть неотрицательным и меньше chunk_size")

    text = doc.text
    if not text:
        return []

    chunks: list[Chunk] = []
    start = 0
    chunk_index = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk_text = text[start:end]

        metadata = dict(doc.metadata)
        metadata.update({"chunk_index": chunk_index, "char_start": start, "char_end": end})

        chunk_id = f"{doc.doc_id}_chunk_{chunk_index}"
        chunks.append(
            Chunk(
                doc_id=doc.doc_id,
                chunk_id=chunk_id,
                text=chunk_text,
                metadata=metadata,
                score=0.0,
                char_start=start,
                char_end=end,
            )
        )

        if end == len(text):
            break
        start = end - overlap
        chunk_index += 1

    return chunks


def chunk_documents(
    docs: Iterable[Document],
    *,
    chunk_size: int,
    overlap: int,
) -> list[Chunk]:
    """
    Массовое разбиение коллекции документов на чанки.

    Parameters
    ----------
    docs : Iterable[Document]
        Итератор документов
    chunk_size : int
        Размер чанка в символах
    overlap : int
        Перекрытие в символах

    Returns
    -------
    list[Chunk]
        Упорядоченный список всех чанков
    """
    setup_logging()
    logger.info("Start chunking documents")
    all_chunks: list[Chunk] = []
    doc_count = 0

    for doc in docs:
        if not doc.text:
            logger.warning("Документ %s пропущен из-за пустого текста", doc.doc_id)
            continue
        doc_chunks = chunk_document(doc, chunk_size=chunk_size, overlap=overlap)
        all_chunks.extend(doc_chunks)
        doc_count += 1
        if doc_count % PROGRESS_DOCS == 0:
            logger.info("Chunked %s документов, всего чанков: %s", doc_count, len(all_chunks))

    logger.info("Chunking завершён: %s документов → %s чанков", doc_count, len(all_chunks))
    return all_chunks
