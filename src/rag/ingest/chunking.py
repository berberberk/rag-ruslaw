from __future__ import annotations

import logging
import re
from collections.abc import Iterable

from rag.ingest.preprocess import clean_text
from rag.ingest.schema import Chunk, Document
from rag.logging import setup_logging

logger = logging.getLogger(__name__)
PROGRESS_DOCS = 50


def _split_into_blocks(text: str) -> list[str]:
    """
    Разбиение текста на логические блоки по разделителям.

    Parameters
    ----------
    text : str
        Очищенный текст

    Returns
    -------
    list[str]
        Список блоков
    """
    # Разделители: двойной перенос, нумерованные пункты, ключевые слова
    pattern = r"(\n\n|^\s*\d+\.\s|^Статья\s|^Глава\s|^Раздел\s)"
    parts = re.split(pattern, text, flags=re.MULTILINE)
    blocks: list[str] = []
    buffer = ""
    for part in parts:
        if not part:
            continue
        if re.match(pattern, part, flags=re.MULTILINE):
            if buffer.strip():
                blocks.append(buffer.strip())
            buffer = part
        else:
            buffer += part
    if buffer.strip():
        blocks.append(buffer.strip())
    return blocks if blocks else [text.strip()]


def chunk_document(
    doc: Document,
    *,
    chunk_size_chars: int,
    overlap_chars: int,
) -> list[Chunk]:
    """
    Разбиение текста документа на перекрывающиеся чанки по символам.

    Parameters
    ----------
    doc : Document
        Нормализованный документ
    chunk_size_chars : int
        Размер чанка в символах (должен быть > 0)
    overlap_chars : int
        Перекрытие между соседними чанками (0 <= overlap < chunk_size)

    Returns
    -------
    list[Chunk]
        Упорядоченный список чанков
    """
    if chunk_size_chars <= 0:
        raise ValueError("chunk_size должен быть положительным")
    if overlap_chars < 0 or overlap_chars >= chunk_size_chars:
        raise ValueError("overlap должен быть неотрицательным и меньше chunk_size")

    text = clean_text(doc.text)
    if not text:
        return []

    blocks = _split_into_blocks(text)
    chunks: list[Chunk] = []
    chunk_index = 0

    for block in blocks:
        if not block:
            continue
        start = 0
        while start < len(block):
            end = min(start + chunk_size_chars, len(block))
            chunk_text = block[start:end]
            if not chunk_text.strip():
                break
            global_start = start  # относительный, так как блоки независимы

            metadata = dict(doc.metadata)
            metadata.update(
                {
                    "chunk_index": chunk_index,
                    "char_start": global_start,
                    "char_end": global_start + len(chunk_text),
                }
            )

            chunk_id = f"{doc.doc_id}_chunk_{chunk_index}"
            chunks.append(
                Chunk(
                    doc_id=doc.doc_id,
                    chunk_id=chunk_id,
                    text=chunk_text.strip(),
                    metadata=metadata,
                    score=0.0,
                    char_start=global_start,
                    char_end=global_start + len(chunk_text),
                )
            )

            if end == len(block):
                break
            start = end - overlap_chars
            chunk_index += 1
        # следующий блок — новый индекс
        chunk_index += 1

    return chunks


def chunk_documents(
    docs: Iterable[Document],
    *,
    chunk_size_chars: int,
    overlap_chars: int,
) -> list[Chunk]:
    """
    Массовое разбиение коллекции документов на чанки.

    Parameters
    ----------
    docs : Iterable[Document]
        Итератор документов
    chunk_size_chars : int
        Размер чанка в символах
    overlap_chars : int
        Перекрытие в символах

    Returns
    -------
    list[Chunk]
        Упорядоченный список всех чанков
    """
    setup_logging()
    logger.info(
        "Start chunking documents (chunk_size=%s, overlap=%s)", chunk_size_chars, overlap_chars
    )
    all_chunks: list[Chunk] = []
    doc_count = 0

    for doc in docs:
        if not doc.text:
            logger.warning("Документ %s пропущен из-за пустого текста", doc.doc_id)
            continue
        doc_chunks = chunk_document(
            doc, chunk_size_chars=chunk_size_chars, overlap_chars=overlap_chars
        )
        all_chunks.extend(doc_chunks)
        doc_count += 1
        if doc_count % PROGRESS_DOCS == 0:
            logger.info("Chunked %s документов, всего чанков: %s", doc_count, len(all_chunks))

    logger.info("Chunking завершён: %s документов → %s чанков", doc_count, len(all_chunks))
    return all_chunks
