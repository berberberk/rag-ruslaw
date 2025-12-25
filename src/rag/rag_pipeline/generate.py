from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from rag.ingest.schema import Chunk
from rag.llm.openrouter import OpenRouterClient
from rag.llm.prompts import FALLBACK_TEXT, build_prompt
from rag.rag_pipeline.citations import build_citations

logger = logging.getLogger(__name__)


def format_context(
    chunks: Iterable[Chunk],
    *,
    max_chars: int = 16000,
    per_chunk_chars: int = 1800,
) -> str:
    """
    Формирует текстовый контекст из чанков с нумерацией и метаданными.

    Parameters
    ----------
    chunks : Iterable[Chunk]
        Чанки в порядке убывания релевантности
    max_chars : int, optional
        Максимальная длина итогового контекста
    per_chunk_chars : int, optional
        Максимальная длина текста на один чанк

    Returns
    -------
    str
        Отформатированный контекст
    """
    parts: list[str] = []
    used = 0
    for idx, ch in enumerate(chunks, start=1):
        if used >= max_chars:
            break
        snippet = (ch.text or "")[:per_chunk_chars]
        md = ch.metadata or {}
        line = (
            f"[{idx}] doc_id={ch.doc_id} chunk_id={ch.chunk_id} "
            f"title={md.get('title') or md.get('heading') or md.get('headingIPS') or '—'} "
            f"date={md.get('docdate') or md.get('docdateIPS') or '—'}\n{snippet}"
        )
        parts.append(line)
        used += len(line)
        if used >= max_chars:
            break
    return "\n\n".join(parts)


@dataclass
class RagResponse:
    """
    Ответ RAG пайплайна.
    """

    answer: str
    citations: list[dict]
    retrieved_chunks: list[dict]


def rag_answer(
    query: str,
    *,
    retriever_func: Callable[[str, int, str | None], list[Chunk]],
    k: int,
    embedding_model: str | None = None,
    llm_client: OpenRouterClient | None = None,
    temperature: float = 0.1,
    max_tokens: int = 600,
) -> RagResponse:
    """
    Выполняет RAG: retrieval → контекст → генерация ответа.

    Parameters
    ----------
    query : str
        Вопрос пользователя
    retriever_func : Callable
        Функция retrieval (query, k, embedding_model) -> List[Chunk]
    k : int
        Размер выборки
    embedding_model : str | None
        Модель эмбеддингов для dense/hybrid
    llm_client : OpenRouterClient, optional
        Клиент LLM

    Returns
    -------
    RagResponse
        Ответ с цитатами и списком чанков
    """
    chunks = retriever_func(query, k, embedding_model)
    if not chunks:
        return RagResponse(answer=FALLBACK_TEXT, citations=[], retrieved_chunks=[])

    context = format_context(chunks)
    citations = build_citations(chunks)
    retrieved_chunks = [
        {
            "doc_id": ch.doc_id,
            "chunk_id": ch.chunk_id,
            "score": ch.score,
            "text_preview": (ch.text or "")[:400],
            "metadata": ch.metadata,
        }
        for ch in chunks
    ]

    if llm_client is None:
        return RagResponse(
            answer=FALLBACK_TEXT, citations=citations, retrieved_chunks=retrieved_chunks
        )

    messages = build_prompt(query, context)
    answer = llm_client.chat(messages, temperature=temperature, max_tokens=max_tokens)
    return RagResponse(answer=answer, citations=citations, retrieved_chunks=retrieved_chunks)
