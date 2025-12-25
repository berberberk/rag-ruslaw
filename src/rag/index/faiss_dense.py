from __future__ import annotations

import logging
from collections.abc import Callable, Iterable

import faiss
import numpy as np

from rag.index.contracts import RetrievedChunk, Retriever
from rag.ingest.schema import Chunk
from rag.logging import setup_logging

logger = logging.getLogger(__name__)

EmbedFunc = Callable[[list[str]], np.ndarray]


def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def build_faiss_index(
    chunks: Iterable[Chunk], embed_func: EmbedFunc
) -> tuple[faiss.IndexFlatIP, list[Chunk]]:
    """
    Построить FAISS IndexFlatIP по чанкам.

    Parameters
    ----------
    chunks : Iterable[Chunk]
        Коллекция чанков
    embed_func : Callable[[list[str]], np.ndarray]
        Функция получения эмбеддингов (batch)

    Returns
    -------
    Tuple[IndexFlatIP, List[Chunk]]
        Индекс и список чанков в порядке индекса
    """
    setup_logging()
    chunk_list = list(chunks)
    logger.info("Строим FAISS индекс по %s чанкам", len(chunk_list))

    texts = [c.text for c in chunk_list]
    embeddings = embed_func(texts).astype(np.float32)
    embeddings = _l2_normalize(embeddings)
    dim = embeddings.shape[1]
    if dim < 32:
        raise ValueError(f"Недопустимая размерность эмбеддингов: {dim} (ожидается >= 32)")

    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    logger.info("FAISS индекс построен: dim=%s", dim)
    return index, chunk_list


class DenseRetriever(Retriever):
    """
    Семантический retriever на основе FAISS IndexFlatIP.
    """

    def __init__(
        self, index: faiss.IndexFlatIP, chunks: list[Chunk], embed_func: EmbedFunc
    ) -> None:
        self._index = index
        self._chunks = chunks
        self._embed = embed_func

    def retrieve(self, query: str, k: int) -> list[RetrievedChunk]:
        """
        Получить top-k чанков по семантическому поиску.

        Parameters
        ----------
        query : str
            Поисковый запрос
        k : int
            Количество возвращаемых чанков

        Returns
        -------
        list[RetrievedChunk]
            Список найденных чанков с score
        """
        if not query:
            return []

        query_vec = self._embed([query]).astype(np.float32)
        query_vec = _l2_normalize(query_vec)
        scores, indices = self._index.search(query_vec, min(k, len(self._chunks)))

        results: list[RetrievedChunk] = []
        for score, idx in zip(scores[0], indices[0], strict=True):
            if idx == -1:
                continue
            chunk = self._chunks[idx]
            results.append(
                RetrievedChunk(
                    doc_id=chunk.doc_id,
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    metadata=chunk.metadata,
                    score=float(score),
                )
            )
        return results
