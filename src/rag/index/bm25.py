from __future__ import annotations

import logging
import re
from collections.abc import Iterable

from rank_bm25 import BM25Okapi

from rag.index.contracts import RetrievedChunk, Retriever
from rag.ingest.schema import Chunk
from rag.logging import setup_logging

logger = logging.getLogger(__name__)


def tokenize(text: str) -> list[str]:
    """
    Детерминированная токенизация: нижний регистр, разбиение по не-буквенно-цифровым символам.

    Parameters
    ----------
    text : str
        Исходный текст

    Returns
    -------
    list[str]
        Токены без пустых значений
    """
    return [token for token in re.split(r"\W+", text.lower()) if token]


class BM25Retriever(Retriever):
    """
    Лексический retriever на основе BM25Okapi.
    """

    def __init__(self, index: BM25Okapi, chunks: list[Chunk]) -> None:
        self._index = index
        self._chunks = chunks

    @classmethod
    def from_chunks(cls, chunks: Iterable[Chunk]) -> BM25Retriever:
        """
        Построить retriever из коллекции чанков.

        Parameters
        ----------
        chunks : Iterable[Chunk]
            Коллекция чанков

        Returns
        -------
        BM25Retriever
            Готовый retriever
        """
        setup_logging()
        chunk_list = list(chunks)
        if not chunk_list:
            raise ValueError("Нельзя строить BM25 индекс по пустому корпусу чанков")
        tokenized_corpus = [tokenize(c.text) for c in chunk_list]
        logger.info("Строим BM25 индекс по %s чанкам", len(chunk_list))
        index = BM25Okapi(tokenized_corpus)
        return cls(index=index, chunks=chunk_list)

    def retrieve(self, query: str, k: int) -> list[RetrievedChunk]:
        """
        Получение top-k чанков.

        Parameters
        ----------
        query : str
            Поисковый запрос
        k : int
            Количество чанков в ответе

        Returns
        -------
        list[RetrievedChunk]
            Список найденных чанков с score
        """
        tokens = tokenize(query)
        scores = self._index.get_scores(tokens)

        sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        top_indices = sorted_indices[: min(k, len(self._chunks))]

        results: list[RetrievedChunk] = []
        for idx in top_indices:
            chunk = self._chunks[idx]
            results.append(
                RetrievedChunk(
                    doc_id=chunk.doc_id,
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    metadata=chunk.metadata,
                    score=float(scores[idx]),
                )
            )
        return results
