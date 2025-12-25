from __future__ import annotations

import re
from typing import Iterable  # noqa: UP035

from rank_bm25 import BM25Okapi

from rag.ingest.schema import Chunk, Document


class BM25Retriever:
    """
    Лексический retriever на основе BM25Okapi.
    """

    def __init__(self, index: BM25Okapi, chunks: list[Chunk]) -> None:
        """
        Создание объекта BM25Retriever.

        Parameters
        ----------
        index : BM25Okapi
            Предподсчитанный индекс для корпуса
        chunks : List[Chunk]
            Список чанков, соответствующих строкам в индексе
        """
        self._index = index
        self._chunks = chunks

    @classmethod
    def from_documents(cls, documents: Iterable[Document]) -> BM25Retriever:
        """
        Построение retriever из списка документов без предварительного чанкинга.

        Parameters
        ----------
        documents : Iterable[Document]
            Список нормализованных документов

        Returns
        -------
        BM25Retriever
            Готовый retriever с индексом
        """
        tokenized_corpus: list[list[str]] = []
        chunks: list[Chunk] = []

        for idx, doc in enumerate(documents):
            tokens = cls._tokenize(doc.text)
            tokenized_corpus.append(tokens)
            chunk_id = f"{doc.doc_id}_chunk_{idx}"
            chunks.append(
                Chunk(
                    doc_id=doc.doc_id,
                    chunk_id=chunk_id,
                    text=doc.text,
                    metadata=doc.metadata,
                    score=0.0,
                    char_start=0,
                    char_end=len(doc.text),
                )
            )

        index = BM25Okapi(tokenized_corpus)
        return cls(index=index, chunks=chunks)

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """
        Детерминированная токенизация: приведение к нижнему регистру и разбиение по не-буквенно-цифровым символам.

        Parameters
        ----------
        text : str
            Исходный текст

        Returns
        -------
        List[str]
            Список токенов
        """
        return [token for token in re.split(r"\W+", text.lower()) if token]

    def retrieve(self, query: str, k: int) -> list[Chunk]:
        """
        Получение top-k чанков по BM25.

        Parameters
        ----------
        query : str
            Текст запроса
        k : int
            Количество возвращаемых чанков

        Returns
        -------
        List[Chunk]
            Отсортированный список чанков с присвоенными score
        """
        tokens = self._tokenize(query)
        scores = self._index.get_scores(tokens)

        sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        top_indices = sorted_indices[: min(k, len(self._chunks))]

        results: list[Chunk] = []
        for idx in top_indices:
            chunk = self._chunks[idx]
            results.append(
                Chunk(
                    doc_id=chunk.doc_id,
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    metadata=chunk.metadata,
                    score=float(scores[idx]),
                )
            )
        return results
