from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass
class RetrievedChunk:
    """
    Результат retrieval — один чанк.

    Parameters
    ----------
    doc_id : str
        Идентификатор документа
    chunk_id : str
        Идентификатор чанка
    text : str
        Текст чанка
    metadata : dict
        Метаданные чанка
    score : float
        Релевантность чанка
    """

    doc_id: str
    chunk_id: str
    text: str
    metadata: dict
    score: float


class Retriever(Protocol):
    """
    Базовый контракт retriever'а.
    """

    def retrieve(self, query: str, k: int) -> list[RetrievedChunk]:
        """
        Получить топ-k релевантных чанков.
        """
        ...
