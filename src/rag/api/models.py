from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class RetrieveRequest(BaseModel):
    """
    Запрос на получение top-k чанков.

    Parameters
    ----------
    query : str
        Поисковый запрос пользователя
    k : int
        Число возвращаемых чанков
    retriever : str
        Идентификатор retriever'а (bm25/dense/hybrid)
    """

    query: str = Field(..., min_length=1)
    k: int = Field(..., gt=0, le=10)
    retriever: Literal["bm25", "dense", "hybrid"] = "bm25"


class RetrievedChunk(BaseModel):
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
    score : float
        Вес/оценка релевантности
    """

    doc_id: str
    chunk_id: str
    text: str
    score: float
    metadata: dict | None = None


class RetrieveResponse(BaseModel):
    """
    Ответ API на /retrieve.

    Parameters
    ----------
    results : list[RetrievedChunk]
        Список найденных чанков
    retriever : str
        Используемый retriever
    """

    retriever: str
    results: list[RetrievedChunk]
