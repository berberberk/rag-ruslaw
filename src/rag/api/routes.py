from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from rag.api.models import RetrievedChunk, RetrieveResponse
from rag.index.bm25 import BM25Retriever
from rag.ingest.chunking import chunk_documents
from rag.ingest.load_dataset import load_jsonl
from rag.ingest.preprocess import normalize_ruslawod_record

router = APIRouter()
logger = logging.getLogger(__name__)
CHUNK_SIZE = 512
OVERLAP = 64


def build_bm25_from_ruslawod_fixture() -> BM25Retriever:
    """
    Построить BM25 retriever из локальных RusLawOD-совместимых фикстур.

    Returns
    -------
    BM25Retriever
        Готовый retriever, собранный из мини-фикстуры
    """
    rows = load_jsonl("data/fixtures/mini_docs.jsonl")
    docs = [normalize_ruslawod_record(r) for r in rows]
    chunks = chunk_documents(docs, chunk_size=CHUNK_SIZE, overlap=OVERLAP)
    return BM25Retriever.from_chunks(chunks)


@router.get("/retrieve", response_model=RetrieveResponse)
def retrieve(query: str, k: int = 5, retriever: str = "bm25") -> RetrieveResponse:
    """
    Эндпоинт получения top-k чанков для заданного запроса.

    Parameters
    ----------
    query : str
        Поисковый запрос
    k : int
        Сколько чанков вернуть
    retriever : str
        Тип retriever'а

    Returns
    -------
    RetrieveResponse
        Результаты retrieval
    """
    if retriever != "bm25":
        raise HTTPException(
            status_code=400, detail="Only bm25 retriever is available in current MVP tests."
        )

    logger.info("BM25 retrieve: query='%s', k=%s", query, k)
    retriever_obj = build_bm25_from_ruslawod_fixture()
    chunks = retriever_obj.retrieve(query, k=k)

    results = [
        RetrievedChunk(
            doc_id=c.doc_id,
            chunk_id=c.chunk_id,
            text=c.text,
            score=c.score,
            metadata=c.metadata,
        )
        for c in chunks
    ]
    return RetrieveResponse(retriever="bm25", results=results)
