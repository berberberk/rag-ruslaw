from __future__ import annotations

from fastapi import APIRouter, HTTPException

from rag.api.models import RetrievedChunk, RetrieveRequest, RetrieveResponse
from rag.index.bm25 import BM25Retriever
from rag.ingest.load_dataset import load_jsonl
from rag.ingest.preprocess import normalize_ruslawod_record

router = APIRouter()


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
    return BM25Retriever.from_documents(docs)


@router.post("/retrieve", response_model=RetrieveResponse)
def retrieve(req: RetrieveRequest) -> RetrieveResponse:
    """
    Эндпоинт получения top-k чанков для заданного запроса.

    Parameters
    ----------
    req : RetrieveRequest
        Тело запроса

    Returns
    -------
    RetrieveResponse
        Результаты retrieval
    """
    if req.retriever != "bm25":
        raise HTTPException(
            status_code=400, detail="Only bm25 retriever is available in current MVP tests."
        )

    retriever = build_bm25_from_ruslawod_fixture()
    chunks = retriever.retrieve(req.query, k=req.k)

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
    return RetrieveResponse(results=results)
