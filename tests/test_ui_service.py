from rag.ingest.schema import Document
from rag.ui.service import build_retrievers_for_ui, chunk_loaded_documents, run_retrieval


def test_run_retrieval_returns_k_results(monkeypatch):
    docs = [
        Document(doc_id="a", title="", text="налог это обязательный платеж", metadata={}),
        Document(doc_id="b", title="", text="договор это соглашение", metadata={}),
    ]
    chunks = chunk_loaded_documents(docs, chunk_size_chars=64, overlap_chars=8)
    retrievers = build_retrievers_for_ui(
        chunks,
        retriever_names=["bm25"],
        embedding_model=None,
        embedding_batch_size=4,
        min_chunk_chars=0,
    )

    results = run_retrieval("налог", "bm25", k=2, retrievers=retrievers)
    assert len(results) == 2
    for r in results:
        assert r.doc_id
        assert r.chunk_id
        assert r.text
        assert r.metadata is not None
        assert isinstance(r.score, float)
