from rag.index.contracts import RetrievedChunk
from rag.index.hybrid import HybridRetriever


class _FakeRetriever:
    def __init__(self, results):
        self._results = results

    def retrieve(self, query: str, k: int):
        return self._results[:k]


def test_hybrid_retrieval_is_deterministic():
    bm25_results = [
        RetrievedChunk(doc_id="a", chunk_id="a1", text="t", metadata={}, score=2.0),
        RetrievedChunk(doc_id="b", chunk_id="b1", text="t", metadata={}, score=1.0),
    ]
    dense_results = [
        RetrievedChunk(doc_id="b", chunk_id="b1", text="t", metadata={}, score=0.8),
        RetrievedChunk(doc_id="c", chunk_id="c1", text="t", metadata={}, score=0.7),
    ]
    hybrid = HybridRetriever(
        bm25_retriever=_FakeRetriever(bm25_results),
        dense_retriever=_FakeRetriever(dense_results),
        w_bm25=0.5,
        w_dense=0.5,
    )

    res1 = hybrid.retrieve("q", k=2)
    res2 = hybrid.retrieve("q", k=2)
    assert res1 == res2


def test_hybrid_contains_expected_doc():
    bm25_results = [
        RetrievedChunk(doc_id="tax", chunk_id="tax1", text="налог", metadata={}, score=2.0),
    ]
    dense_results = [
        RetrievedChunk(doc_id="other", chunk_id="o1", text="другое", metadata={}, score=0.5),
    ]
    hybrid = HybridRetriever(
        bm25_retriever=_FakeRetriever(bm25_results),
        dense_retriever=_FakeRetriever(dense_results),
    )

    results = hybrid.retrieve("налог", k=2)
    doc_ids = [r.doc_id for r in results]
    assert "tax" in doc_ids
