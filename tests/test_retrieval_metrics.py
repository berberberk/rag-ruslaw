from rag.eval.retrieval_metrics import (
    compute_doc_hit_at_k,
    compute_hit_at_k,
    compute_precision_at_k,
    compute_recall_at_k,
)
from rag.index.contracts import RetrievedChunk


def _res(doc_id: str, chunk_id: str) -> RetrievedChunk:
    return RetrievedChunk(doc_id=doc_id, chunk_id=chunk_id, text="", metadata={}, score=1.0)


def test_hit_at_k_and_doc_hit():
    results = [_res("d1", "c1"), _res("d2", "c2")]
    assert compute_hit_at_k(results, {"c1"}, k=1) == 1.0
    assert compute_hit_at_k(results, {"c3"}, k=2) == 0.0
    assert compute_doc_hit_at_k(results, {"d2"}, k=1) == 0.0
    assert compute_doc_hit_at_k(results, {"d2"}, k=2) == 1.0


def test_precision_and_recall():
    results = [_res("d1", "c1"), _res("d1", "c2"), _res("d2", "c3")]
    gold_chunks = {"c1", "c3"}
    assert compute_precision_at_k(results, gold_chunks, k=2) == 0.5
    assert compute_recall_at_k(results, gold_chunks, k=2) == 0.5
    assert compute_precision_at_k(results, set(), k=2) is None
    assert compute_recall_at_k(results, set(), k=2) is None
