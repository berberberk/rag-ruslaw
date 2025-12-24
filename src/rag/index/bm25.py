from rag.index.bm25 import BM25Retriever
from rag.ingest.load_dataset import load_jsonl
from rag.ingest.preprocess import normalize_ruslawod_record


def test_bm25_retriever_returns_expected_doc_in_topk_from_ruslawod_fixture():
    rows = load_jsonl("data/fixtures/ruslawod_mini.jsonl")
    docs = [normalize_ruslawod_record(r) for r in rows]

    retriever = BM25Retriever.from_documents(docs)
    results = retriever.retrieve("что такое налог", k=2)

    assert len(results) == 2
    # В фикстурах "налог" есть в записи с doc_id 123456789
    assert results[0].doc_id == "123456789"
    assert results[0].score >= results[1].score
    assert isinstance(results[0].text, str) and len(results[0].text) > 0
