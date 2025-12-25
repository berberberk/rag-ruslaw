from rag.index.bm25 import BM25Retriever
from rag.ingest.chunking import chunk_document
from rag.ingest.schema import Document


def test_bm25_retriever_returns_expected_doc():
    docs = [
        Document(doc_id="tax", title="", text="налог это платеж", metadata={"source": "t"}),
        Document(doc_id="civil", title="", text="договор это соглашение", metadata={"source": "t"}),
    ]
    chunks = []
    for doc in docs:
        chunks.extend(chunk_document(doc, chunk_size_chars=50, overlap_chars=0))

    retriever = BM25Retriever.from_chunks(chunks)
    results = retriever.retrieve("налог", k=1)

    assert len(results) == 1
    assert results[0].doc_id == "tax"
    assert isinstance(results[0].score, float)


def test_bm25_retriever_is_stable_between_calls():
    doc = Document(doc_id="d1", title="", text="a b c", metadata={})
    chunks = chunk_document(doc, chunk_size_chars=10, overlap_chars=0)
    retriever = BM25Retriever.from_chunks(chunks)

    res1 = retriever.retrieve("a", k=1)
    res2 = retriever.retrieve("a", k=1)
    assert res1 == res2


def test_bm25_retriever_handles_empty_query_and_large_k():
    doc = Document(doc_id="d2", title="", text="something", metadata={})
    chunks = chunk_document(doc, chunk_size_chars=5, overlap_chars=0)
    retriever = BM25Retriever.from_chunks(chunks)

    results = retriever.retrieve("", k=10)
    assert len(results) == len(chunks)
    assert all(isinstance(r.score, float) for r in results)
