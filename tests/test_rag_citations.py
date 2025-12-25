from rag.ingest.schema import Chunk
from rag.rag_pipeline.citations import build_citations


def test_citations_dedup_and_sorted():
    chunks = [
        Chunk(
            doc_id="d1",
            chunk_id="c1",
            text="t1",
            metadata={"title": "A", "docdate": "2020-01-01", "doc_type": "law", "doc_number": "1"},
            score=0.9,
        ),
        Chunk(
            doc_id="d1",
            chunk_id="c2",
            text="t2",
            metadata={"title": "A2", "docdate": "2020-01-02"},
            score=0.8,
        ),
        Chunk(
            doc_id="d2",
            chunk_id="c3",
            text="t3",
            metadata={"title": "B", "docdate": "2019-01-01"},
            score=0.95,
        ),
    ]
    cites = build_citations(chunks)
    assert cites[0]["doc_id"] == "d2"  # выше score
    assert cites[1]["doc_id"] == "d1"
    assert cites[1]["title"] == "A"  # берет лучший
