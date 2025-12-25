from rag.ingest.schema import Chunk
from rag.rag_pipeline.generate import format_context


def _chunk(idx: int, text: str, score: float) -> Chunk:
    return Chunk(
        doc_id="doc1" if idx < 2 else "doc2",
        chunk_id=f"c{idx}",
        text=text,
        metadata={"title": f"Title {idx}", "docdate": "2020-01-01"},
        score=score,
    )


def test_format_context_includes_metadata_and_numbers():
    chunks = [
        _chunk(0, "Первый текст чанка", 1.0),
        _chunk(1, "Второй текст чанка", 0.9),
        _chunk(2, "Третий текст чанка", 0.8),
    ]
    ctx = format_context(chunks, max_chars=500, per_chunk_chars=30)
    assert "[1]" in ctx and "[2]" in ctx and "[3]" in ctx
    assert "doc_id=doc1" in ctx
    assert "chunk_id=c0" in ctx
    assert "Title 0" in ctx
    assert "2020-01-01" in ctx
    assert "Первый текст чан" in ctx  # обрезка работает
    assert "Третий текст чан" in ctx
