import json
from pathlib import Path

from rag.cli.chunk_stats import compute_chunk_stats
from rag.ingest.schema import Document


def test_chunk_stats_outputs_basic_fields(tmp_path: Path):
    docs = [
        Document(doc_id="1", title="t1", text="abcde" * 10, metadata={}),
        Document(doc_id="2", title="t2", text="fghij" * 8, metadata={}),
    ]
    output = tmp_path / "chunk_stats.json"
    compute_chunk_stats(docs, output, chunk_size_chars=20, overlap_chars=5)

    data = json.loads(output.read_text(encoding="utf-8"))
    assert "num_docs" in data and "num_chunks" in data
    assert data["num_chunks"] > 0
    assert "html_ratio" in data
