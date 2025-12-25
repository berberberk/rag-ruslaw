import gzip
import json
from pathlib import Path

import pytest

from rag.index.bm25 import BM25Retriever
from rag.ingest.chunking import chunk_documents
from rag.ingest.load_dataset import load_documents_from_slice
from rag.ingest.schema import Document


def test_load_documents_from_normalized_slice(tmp_path: Path):
    slice_path = tmp_path / "slice.jsonl.gz"
    rows = [
        {"doc_id": "1", "title": "t1", "text": "abc", "metadata": {"doc_type": "law"}},
        {"doc_id": "2", "title": "t2", "text": "def", "metadata": {"docdate": "2020-01-01"}},
    ]
    with gzip.open(slice_path, "wt", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    docs = load_documents_from_slice(slice_path)
    assert len(docs) == 2
    assert docs[0].doc_id == "1"
    assert docs[0].text == "abc"
    assert isinstance(docs[0].metadata, dict)


def test_chunk_documents_produces_chunks():
    doc = Document(doc_id="d1", title="", text="abcdef", metadata={})
    chunks = chunk_documents([doc], chunk_size=3, overlap=1)

    assert len(chunks) > 0
    assert chunks[0].doc_id == "d1"
    assert chunks[0].chunk_id.startswith("d1_chunk_")


def test_bm25_from_chunks_raises_on_empty():
    with pytest.raises(ValueError) as err:
        BM25Retriever.from_chunks([])
    assert "пустому корпусу" in str(err.value)
