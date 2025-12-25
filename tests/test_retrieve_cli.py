import argparse
from pathlib import Path

import pytest

from rag.cli import retrieve_cli
from rag.cli.retrieve_cli import run_retrieval
from rag.ingest.schema import Chunk, Document


def _build_synthetic_chunks() -> list[Chunk]:
    docs = [
        Document(doc_id="tax", title="Налоги", text="налог это обязательный платеж", metadata={}),
        Document(doc_id="civil", title="ГК", text="договор это соглашение", metadata={}),
    ]
    chunks = []
    for doc in docs:
        chunks.append(
            Chunk(
                doc_id=doc.doc_id,
                chunk_id=f"{doc.doc_id}_chunk_0",
                text=doc.text,
                metadata=doc.metadata,
                score=0.0,
                char_start=0,
                char_end=len(doc.text),
            )
        )
    return chunks


def _make_chunk(doc_id: str, text: str, idx: int = 0) -> Chunk:
    return Chunk(
        doc_id=doc_id,
        chunk_id=f"{doc_id}_chunk_{idx}",
        text=text,
        metadata={},
        score=0.0,
        char_start=0,
        char_end=len(text),
    )


def test_run_retrieval_returns_expected_doc_id(monkeypatch, capsys, tmp_path: Path):
    synthetic_chunks = _build_synthetic_chunks()

    def _load_chunks(*_, **__):
        return synthetic_chunks

    monkeypatch.setattr("rag.cli.retrieve_cli.load_and_prepare_chunks", _load_chunks)

    args = argparse.Namespace(
        query="налог",
        k=2,
        retriever="bm25",
        slice_path=tmp_path / "dummy.gz",
        chunk_size_chars=128,
        overlap_chars=16,
    )

    run_retrieval(args)
    captured = capsys.readouterr()
    assert "tax" in captured.out


def test_retrieve_cli_builds_only_requested_retriever(monkeypatch):
    calls = {"dense": 0}

    def _fake_build_faiss_index(chunks, embed_func):
        calls["dense"] += 1
        raise AssertionError("should not be called")

    monkeypatch.setattr("rag.cli.retrieve_cli.build_faiss_index", _fake_build_faiss_index)

    chunks = _build_synthetic_chunks()
    # only bm25 expected
    retrievers = retrieve_cli.build_retrievers(chunks, use_dense=False, use_hybrid=False)
    assert "bm25" in retrievers
    assert "dense" not in retrievers
    assert calls["dense"] == 0


def test_dense_dim_guard(monkeypatch):
    import numpy as np

    def _bad_embed(texts: list[str]):
        return np.zeros((len(texts), 1), dtype=np.float32)

    chunk = _make_chunk("x", "text")
    with pytest.raises(ValueError):
        retrieve_cli.build_retrievers(
            [chunk], use_dense=True, use_hybrid=False, embed_func_override=_bad_embed
        )
