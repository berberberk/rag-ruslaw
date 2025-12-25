import argparse
from pathlib import Path

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
        chunk_size=128,
        overlap=16,
    )

    run_retrieval(args)
    captured = capsys.readouterr()
    assert "tax" in captured.out
