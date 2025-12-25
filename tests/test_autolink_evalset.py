import json
from pathlib import Path

from rag.eval.evalset import autolink_evalset_chunks
from rag.index.contracts import RetrievedChunk


class _FakeRetriever:
    def retrieve(self, query: str, k: int):
        return [
            RetrievedChunk(
                doc_id="d1",
                chunk_id="c1",
                text="t",
                metadata={},
                score=1.0,
            )
        ]


def test_autolink_populates_chunk_ids(tmp_path: Path):
    draft = tmp_path / "draft.jsonl"
    entry = {
        "id": "q_001",
        "type": "fact",
        "question": "Что такое налог?",
        "gold_doc_ids": ["d1"],
        "gold_chunk_ids": [],
        "notes": "auto-generated",
        "source": "auto",
    }
    draft.write_text(json.dumps(entry, ensure_ascii=False) + "\n", encoding="utf-8")

    output = tmp_path / "evalset.jsonl"
    autolink_evalset_chunks(draft, output, retriever=_FakeRetriever(), top_k=1)

    lines = output.read_text(encoding="utf-8").splitlines()
    data = json.loads(lines[0])
    assert data["gold_chunk_ids"] == ["c1"]
