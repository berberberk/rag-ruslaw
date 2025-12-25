import json
from pathlib import Path

from rag.eval.evalset_llm import generate_evalset_llm
from rag.index.contracts import RetrievedChunk
from rag.ingest.schema import Document


class _FakeLLM:
    def generate_questions(self, doc: Document, n_questions: int):
        return [
            {
                "question": f"Что такое {doc.title}?",
                "type": "definition",
                "reason": "title-based",
                "keywords_used": [doc.title],
            }
        ]

    def generate_negative(self, n_questions: int):
        return [
            {
                "question": "Что такое марсоход?",
                "type": "negative",
                "reason": "out-of-domain",
                "keywords_used": [],
            }
        ]


class _FakeRetriever:
    def retrieve(self, query: str, k: int):
        return [
            RetrievedChunk(
                doc_id="doc1",
                chunk_id="doc1_chunk_0",
                text="t",
                metadata={},
                score=1.0,
            )
        ]


def test_generate_evalset_llm_writes_files(tmp_path: Path):
    slice_path = tmp_path / "slice.jsonl.gz"
    import gzip

    rows = [
        {
            "doc_id": "doc1",
            "title": "Налог",
            "text": "налог это платеж",
            "metadata": {"doc_type": "law"},
        },
    ]
    with gzip.open(slice_path, "wt", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    output = tmp_path / "evalset.jsonl"
    report = tmp_path / "report.json"
    traces_dir = tmp_path / "llm_traces"

    generate_evalset_llm(
        slice_path=slice_path,
        output_path=output,
        report_path=report,
        n_docs=1,
        questions_per_doc=1,
        n_negative=1,
        retriever_name="bm25",
        top_k=1,
        dry_run=False,
        llm_client=_FakeLLM(),
        retriever=_FakeRetriever(),
        traces_dir=traces_dir,
    )

    assert output.exists()
    entries = [
        json.loads(line) for line in output.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    assert len(entries) == 2  # 1 positive + 1 negative
    assert entries[0]["gold_doc_ids"] == ["doc1"]
    assert entries[0]["gold_chunk_ids"] == ["doc1_chunk_0"]
    assert report.exists()
