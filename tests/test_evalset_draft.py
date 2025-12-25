import csv
import json
from pathlib import Path

from rag.eval.evalset import generate_evalset_draft


def test_generate_evalset_draft_creates_questions(tmp_path: Path):
    catalog = tmp_path / "catalog.csv"
    rows = [
        [
            "doc_id",
            "title",
            "doc_type",
            "docdate",
            "doc_number",
            "status",
            "author",
            "signed",
            "keywords",
        ],
        ["1", "Закон о налогах", "law", "2020-01-02", "1", "ok", "a", "", "налог,платеж"],
        ["2", "Гражданский кодекс", "code", "2020-01-01", "2", "ok", "b", "", "договор,соглашение"],
    ]
    with catalog.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    output = tmp_path / "draft.jsonl"
    generate_evalset_draft(catalog, output, target_count=2, negative_count=1)

    entries = [
        json.loads(line) for line in output.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    assert len(entries) == 3  # 2 positives + 1 negative
    ids = [e["id"] for e in entries]
    assert ids[0] == "q_001"
    assert entries[0]["gold_doc_ids"] == ["1"]
    assert entries[-1]["gold_doc_ids"] == []  # negative
