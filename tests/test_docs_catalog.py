import csv
from pathlib import Path

from rag.cli.docs_catalog import build_docs_catalog


def test_docs_catalog_writes_expected_csv(tmp_path: Path):
    slice_path = tmp_path / "slice.jsonl.gz"
    rows = [
        {
            "doc_id": "2",
            "title": "t2",
            "text": "b",
            "metadata": {
                "doc_type": "B",
                "docdate": "2020-01-02",
                "doc_number": "2",
                "status": "ok",
            },
        },
        {
            "doc_id": "1",
            "title": "t1",
            "text": "a",
            "metadata": {
                "doc_type": "A",
                "docdate": "2020-01-01",
                "doc_number": "1",
                "status": "ok",
            },
        },
    ]
    import gzip
    import json

    with gzip.open(slice_path, "wt", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    output_csv = tmp_path / "docs_catalog.csv"
    build_docs_catalog(slice_path, output_csv)

    assert output_csv.exists()
    with output_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        data = list(reader)

    assert reader.fieldnames == [
        "doc_id",
        "title",
        "doc_type",
        "docdate",
        "doc_number",
        "status",
        "author",
        "signed",
        "keywords",
    ]
    assert [row["doc_id"] for row in data] == ["2", "1"]  # sorted by docdate desc then doc_id
