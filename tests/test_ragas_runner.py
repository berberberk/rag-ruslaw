import json
from pathlib import Path

from rag.eval.ragas_runner import RetrievalResult, save_summary


def test_save_summary_creates_csv_and_json(tmp_path: Path):
    results = [
        RetrievalResult(retriever="bm25", k=5, metric="context_precision", scores=[0.5, 0.75]),
        RetrievalResult(retriever="dense", k=5, metric="context_precision", scores=[0.6]),
    ]
    out_csv = tmp_path / "summary.csv"
    out_json = tmp_path / "summary.json"
    save_summary(results, out_csv, out_json)

    assert out_csv.exists()
    assert out_json.exists()
    data = json.loads(out_json.read_text(encoding="utf-8"))
    assert data[0]["retriever"] == "bm25"
    assert data[0]["mean"] == 0.625
