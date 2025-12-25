import json
import os
from pathlib import Path

from ragas.metrics._context_precision import IDBasedContextPrecision
from ragas.metrics._context_recall import IDBasedContextRecall

from rag.eval.ragas_runner import EvaluationConfig, RetrievalResult, run_ragas, save_summary


def test_save_summary_creates_csv_and_json(tmp_path: Path):
    results = [
        RetrievalResult(
            retriever="bm25",
            embedding_model="none",
            k=5,
            metric="context_precision",
            scores=[0.5, 0.75],
        ),
        RetrievalResult(
            retriever="dense",
            embedding_model="MiniLM",
            k=5,
            metric="context_precision",
            scores=[0.6],
        ),
    ]
    out_csv = tmp_path / "summary.csv"
    out_json = tmp_path / "summary.json"
    save_summary(results, out_csv, out_json)

    assert out_csv.exists()
    assert out_json.exists()
    data = json.loads(out_json.read_text(encoding="utf-8"))
    assert data[0]["retriever"] == "bm25"
    assert data[0]["mean"] == 0.625
    assert data[0]["embedding_model"] == "none"


def test_ragas_metrics_are_metric_objects():
    metrics = [IDBasedContextPrecision(), IDBasedContextRecall()]
    for metric in metrics:
        assert hasattr(metric, "name")
        assert metric.name


def test_run_ragas_returns_empty_on_empty_evalset(tmp_path: Path):
    evalset_path = tmp_path / "evalset.jsonl"
    evalset_path.write_text("", encoding="utf-8")
    cfg = EvaluationConfig(
        evalset_path=evalset_path,
        retrievers=["bm25"],
        k=1,
        output_dir=tmp_path,
    )
    results = run_ragas(cfg)
    assert results == []
    assert os.environ.get("KMP_DUPLICATE_LIB_OK", "").lower() == "true"
