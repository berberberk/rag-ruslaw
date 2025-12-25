from pathlib import Path

import pandas as pd

from rag.embeddings.config import normalize_model_id
from rag.eval.metrics_loader import load_metrics_from_results


def test_load_metrics_fills_embedding_model_if_missing(tmp_path: Path):
    data = [
        {"retriever": "bm25", "k": 5, "metric_name": "context_precision", "mean": 0.5, "std": 0.1},
    ]
    csv_path = tmp_path / "summary.csv"
    pd.DataFrame(data).to_csv(csv_path, index=False)

    df = load_metrics_from_results(tmp_path)
    assert set(df["embedding_model"]) == {"none"}


def test_load_metrics_reads_embedding_model(tmp_path: Path):
    data = [
        {
            "retriever": "dense",
            "k": 5,
            "metric_name": "context_precision",
            "mean": 0.6,
            "std": 0.05,
            "embedding_model": "MiniLM-L12-v2",
        },
    ]
    csv_path = tmp_path / "summary_with_model.csv"
    pd.DataFrame(data).to_csv(csv_path, index=False)

    df = load_metrics_from_results(tmp_path)
    assert set(df["embedding_model"]) == {"MiniLM-L12-v2"}


def test_load_metrics_ignores_per_question(tmp_path: Path):
    per_question = pd.DataFrame([{"question": "q", "context_precision": 0.5}])
    per_question.to_csv(tmp_path / "per_question.csv", index=False)

    df = load_metrics_from_results(tmp_path)
    assert df.empty


def test_normalize_model_id_slug():
    name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    assert normalize_model_id(name) == "paraphrase-multilingual-MiniLM-L12-v2"
