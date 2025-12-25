import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

matplotlib = pytest.importorskip("matplotlib")

from scripts.analysis.plot_retrieval_metrics import (  # noqa: E402
    build_comparison_table,
    load_summary,
    plot_metric,
)


def _write_summary(path: Path) -> None:
    data = [
        {
            "retriever": "bm25",
            "metric_name": "id_based_context_precision",
            "mean": 0.4,
            "std": 0.05,
            "k": 5,
            "num_questions": 10,
        },
        {
            "retriever": "bm25",
            "metric_name": "id_based_context_recall",
            "mean": 0.6,
            "std": 0.04,
            "k": 5,
            "num_questions": 10,
        },
        {
            "retriever": "dense",
            "metric_name": "id_based_context_precision",
            "mean": 0.5,
            "std": 0.02,
            "k": 5,
            "num_questions": 10,
        },
        {
            "retriever": "dense",
            "metric_name": "id_based_context_recall",
            "mean": 0.7,
            "std": 0.03,
            "k": 5,
            "num_questions": 10,
        },
    ]
    pd.DataFrame(data).to_csv(path, index=False)


def test_analysis_builds_table_and_plots(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("MPLBACKEND", "Agg")
    summary_path = tmp_path / "summary.csv"
    _write_summary(summary_path)

    df = load_summary(summary_path)
    table = build_comparison_table(df)
    assert {"bm25", "dense"} == set(table["retriever"])

    figs_dir = tmp_path / "figs"
    plot_metric(df, "context_precision", figs_dir / "context_precision.png")
    plot_metric(df, "context_recall", figs_dir / "context_recall.png")
    assert (figs_dir / "context_precision.png").exists()
    assert (figs_dir / "context_recall.png").exists()
