from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from rag.logging import setup_logging

logger = logging.getLogger(__name__)


def load_summary(path: Path) -> pd.DataFrame:
    """
    Читает summary.csv с агрегированными метриками.

    Parameters
    ----------
    path : Path
        Путь к summary.csv

    Returns
    -------
    pd.DataFrame
        Датафрейм с колонками retriever, metric_name, mean, std
    """
    df = pd.read_csv(path)
    expected = {"retriever", "metric_name", "mean"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Отсутствуют колонки в summary: {missing}")
    if "std" not in df.columns:
        df["std"] = pd.NA
    if "embedding_model" not in df.columns:
        df["embedding_model"] = "none"
    return df


def build_comparison_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Формирует таблицу сравнения retriever'ов по precision/recall.

    Parameters
    ----------
    df : pd.DataFrame
        Данные summary

    Returns
    -------
    pd.DataFrame
        Таблица retriever -> метрика (mean ± std)
    """

    def _normalize_metric(name: str) -> str:
        if name.startswith("id_based_"):
            return name.replace("id_based_", "")
        return name

    def _fmt(row):
        std = "" if pd.isna(row["std"]) else f" ± {row['std']:.3f}"
        return f"{row['mean']:.3f}{std}"

    df = df.assign(metric_name=df["metric_name"].apply(_normalize_metric))
    filtered = df[df["metric_name"].isin(["context_precision", "context_recall"])]
    pivot = (
        filtered.assign(value=filtered.apply(_fmt, axis=1))
        .pivot_table(index="retriever", columns="metric_name", values="value", aggfunc="first")
        .reset_index()
    )
    return pivot


def plot_metric(df: pd.DataFrame, metric: str, out_path: Path) -> None:
    """
    Строит bar plot для метрики с погрешностями.

    Parameters
    ----------
    df : pd.DataFrame
        Данные summary
    metric : str
        Название метрики
    out_path : Path
        Путь для сохранения png
    """
    df = df.assign(metric_name=df["metric_name"].apply(lambda n: n.replace("id_based_", "")))
    data = df[df["metric_name"] == metric]
    if data.empty:
        logger.warning("Нет данных для метрики %s", metric)
        return

    plt.figure(figsize=(6, 4))
    plt.bar(data["retriever"], data["mean"], yerr=data["std"], capsize=5, color="#5b8ff9")
    plt.ylabel(metric)
    plt.xlabel("retriever")
    plt.title(f"{metric} по retriever'ам")
    plt.ylim(0, 1)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()
    logger.info("Сохранён график %s", out_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Визуализация retrieval метрик.")
    parser.add_argument("--summary", type=Path, default=Path("results/ragas/summary.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()
    df = load_summary(args.summary)
    table = build_comparison_table(df)

    tables_dir = args.output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    table_csv = tables_dir / "retriever_comparison.csv"
    table_md = tables_dir / "retriever_comparison.md"
    table.to_csv(table_csv, index=False)
    table.to_markdown(table_md, index=False)
    logger.info("Сохранена таблица сравнения: %s", table_csv)

    figs_dir = args.output_dir / "figures"
    plot_metric(df, "context_precision", figs_dir / "context_precision.png")
    plot_metric(df, "context_recall", figs_dir / "context_recall.png")


if __name__ == "__main__":
    main()
