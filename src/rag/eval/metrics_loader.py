from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

REQUIRED_COLS = {"retriever", "k", "metric_name", "mean"}


def _is_summary_schema(df: pd.DataFrame) -> bool:
    return REQUIRED_COLS.issubset(set(df.columns))


def load_metrics_from_results(results_dir: Path) -> pd.DataFrame:
    """
    Рекурсивно читает summary CSV с метриками retrieval.

    Parameters
    ----------
    results_dir : Path
        Корневая директория с CSV

    Returns
    -------
    pd.DataFrame
        Объединённый датафрейм метрик. Если embedding_model отсутствует, подставляется "none".
    """
    csv_files = list(results_dir.rglob("*.csv"))
    rows: list[pd.DataFrame] = []
    for file in csv_files:
        try:
            df = pd.read_csv(file)
        except Exception as exc:  # pragma: no cover
            logger.warning("Пропуск файла %s: %s", file, exc)
            continue
        if not _is_summary_schema(df):
            continue
        if "embedding_model" not in df.columns:
            df["embedding_model"] = "none"
        rows.append(df)
    if not rows:
        logger.warning("Метрики не найдены в %s", results_dir)
        return pd.DataFrame(
            columns=list(REQUIRED_COLS) + ["std", "num_questions", "embedding_model"]
        )

    df_all = pd.concat(rows, ignore_index=True)
    df_all["k"] = df_all["k"].astype(int)
    df_all["mean"] = df_all["mean"].astype(float)
    if "std" in df_all.columns:
        df_all["std"] = df_all["std"].astype(float)
    if "num_questions" in df_all.columns:
        df_all["num_questions"] = df_all["num_questions"].astype(int)
    # Заполняем отсутствующие столбцы NaN
    for col in ["std", "num_questions"]:
        if col not in df_all.columns:
            df_all[col] = pd.NA
    return df_all
