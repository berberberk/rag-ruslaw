from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """
    Загрузка локального JSONL-файла с RusLawOD-совместимыми записями.

    Parameters
    ----------
    path : str | Path
        Путь до JSONL файла

    Returns
    -------
    List[Dict[str, Any]]
        Список записей, по одной на строку
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"JSONL not found: {p}")

    rows: list[dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows
