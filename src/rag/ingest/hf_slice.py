from __future__ import annotations

import argparse
import gzip
import json
import logging
import os
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from datasets import load_dataset  # type: ignore

from rag.ingest.preprocess import normalize_ruslawod_record
from rag.logging import setup_logging

logger = logging.getLogger(__name__)


def parse_data_files(raw: str | Sequence[str] | Mapping[str, Sequence[str]] | None):
    """
    Преобразование значения data_files из CLI/env в формат для load_dataset.

    Parameters
    ----------
    raw : str | Sequence[str] | Mapping[str, Sequence[str]] | None
        Исходное значение из аргументов или переменных окружения

    Returns
    -------
    str | list[str] | Mapping[str, list[str]] | None
        Нормализованное значение либо None, если вход пуст
    """
    if raw is None:
        return None
    if isinstance(raw, str):
        cleaned = [part.strip() for part in raw.split(",") if part.strip()]
        return cleaned or None
    if isinstance(raw, Mapping):
        result: dict[str, list[str]] = {}
        for key, values in raw.items():
            filtered = [v for v in values if v and str(v).strip()]
            if filtered:
                result[key] = filtered
        return result or None
    cleaned_list = [str(v).strip() for v in raw if str(v).strip()]
    return cleaned_list or None


def _write_jsonl_gzip(path: Path, rows: Iterable[dict]) -> None:
    """
    Сериализация записей в gzip-сжатый JSONL.

    Parameters
    ----------
    path : Path
        Целевой путь для сохранения
    rows : Iterable[dict]
        Итератор по словарям для записи
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")


def build_hf_slice(
    dataset_id: str,
    split: str,
    n: int,
    seed: int,
    output_dir: Path,
    buffer_size: int = 10_000,
    strategy: Literal["download", "stream"] = "download",
    max_retries: int = 3,
    backoff_base: float = 1.5,
    data_files: str | Sequence[str] | Mapping[str, Sequence[str]] | None = None,
    filters: Mapping[str, Sequence[str]] | None = None,
) -> tuple[Path, Path]:
    """
    Загрузить RusLawOD с HuggingFace, сделать детерминированный срез и сохранить gzip JSONL.

    Parameters
    ----------
    dataset_id : str
        Идентификатор датасета на HuggingFace
    split : str
        Сплит (например, 'train')
    n : int
        Количество записей в срезе
    seed : int
        Зерно для детерминированного shuffle
    output_dir : Path
        Директория для сохранения артефактов
    buffer_size : int, optional
        Буфер для стримингового shuffle
    strategy : Literal["download", "stream"], optional
        Режим загрузки: скачивание в кэш (по умолчанию) или стриминг
    max_retries : int, optional
        Число попыток при сетевых сбоях
    backoff_base : float, optional
        Основание экспоненциальной задержки между повторами
    data_files : str | Sequence[str] | Mapping[str, Sequence[str]] | None, optional
        Явное ограничение списка файлов/шардов HF
    filters : Mapping[str, Sequence[str]] | None, optional
        Фильтры по полям raw-записей (например, doc_typeIPS/statusIPS)
        При отсутствии слайса в split автоматически применяется ограничение split[:N*5],
        чтобы не тянуть весь датасет

    Returns
    -------
    tuple[Path, Path]
        Пути к файлам среза и манифеста
    """
    setup_logging()
    streaming = strategy == "stream"
    if streaming and ":" in split:
        raise ValueError(
            "При strategy='stream' нельзя использовать split с двоеточием (train[:N]). "
            "Укажи split без слайса или выбери strategy='download'."
        )
    if streaming:
        effective_split = split  # только 'train'
    else:
        effective_split = split if ":" in split else f"{split}[:{max(n * 5, n)}]"
    output_dir.mkdir(parents=True, exist_ok=True)
    slice_path = output_dir / "ruslawod_slice.jsonl.gz"
    manifest_path = output_dir / "slice_manifest.json"

    def _match_filters(raw: Mapping[str, Any]) -> bool:
        if not filters:
            return True
        for key, allowed in filters.items():
            if raw.get(key) not in allowed:
                return False
        return True

    logger.info(
        "HF slice: dataset=%s split=%s strategy=%s n=%s data_files=%s filters=%s",
        dataset_id,
        effective_split,
        strategy,
        n,
        data_files,
        filters,
    )

    normalized_rows = None
    for attempt in range(1, max_retries + 1):
        try:
            logger.info("Перед load_dataset(...)")
            ds = load_dataset(
                dataset_id,
                split=effective_split,
                streaming=streaming,
                download_mode="reuse_cache_if_exists",
                data_files=data_files,
            )
            logger.info("После load_dataset(...)")
            shuffled = (
                ds.shuffle(seed=seed, buffer_size=buffer_size)
                if streaming
                else ds.shuffle(seed=seed)
            )
            raw_iter = (row for row in shuffled if _match_filters(row))
            normalized_rows = []
            for idx, raw in enumerate(raw_iter, start=1):
                normalized_rows.append(asdict(normalize_ruslawod_record(raw)))
                if idx % 100 == 0:
                    logger.info("HF slice прогресс: %s/%s записей", idx, n)
                if idx >= n:
                    break
            break
        except Exception as exc:
            msg = str(exc).lower()
            if isinstance(exc, ValueError) and ("bad split" in msg or "available splits" in msg):
                raise
            logger.exception(
                "Попытка %s/%s построения HF-среза провалилась", attempt, max_retries, exc_info=exc
            )
            if attempt == max_retries:
                msg = (
                    "HuggingFace Hub недоступен или кэш пуст. "
                    "Проверь интернет/VPN или задай HF_HOME и повтори."
                )
                logger.error(
                    "%s (dataset=%s split=%s strategy=%s data_files=%s)",
                    msg,
                    dataset_id,
                    effective_split,
                    strategy,
                    data_files,
                )
                raise RuntimeError(msg) from exc
            delay = backoff_base ** (attempt - 1)
            logger.info("Повторная попытка через %.1f секунд", delay)
            time.sleep(delay)

    assert normalized_rows is not None
    if not normalized_rows:
        raise ValueError("Срез HF пуст: фильтры слишком жёсткие или данных нет")

    _write_jsonl_gzip(slice_path, normalized_rows)
    logger.info(
        "HF slice готов: %s записей → %s (manifest %s)",
        len(normalized_rows),
        slice_path,
        manifest_path,
    )

    manifest = {
        "dataset_id": dataset_id,
        "split": effective_split,
        "seed": seed,
        "n": n,
        "buffer_size": buffer_size,
        "strategy": strategy,
        "max_retries": max_retries,
        "hf_home": os.environ.get("HF_HOME"),
        "data_files": data_files,
        "filters": dict(filters) if filters else None,
        "output_file": slice_path.name,
        "created_at": datetime.now(UTC).isoformat(),
        "fields": [
            "pravogovruNd",
            "doc_typeIPS",
            "headingIPS",
            "docdateIPS",
            "docNumberIPS",
            "statusIPS",
            "textIPS",
            "keywordsByIPS",
            "classifierByIPS",
        ],
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))

    return slice_path, manifest_path


def main() -> None:
    """
    CLI для построения детерминированного среза RusLawOD из HuggingFace.
    """
    parser = argparse.ArgumentParser(description="Build deterministic RusLawOD slice (HF).")
    parser.add_argument("--dataset-id", default=os.environ.get("DATASET_ID", "irlspbru/RusLawOD"))
    parser.add_argument("--split", default=os.environ.get("SPLIT", "train"))
    parser.add_argument("--n", type=int, default=int(os.environ.get("N", 1000)))
    parser.add_argument("--seed", type=int, default=int(os.environ.get("SEED", 42)))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(os.environ.get("OUTPUT_DIR", "data/raw")),
        help="Директория для сохранения среза и манифеста",
    )
    parser.add_argument(
        "--buffer-size",
        type=int,
        default=10_000,
        help="Буфер для стримингового shuffle",
    )
    parser.add_argument(
        "--strategy",
        choices=["download", "stream"],
        default=os.environ.get("STRATEGY", "stream"),
        help="download — сначала скачивает в кэш HF; stream — стриминговая выдача",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Количество попыток при сетевых сбоях",
    )
    parser.add_argument(
        "--backoff-base",
        type=float,
        default=1.5,
        help="Базовый множитель экспоненциальной задержки",
    )
    parser.add_argument(
        "--data-files",
        default=os.environ.get("DATA_FILES")
    )

    parser.add_argument(
        "--status",
        nargs="+",
        default=os.environ.get("STATUS_FILTER"),
        help="Фильтр по statusIPS (через пробел или запятую)",
    )
    parser.add_argument(
        "--doc-type",
        nargs="+",
        default=os.environ.get("DOC_TYPE_FILTER"),
        help="Фильтр по doc_typeIPS (через пробел или запятую)",
    )

    args = parser.parse_args()
    data_files = parse_data_files(args.data_files)
    filters = {}

    def _split_csv_tokens(tokens: list[str]) -> list[str]:
        out = []
        for t in tokens:
            out.extend([p.strip() for p in t.split(",") if p.strip()])
        return out


    if args.status:
        filters["statusIPS"] = _split_csv_tokens(args.status)
    if args.doc_type:
        filters["doc_typeIPS"] = (
            args.doc_type.split(",") if isinstance(args.doc_type, str) else list(args.doc_type)
        )
    if not filters:
        filters = None

    build_hf_slice(
        dataset_id=args.dataset_id,
        split=args.split,
        n=args.n,
        seed=args.seed,
        output_dir=args.output_dir,
        buffer_size=args.buffer_size,
        strategy=args.strategy,
        max_retries=args.max_retries,
        backoff_base=args.backoff_base,
        data_files=data_files,
        filters=filters,
    )


if __name__ == "__main__":
    main()
