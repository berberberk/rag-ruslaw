from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path

from rag.index.bm25 import BM25Retriever
from rag.index.contracts import RetrievedChunk
from rag.ingest.chunking import chunk_documents
from rag.ingest.load_dataset import load_documents_from_slice
from rag.logging import setup_logging

logger = logging.getLogger(__name__)

NEGATIVE_TERMS = [
    "криптовалюта",
    "блокчейн",
    "квантовый",
    "марсоход",
    "биочип",
    "метавселенная",
    "искусственный интеллект",
    "робот-пылесос",
]


def _pick_keyword(metadata: dict) -> str | None:
    keywords = metadata.get("keywords")
    if isinstance(keywords, str):
        parts = [k.strip() for k in keywords.split(",") if k.strip()]
        return parts[0] if parts else None
    if isinstance(keywords, list) and keywords:
        return str(keywords[0])
    return None


def _generate_questions_from_row(row: dict) -> list[dict]:
    doc_id = row["doc_id"]
    title = row.get("title") or ""
    keyword = _pick_keyword({"keywords": row.get("keywords")})
    questions = []

    term = keyword or title
    if term:
        questions.append(
            {
                "type": "definition",
                "question": f"Что такое {term}?",
                "gold_doc_ids": [doc_id],
            }
        )
    questions.append(
        {
            "type": "fact",
            "question": f"О чем говорится в документе «{title or doc_id}»?",
            "gold_doc_ids": [doc_id],
        }
    )
    return questions


def generate_evalset_draft(
    catalog_csv: Path,
    output_jsonl: Path,
    target_count: int = 30,
    negative_count: int = 8,
) -> None:
    """
    Генерация чернового evalset без LLM.

    Parameters
    ----------
    catalog_csv : Path
        Путь до каталога документов
    output_jsonl : Path
        Путь для записи драфта
    target_count : int, optional
        Желаемое число положительных вопросов
    negative_count : int, optional
        Количество негативных вопросов
    """
    setup_logging()
    rows = []
    with catalog_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    logger.info("Генерация чернового evalset: документов=%s", len(rows))

    questions: list[dict] = []
    for row in rows:
        questions.extend(_generate_questions_from_row(row))
        if len(questions) >= target_count:
            break

    # добавляем negative
    for term in NEGATIVE_TERMS[:negative_count]:
        questions.append(
            {
                "type": "negative",
                "question": f"Что говорится о {term}?",
                "gold_doc_ids": [],
            }
        )

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as f:
        for idx, q in enumerate(questions, start=1):
            q_entry = {
                "id": f"q_{idx:03d}",
                "type": q["type"],
                "question": q["question"],
                "gold_doc_ids": q["gold_doc_ids"],
                "gold_chunk_ids": [],
                "notes": "auto-generated",
                "source": "auto",
            }
            f.write(json.dumps(q_entry, ensure_ascii=False) + "\n")

    logger.info("Драфт evalset записан: %s вопросов → %s", len(questions), output_jsonl)


def autolink_evalset_chunks(
    draft_path: Path,
    output_path: Path,
    *,
    retriever=None,
    top_k: int = 3,
) -> None:
    """
    Авто-присвоение chunk_id для evalset на основе retrieval.

    Parameters
    ----------
    draft_path : Path
        Путь до чернового evalset
    output_path : Path
        Путь для сохранения финального evalset
    retriever : object, optional
        Retriever, реализующий retrieve(query,k)
    top_k : int, optional
        Сколько чанков брать
    """
    setup_logging()
    entries = []
    with draft_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entries.append(json.loads(line))

    if retriever is None:
        # построим bm25 по умолчанию
        slice_path = Path("data/raw/ruslawod_slice.jsonl.gz")
        docs = load_documents_from_slice(slice_path)
        chunks = chunk_documents(docs, chunk_size=512, overlap=64)
        retriever = BM25Retriever.from_chunks(chunks)

    updated = []
    for entry in entries:
        if entry["gold_doc_ids"]:
            results: list[RetrievedChunk] = retriever.retrieve(entry["question"], k=top_k)
            entry["gold_chunk_ids"] = [r.chunk_id for r in results]
        updated.append(entry)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for entry in updated:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    logger.info("Autolink завершён: %s вопросов → %s", len(updated), output_path)


def main_generate() -> None:
    """
    CLI генерации драфта evalset.
    """
    parser = argparse.ArgumentParser(description="Сгенерировать черновой evalset без LLM.")
    parser.add_argument(
        "--catalog",
        type=Path,
        default=Path("data/eval/docs_catalog.csv"),
        help="Путь до CSV каталога документов",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/eval/evalset_draft.jsonl"),
        help="Путь для записи драфта",
    )
    parser.add_argument(
        "--target-count", type=int, default=30, help="Сколько положительных вопросов"
    )
    parser.add_argument("--negative-count", type=int, default=8, help="Сколько негативных вопросов")
    args = parser.parse_args()
    generate_evalset_draft(
        args.catalog,
        args.output,
        target_count=args.target_count,
        negative_count=args.negative_count,
    )


def main_autolink() -> None:
    """
    CLI автолинковки evalset к чанкам.
    """
    parser = argparse.ArgumentParser(description="Автопривязка evalset вопросов к чанкам.")
    parser.add_argument(
        "--draft",
        type=Path,
        default=Path("data/eval/evalset_draft.jsonl"),
        help="Путь до драфта",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/eval/evalset.jsonl"),
        help="Путь для записи финального evalset",
    )
    parser.add_argument("--top-k", type=int, default=3, help="Сколько чанков выбирать")
    args = parser.parse_args()
    autolink_evalset_chunks(args.draft, args.output, top_k=args.top_k)


def validate_evalset(path: Path) -> None:
    """
    Валидация evalset JSONL на обязательные поля.

    Parameters
    ----------
    path : Path
        Путь до evalset
    """
    required = {"id", "type", "question", "gold_doc_ids", "gold_chunk_ids", "notes", "source"}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            missing = required - set(obj.keys())
            if missing:
                raise ValueError(f"Отсутствуют поля {missing} в записи {obj.get('id')}")


def main_validate() -> None:
    """
    CLI валидации evalset.
    """
    parser = argparse.ArgumentParser(description="Валидация evalset.jsonl")
    parser.add_argument("--path", type=Path, default=Path("data/eval/evalset.jsonl"))
    args = parser.parse_args()
    validate_evalset(args.path)
    logger.info("Evalset валиден: %s", args.path)
