from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

from rag.ingest.load_dataset import load_documents_from_slice
from rag.logging import setup_logging

logger = logging.getLogger(__name__)

HEADERS = [
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


def build_docs_catalog(slice_path: Path, output_csv: Path) -> None:
    """
    Построить каталог документов из нормализованного среза.

    Parameters
    ----------
    slice_path : Path
        Путь до нормализованного .jsonl.gz
    output_csv : Path
        Путь для записи CSV
    """
    setup_logging()
    docs = load_documents_from_slice(slice_path)
    logger.info("Формируем каталог документов: %s документов", len(docs))

    def sort_key(doc):
        docdate = doc.metadata.get("docdate") or ""
        return (docdate, doc.doc_id)

    sorted_docs = sorted(docs, key=sort_key, reverse=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HEADERS)
        writer.writeheader()
        for doc in sorted_docs:
            md = doc.metadata or {}
            writer.writerow(
                {
                    "doc_id": doc.doc_id,
                    "title": doc.title,
                    "doc_type": md.get("doc_type"),
                    "docdate": md.get("docdate"),
                    "doc_number": md.get("doc_number"),
                    "status": md.get("status"),
                    "author": md.get("author"),
                    "signed": md.get("signed"),
                    "keywords": md.get("keywords"),
                }
            )
    logger.info("Каталог записан: %s", output_csv)


def parse_args() -> argparse.Namespace:
    """
    Разбор аргументов CLI.

    Returns
    -------
    argparse.Namespace
        Аргументы
    """
    parser = argparse.ArgumentParser(description="Построение каталога документов из среза.")
    parser.add_argument(
        "--slice-path",
        type=Path,
        default=Path("data/raw/ruslawod_slice.jsonl.gz"),
        help="Путь до нормализованного gzip JSONL среза",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/eval/docs_catalog.csv"),
        help="Путь для вывода CSV каталога",
    )
    return parser.parse_args()


def main() -> None:
    """
    Точка входа CLI.
    """
    args = parse_args()
    build_docs_catalog(args.slice_path, args.output)
