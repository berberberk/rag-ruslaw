from __future__ import annotations

import argparse
import gzip
import json
import logging
import time
from collections.abc import Iterable
from pathlib import Path

from rag.index.bm25 import BM25Retriever
from rag.index.contracts import RetrievedChunk
from rag.index.faiss_dense import DenseRetriever, build_faiss_index
from rag.index.hybrid import HybridRetriever
from rag.ingest.chunking import chunk_documents
from rag.ingest.preprocess import normalize_ruslawod_record
from rag.ingest.schema import Chunk, Document
from rag.logging import setup_logging

logger = logging.getLogger(__name__)


def load_slice(path: Path) -> list[dict]:
    """
    Загрузка gzip JSONL с RusLawOD срезом.

    Parameters
    ----------
    path : Path
        Путь до gzipped JSONL

    Returns
    -------
    List[dict]
        Сырые записи RusLawOD
    """
    if not path.exists():
        raise FileNotFoundError(f"Slice not found: {path}")
    rows: list[dict] = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def build_retrievers(
    chunks: Iterable[Chunk],
    *,
    use_dense: bool = True,
    use_hybrid: bool = True,
) -> dict[str, object]:
    """
    Построение retriever'ов по чанкам.

    Parameters
    ----------
    chunks : Iterable[Chunk]
        Коллекция чанков
    use_dense : bool, optional
        Строить ли dense retriever
    use_hybrid : bool, optional
        Строить ли hybrid retriever

    Returns
    -------
    dict[str, object]
        Словарь retriever'ов по имени
    """
    chunk_list = list(chunks)
    retrievers: dict[str, object] = {}

    bm25 = BM25Retriever.from_chunks(chunk_list)
    retrievers["bm25"] = bm25

    if use_dense:
        # Для CLI используем очень простой embed_func на базе bm25 токенов (stub), чтобы не тянуть модель.
        # В реальной фазе можно подключить sentence-transformers; здесь детерминированно и офлайн.
        def _stub_embed(texts: list[str]):
            import numpy as np

            # Bag-of-words длины 1 (число токенов)
            return np.array([[len(t.split())] for t in texts], dtype=np.float32)

        index, chunk_order = build_faiss_index(chunk_list, _stub_embed)
        dense = DenseRetriever(index=index, chunks=chunk_order, embed_func=_stub_embed)
        retrievers["dense"] = dense

    if use_hybrid and "dense" in retrievers:
        hybrid = HybridRetriever(bm25_retriever=bm25, dense_retriever=retrievers["dense"])
        retrievers["hybrid"] = hybrid

    return retrievers


def load_and_prepare_chunks(slice_path: Path, chunk_size: int, overlap: int) -> list[Chunk]:
    """
    Загрузка среза, нормализация и чанкинг.

    Parameters
    ----------
    slice_path : Path
        Путь до gzipped JSONL
    chunk_size : int
        Размер чанка
    overlap : int
        Перекрытие чанков

    Returns
    -------
    List[Chunk]
        Список чанков
    """
    raw_rows = load_slice(slice_path)
    docs: list[Document] = [normalize_ruslawod_record(r) for r in raw_rows]
    chunks = chunk_documents(docs, chunk_size=chunk_size, overlap=overlap)
    return chunks


def run_retrieval(args: argparse.Namespace) -> None:
    """
    Запуск retrieval по CLI аргументам.

    Parameters
    ----------
    args : argparse.Namespace
        Аргументы CLI
    """
    setup_logging()
    start = time.time()
    slice_path = Path(args.slice_path)
    logger.info("Чтение среза из %s", slice_path)
    chunks = load_and_prepare_chunks(slice_path, args.chunk_size, args.overlap)
    logger.info("Готово: документов=%s, чанков=%s", len(set(c.doc_id for c in chunks)), len(chunks))

    retrievers = build_retrievers(chunks, use_dense=True, use_hybrid=True)
    if args.retriever not in retrievers:
        raise ValueError(f"Retriever '{args.retriever}' не поддерживается")

    retriever = retrievers[args.retriever]
    logger.info("Retrieval: retriever=%s, query='%s', k=%s", args.retriever, args.query, args.k)
    results: list[RetrievedChunk] = retriever.retrieve(args.query, k=args.k)
    elapsed = time.time() - start
    logger.info("Retrieval завершён за %.2fs, найдено %s", elapsed, len(results))

    for rank, item in enumerate(results, start=1):
        heading = item.metadata.get("headingIPS") or item.metadata.get("heading") or ""
        docdate = item.metadata.get("docdate") or item.metadata.get("docdateIPS") or ""
        doc_type = item.metadata.get("doc_type") or item.metadata.get("doc_typeIPS") or ""
        snippet = (item.text or "")[:200].replace("\n", " ")
        print(
            f"{rank:02d}. score={item.score:.4f} doc_id={item.doc_id} chunk_id={item.chunk_id} "
            f"doc_type={doc_type} docdate={docdate} heading={heading} text='{snippet}'"
        )


def parse_args() -> argparse.Namespace:
    """
    Разбор аргументов CLI для retrieval.

    Returns
    -------
    argparse.Namespace
        Аргументы
    """
    parser = argparse.ArgumentParser(description="Offline retrieval CLI (bm25/dense/hybrid).")
    parser.add_argument("--query", required=True, help="Запрос для поиска")
    parser.add_argument("--k", type=int, default=10, help="Сколько чанков вернуть")
    parser.add_argument(
        "--retriever",
        choices=["bm25", "dense", "hybrid"],
        default="bm25",
        help="Какой retriever использовать",
    )
    parser.add_argument(
        "--slice-path",
        type=Path,
        default=Path("data/raw/ruslawod_slice.jsonl.gz"),
        help="Путь до gzipped JSONL среза",
    )
    parser.add_argument("--chunk-size", type=int, default=512, help="Размер чанка в символах")
    parser.add_argument("--overlap", type=int, default=64, help="Перекрытие чанков в символах")
    return parser.parse_args()


def main() -> None:
    """
    CLI точка входа.
    """
    args = parse_args()
    run_retrieval(args)


if __name__ == "__main__":
    main()
