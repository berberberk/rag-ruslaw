from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from statistics import median

from rag.ingest.chunking import chunk_documents
from rag.ingest.load_dataset import load_documents_from_slice
from rag.logging import setup_logging

logger = logging.getLogger(__name__)


def compute_chunk_stats(docs, output_path: Path, chunk_size_chars: int, overlap_chars: int) -> None:
    """
    Подсчёт метрик чанков и запись в JSON.
    """
    setup_logging()
    chunks = chunk_documents(docs, chunk_size_chars=chunk_size_chars, overlap_chars=overlap_chars)
    lengths = [len(c.text) for c in chunks]
    lengths_sorted = sorted(lengths)
    p50 = median(lengths_sorted) if lengths_sorted else 0
    p90 = lengths_sorted[int(0.9 * len(lengths_sorted))] if lengths_sorted else 0

    html_count = sum(1 for c in chunks if ("<" in c.text or "img" in c.text or "http" in c.text))
    html_ratio = html_count / len(chunks) if chunks else 0

    top_long = sorted(((len(c.text), c.doc_id, c.chunk_id) for c in chunks), reverse=True)[:10]

    stats = {
        "num_docs": len({c.doc_id for c in chunks}),
        "num_chunks": len(chunks),
        "avg_chunk_len_chars": sum(lengths) / len(lengths) if lengths else 0,
        "p50_chunk_len_chars": p50,
        "p90_chunk_len_chars": p90,
        "html_ratio": html_ratio,
        "top_longest_chunks": [
            {"len": length, "doc_id": doc_id, "chunk_id": cid} for length, doc_id, cid in top_long
        ],
        "chunk_size_chars": chunk_size_chars,
        "overlap_chars": overlap_chars,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Chunk stats записаны: %s", output_path)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Подсчёт метрик чанков.")
    parser.add_argument("--slice-path", type=Path, default=Path("data/raw/ruslawod_slice.jsonl.gz"))
    parser.add_argument("--chunk-size-chars", type=int, default=2000)
    parser.add_argument("--overlap-chars", type=int, default=200)
    parser.add_argument("--output", type=Path, default=Path("data/metrics/chunk_stats.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    docs = load_documents_from_slice(args.slice_path)
    compute_chunk_stats(
        docs, args.output, chunk_size_chars=args.chunk_size_chars, overlap_chars=args.overlap_chars
    )
