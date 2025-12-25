from __future__ import annotations

import argparse
import csv
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev

import pandas as pd
from ragas import EvaluationDataset, evaluate  # type: ignore
from ragas.metrics.collections import context_precision, context_recall  # type: ignore

from rag.index.bm25 import BM25Retriever
from rag.index.contracts import RetrievedChunk
from rag.index.faiss_dense import DenseRetriever, build_faiss_index
from rag.index.hybrid import HybridRetriever
from rag.ingest.chunking import chunk_documents
from rag.ingest.load_dataset import load_documents_from_slice
from rag.logging import setup_logging

logger = logging.getLogger(__name__)


@dataclass
class EvaluationConfig:
    """
    Настройки запуска RAGAS оценки.
    """

    evalset_path: Path
    retrievers: list[str]
    k: int
    use_llm: bool = False
    output_dir: Path = Path("results/ragas")
    chunk_size_chars: int = 1024
    overlap_chars: int = 64


@dataclass
class RetrievalResult:
    retriever: str
    k: int
    metric: str
    scores: list[float]


def _load_evalset(evalset_path: Path) -> list[dict]:
    entries = []
    with evalset_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entries.append(json.loads(line))
    return entries


def _build_retriever(name: str, chunk_size_chars: int, overlap_chars: int):
    docs = load_documents_from_slice(Path("data/raw/ruslawod_slice.jsonl.gz"))
    chunks = chunk_documents(docs, chunk_size_chars=chunk_size_chars, overlap_chars=overlap_chars)
    bm25 = BM25Retriever.from_chunks(chunks)
    if name == "bm25":
        return bm25

    # simple dense using stub embedding (как в CLI)
    def _stub_embed(texts: list[str]):
        import numpy as np

        vecs = []
        for t in texts:
            vec = np.zeros(32, dtype=np.float32)
            vec[0] = float(len(t.split()))
            vec[1] = float(len(t))
            vecs.append(vec)
        return np.stack(vecs, axis=0)

    index, chunk_order = build_faiss_index(chunks, _stub_embed)
    dense = DenseRetriever(index=index, chunks=chunk_order, embed_func=_stub_embed)
    if name == "dense":
        return dense
    if name == "hybrid":
        return HybridRetriever(bm25_retriever=bm25, dense_retriever=dense)
    raise ValueError(f"Unknown retriever: {name}")


def _prepare_ragas_dataset(eval_entries: list[dict], retriever, k: int) -> pd.DataFrame:
    rows = []
    for entry in eval_entries:
        query = entry["question"]
        gold_docs = entry.get("gold_doc_ids") or []
        gold_chunks = entry.get("gold_chunk_ids") or []
        results: list[RetrievedChunk] = retriever.retrieve(query, k=k)
        contexts = [r.text for r in results]
        retrieved_ids = [r.chunk_id for r in results]
        # для context_* нам нужен reference context: используем gold_chunks тексты если найдены
        rows.append(
            {
                "question": query,
                "contexts": contexts,
                "retrieved_ids": retrieved_ids,
                "references": gold_chunks,
                "doc_references": gold_docs,
            }
        )
    return pd.DataFrame(rows)


def run_ragas(config: EvaluationConfig) -> list[RetrievalResult]:
    """
    Запуск RAGAS оценки для списка retriever'ов.
    """
    setup_logging()
    logger.info("RAGAS evaluation start: retrievers=%s, k=%s", config.retrievers, config.k)
    entries = _load_evalset(config.evalset_path)
    results: list[RetrievalResult] = []
    config.output_dir.mkdir(parents=True, exist_ok=True)

    for name in config.retrievers:
        logger.info("Evaluating retriever=%s", name)
        retriever = _build_retriever(name, config.chunk_size_chars, config.overlap_chars)
        df = _prepare_ragas_dataset(entries, retriever, config.k)
        dataset = EvaluationDataset.from_pandas(df)
        # Без LLM считаем только context_precision/recall
        metrics = [context_precision, context_recall]
        try:
            ragas_res = evaluate(dataset, metrics=metrics)
        except Exception as exc:  # pragma: no cover - ragas internals
            logger.error("RAGAS evaluation failed for %s: %s", name, exc)
            continue

        per_question = ragas_res.to_pandas()
        csv_path = config.output_dir / f"{name}_k{config.k}.csv"
        per_question.to_csv(csv_path, index=False)
        logger.info("Saved per-question metrics: %s", csv_path)

        for metric in metrics:
            mname = metric.name
            scores = per_question[mname].dropna().tolist()
            results.append(RetrievalResult(retriever=name, k=config.k, metric=mname, scores=scores))

    summary_csv = config.output_dir / "summary.csv"
    summary_json = config.output_dir / "summary.json"
    save_summary(results, summary_csv, summary_json)
    logger.info("RAGAS evaluation done. Summary saved to %s and %s", summary_csv, summary_json)
    return results


def save_summary(results: list[RetrievalResult], csv_path: Path, json_path: Path) -> None:
    """
    Сохранение агрегированных метрик в CSV и JSON.

    Parameters
    ----------
    results : List[RetrievalResult]
        Список результатов
    csv_path : Path
        CSV путь
    json_path : Path
        JSON путь
    """
    summary_rows = []
    for res in results:
        if res.scores:
            m = mean(res.scores)
            s = pstdev(res.scores) if len(res.scores) > 1 else 0.0
        else:
            m = 0.0
        summary_rows.append(
            {
                "retriever": res.retriever,
                "k": res.k,
                "metric_name": res.metric,
                "mean": m,
                "std": s if res.scores else None,
                "num_questions": len(res.scores),
            }
        )

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        writer.writerows(summary_rows)

    json_path.write_text(json.dumps(summary_rows, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RAGAS evaluation runner.")
    parser.add_argument("--evalset", type=Path, default=Path("data/eval/evalset.jsonl"))
    parser.add_argument("--retrievers", type=str, default="bm25,dense,hybrid")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--use-llm", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("results/ragas"))
    parser.add_argument("--chunk-size-chars", type=int, default=1024)
    parser.add_argument("--overlap-chars", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = EvaluationConfig(
        evalset_path=args.evalset,
        retrievers=[r.strip() for r in args.retrievers.split(",") if r.strip()],
        k=args.k,
        use_llm=args.use_llm,
        output_dir=args.output_dir,
        chunk_size_chars=args.chunk_size_chars,
        overlap_chars=args.overlap_chars,
    )
    run_ragas(cfg)


if __name__ == "__main__":
    main()
