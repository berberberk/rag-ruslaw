from __future__ import annotations

import argparse
import json
import logging
import re
from collections.abc import Iterable
from pathlib import Path

from rag.cli.retrieve_cli import build_retrievers, load_and_prepare_chunks
from rag.index.contracts import RetrievedChunk
from rag.ingest.load_dataset import load_documents_from_slice
from rag.ingest.schema import Document
from rag.logging import setup_logging

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "mistralai/Mixtral-8x7B-Instruct-v0.1"


def _extract_json(text: str):
    """
    Попытка извлечь JSON из ответа модели.
    """
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        raise ValueError("LLM не вернул JSON массив")
    return json.loads(match.group(0))


class OpenRouterLLM:
    """
    Клиент OpenRouter для генерации вопросов.
    """

    def __init__(self, api_key: str, model: str = DEFAULT_MODEL):
        self.api_key = api_key
        self.model = model

    def _post(self, prompt: str) -> str:
        import requests

        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        }
        resp = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]

    def generate_questions(self, doc: Document, n_questions: int) -> list[dict]:
        prompt = (
            f"Сгенерируй {n_questions} вопросов по документу.\n"
            f"Заголовок: {doc.title}\n"
            f"Текст: {doc.text[:1000]}\n"
        )
        text = self._post(prompt)
        try:
            return _extract_json(text)
        except Exception:
            text = self._post(prompt + "\nВЫВЕДИ СТРОГО JSON список без текста.")
            return _extract_json(text)

    def generate_negative(self, n_questions: int) -> list[dict]:
        prompt = (
            f"Сгенерируй {n_questions} правдоподобных юридических вопросов, которые отсутствуют в датасете. "
            f"Выведи JSON массив объектов с полями question,type='negative',reason,keywords_used."
        )
        text = self._post(prompt)
        try:
            return _extract_json(text)
        except Exception:
            text = self._post(prompt + "\nТолько JSON массив, без текста.")
            return _extract_json(text)


SYSTEM_PROMPT = (
    "Ты помощник, генерирующий вопросы для evalset. Отвечай строго JSON массивом объектов:\n"
    '[{"question":"...","type":"definition|fact|negative","reason":"...","keywords_used":[...]}, ...]'
)


def select_documents(docs: Iterable[Document], n_docs: int, seed: int = 42) -> list[Document]:
    """
    Детерминированный выбор документов для LLM генерации.

    Parameters
    ----------
    docs : Iterable[Document]
        Коллекция документов
    n_docs : int
        Сколько документов выбрать
    seed : int, optional
        Зерно для выбора (на случай рандома)

    Returns
    -------
    list[Document]
        Отсортированный список выбранных документов
    """
    docs = [d for d in docs if d.text]
    docs.sort(
        key=lambda d: (d.metadata.get("doc_type") or "", d.metadata.get("docdate") or "", d.doc_id)
    )
    return docs[:n_docs]


def load_and_prepare_documents(slice_path: Path) -> list[Document]:
    """
    Загрузка нормализованных документов для LLM-генерации.

    Parameters
    ----------
    slice_path : Path
        Путь до нормализованного среза

    Returns
    -------
    list[Document]
        Список документов
    """
    return load_documents_from_slice(slice_path)


def _autolink_chunks(
    question: str,
    doc_id: str,
    retriever,
    top_k: int,
) -> list[str]:
    results: list[RetrievedChunk] = retriever.retrieve(question, k=top_k)
    filtered = [r.chunk_id for r in results if r.doc_id == doc_id]
    return filtered[:3]


def generate_evalset_llm(
    *,
    slice_path: Path,
    output_path: Path,
    report_path: Path,
    n_docs: int,
    questions_per_doc: int,
    n_negative: int,
    retriever_name: str,
    top_k: int,
    dry_run: bool,
    llm_client=None,
    retriever=None,
    traces_dir: Path | None = None,
) -> None:
    """
    Генерация evalset с помощью LLM и автолинковкой чанков.
    """
    setup_logging()
    docs = select_documents(load_and_prepare_documents(slice_path), n_docs=n_docs)
    if not docs:
        raise ValueError("Нет документов для генерации evalset")

    if llm_client is None:
        import os

        if load_dotenv:
            load_dotenv()
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY не задан")
        model = os.getenv("OPENROUTER_MODEL", DEFAULT_MODEL)
        llm_client = OpenRouterLLM(api_key=api_key, model=model)

    # построим retriever при необходимости
    if retriever is None:
        # reuse chunking/retriever builder from CLI
        chunks = load_and_prepare_chunks(slice_path, chunk_size=512, overlap=64)
        retrievers = build_retrievers(
            chunks,
            use_dense=retriever_name in {"dense", "hybrid"},
            use_hybrid=retriever_name == "hybrid",
        )
        if retriever_name not in retrievers:
            raise ValueError(f"Retriever {retriever_name} не поддерживается")
        retriever = retrievers[retriever_name]

    traces_dir = traces_dir or Path("data/eval/llm_traces")
    traces_dir.mkdir(parents=True, exist_ok=True)

    entries: list[dict] = []
    llm_traces: list[dict] = []

    for _idx, doc in enumerate(docs, start=1):
        try:
            llm_response = llm_client.generate_questions(doc, questions_per_doc)
        except Exception as exc:  # pragma: no cover - сетевые ошибки
            logger.error("LLM ошибка для doc %s: %s", doc.doc_id, exc)
            continue

        trace_path = traces_dir / f"{doc.doc_id}.json"
        trace_path.write_text(
            json.dumps(llm_response, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        llm_traces.append({"doc_id": doc.doc_id, "trace": trace_path.name})

        for q in llm_response:
            entry = {
                "id": f"q_{len(entries) + 1:03d}",
                "type": q.get("type") or "fact",
                "question": q.get("question") or "",
                "gold_doc_ids": [doc.doc_id],
                "gold_chunk_ids": [],
                "notes": "auto-generated",
                "source": "llm",
            }
            if not entry["question"]:
                continue
            try:
                chunk_ids = _autolink_chunks(entry["question"], doc.doc_id, retriever, top_k)
                entry["gold_chunk_ids"] = chunk_ids
                if not chunk_ids:
                    entry["notes"] = "needs_review"
            except Exception as exc:
                logger.warning("Autolink failed for %s: %s", entry["id"], exc)
                entry["notes"] = "needs_review"
            entries.append(entry)

    # negative questions
    try:
        negatives = llm_client.generate_negative(n_negative)
    except Exception:  # pragma: no cover - сетевые ошибки
        negatives = []
    for q in negatives:
        entry = {
            "id": f"q_{len(entries) + 1:03d}",
            "type": "negative",
            "question": q.get("question") or "",
            "gold_doc_ids": [],
            "gold_chunk_ids": [],
            "notes": "auto-generated",
            "source": "llm",
        }
        if entry["question"]:
            entries.append(entry)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not dry_run:
        with output_path.open("w", encoding="utf-8") as f:
            for e in entries:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")

    # report
    report = {
        "total_questions": len(entries),
        "needs_review": sum(1 for e in entries if e.get("notes") == "needs_review"),
        "by_type": {},
        "retriever": retriever_name,
    }
    for e in entries:
        report["by_type"].setdefault(e["type"], 0)
        report["by_type"][e["type"]] += 1
    if not dry_run:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info("LLM evalset готов: вопросов=%s, output=%s", len(entries), output_path)


def parse_args() -> argparse.Namespace:
    """
    Разбор аргументов CLI для LLM-генерации evalset.

    Returns
    -------
    argparse.Namespace
        Аргументы
    """
    parser = argparse.ArgumentParser(description="LLM-assisted генерация evalset.")
    parser.add_argument("--slice-path", type=Path, default=Path("data/raw/ruslawod_slice.jsonl.gz"))
    parser.add_argument("--out", type=Path, default=Path("data/eval/evalset.jsonl"))
    parser.add_argument(
        "--report", type=Path, default=Path("data/eval/evalset_generation_report.json")
    )
    parser.add_argument("--n-docs", type=int, default=25)
    parser.add_argument("--questions-per-doc", type=int, default=1)
    parser.add_argument("--n-negative", type=int, default=8)
    parser.add_argument("--retriever", choices=["bm25", "dense", "hybrid"], default="bm25")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    """
    CLI точка входа.
    """
    args = parse_args()
    generate_evalset_llm(
        slice_path=args.slice_path,
        output_path=args.out,
        report_path=args.report,
        n_docs=args.n_docs,
        questions_per_doc=args.questions_per_doc,
        n_negative=args.n_negative,
        retriever_name=args.retriever,
        top_k=args.top_k,
        dry_run=args.dry_run,
    )
