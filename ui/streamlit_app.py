from __future__ import annotations

import os
import time
from collections import defaultdict
from pathlib import Path

# Безопасность OpenMP/torch/faiss (часто требуется на macOS)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import streamlit as st

from rag.logging import setup_logging
from rag.ui.service import (
    build_retrievers_for_ui,
    chunk_loaded_documents,
    load_documents,
    run_retrieval,
)

setup_logging()

SLICE_PATH_DEFAULT = Path("data/raw/ruslawod_slice.jsonl.gz")
CHUNK_SIZE_DEFAULT = int(os.getenv("CHUNK_SIZE_CHARS", "1024"))
OVERLAP_DEFAULT = int(os.getenv("OVERLAP_CHARS", "64"))
MIN_CHUNK_CHARS_DEFAULT = int(os.getenv("MIN_CHUNK_CHARS", "50"))
EMBED_MODEL_DEFAULT = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-small")
EMBED_BATCH_DEFAULT = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))


@st.cache_data(show_spinner=False)
def _load_docs_cached(slice_path: Path):
    docs = load_documents(slice_path)
    return docs, len(docs)


@st.cache_data(show_spinner=False)
def _chunk_docs_cached(docs, chunk_size: int, overlap: int):
    chunks = chunk_loaded_documents(docs, chunk_size_chars=chunk_size, overlap_chars=overlap)
    return chunks, len(chunks)


@st.cache_resource(show_spinner=False)
def _build_retrievers_cached(
    chunks,
    retriever_names: tuple[str, ...],
    embedding_model: str | None,
    embedding_batch_size: int,
    min_chunk_chars: int,
):
    retrievers = build_retrievers_for_ui(
        chunks,
        retriever_names=retriever_names,
        embedding_model=embedding_model,
        embedding_batch_size=embedding_batch_size,
        min_chunk_chars=min_chunk_chars,
    )
    return retrievers


def main() -> None:
    st.set_page_config(page_title="RusLawOD Retrieval Explorer", layout="wide")
    st.title("RusLawOD Retrieval Explorer")
    st.caption("Офлайн исследование BM25 / Dense / Hybrid по локальному срезу RusLawOD")

    slice_path = SLICE_PATH_DEFAULT
    if not slice_path.exists():
        st.error("Файл среза не найден. Выполните `make hf-slice` и повторите.")
        return

    with st.sidebar:
        st.header("Параметры")
        retriever = st.selectbox("Retriever", ["bm25", "dense", "hybrid"], index=0)
        k = st.slider("Top-k", min_value=1, max_value=50, value=10, step=1)
        chunk_size = st.number_input(
            "Chunk size (chars)", value=CHUNK_SIZE_DEFAULT, min_value=128, max_value=4096, step=64
        )
        overlap = st.number_input(
            "Overlap (chars)", value=OVERLAP_DEFAULT, min_value=0, max_value=512, step=16
        )
        min_chunk_chars = st.number_input(
            "Min chunk length", value=MIN_CHUNK_CHARS_DEFAULT, min_value=0, max_value=512, step=10
        )
        embed_model = st.text_input("Embedding model", value=EMBED_MODEL_DEFAULT or "")
        embed_batch = st.number_input(
            "Embedding batch size", value=EMBED_BATCH_DEFAULT, min_value=1, max_value=128, step=1
        )
        st.markdown("Нажмите Search в основной области для запроса.")

    query = st.text_input("Введите запрос", value="налог", help="Введите текст запроса на русском")
    search = st.button("Search", type="primary")

    docs, doc_count = _load_docs_cached(slice_path)
    chunks, chunk_count = _chunk_docs_cached(docs, chunk_size, overlap)
    try:
        retrievers = _build_retrievers_cached(
            chunks,
            retriever_names=(retriever,),
            embedding_model=embed_model or None,
            embedding_batch_size=int(embed_batch),
            min_chunk_chars=int(min_chunk_chars),
        )
    except Exception as exc:  # pragma: no cover - UI guard
        st.error(f"Не удалось построить retrievers: {exc}")
        return

    st.info(
        f"Документов: {doc_count} • Чанков: {chunk_count} • Chunk={chunk_size} • Overlap={overlap}"
    )

    if search and query.strip():
        start = time.time()
        results = run_retrieval(query, retriever, k, retrievers=retrievers)
        duration = time.time() - start
        st.success(f"Найдено {len(results)} чанков за {duration:.2f} c")

        grouped: dict[str, list] = defaultdict(list)
        for r in results:
            grouped[r.doc_id].append(r)

        st.markdown(f"Уникальных документов: {len(grouped)}")

        for doc_id, items in grouped.items():
            items_sorted = sorted(items, key=lambda x: x.score, reverse=True)
            meta = items_sorted[0].metadata or {}
            title = meta.get("title") or meta.get("heading") or meta.get("headingIPS") or "—"
            doc_type = meta.get("doc_type") or meta.get("doc_typeIPS") or "—"
            docdate = meta.get("docdate") or meta.get("docdateIPS") or "—"
            header = f"{doc_id} | {doc_type} | {docdate} | {title}"
            with st.expander(header, expanded=False):
                for item in items_sorted:
                    snippet = (item.text or "")[:500]
                    st.markdown(
                        f"**score:** {item.score:.4f} • **chunk_id:** {item.chunk_id}  \n"
                        f"**doc_type:** {doc_type} • **docdate:** {docdate}"
                    )
                    with st.expander("Текст чанка", expanded=False):
                        st.write(snippet)
    elif search:
        st.warning("Введите непустой запрос")


if __name__ == "__main__":
    main()
