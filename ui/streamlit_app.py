from __future__ import annotations

import os
import time
from collections import defaultdict
from pathlib import Path

# Безопасность OpenMP/torch/faiss (часто требуется на macOS)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from rag.eval.metrics_loader import load_metrics_from_results
from rag.llm.openrouter import OpenRouterClient, get_llm_env_config
from rag.logging import setup_logging
from rag.rag_pipeline.generate import rag_answer
from rag.ui.service import (
    available_embedding_models,
    build_retrievers_for_ui,
    chunk_loaded_documents,
    default_embedding_model,
    load_documents,
    run_retrieval,
)

load_dotenv()
setup_logging()

SLICE_PATH_DEFAULT = Path("data/raw/ruslawod_slice.jsonl.gz")
CHUNK_SIZE_DEFAULT = int(os.getenv("CHUNK_SIZE_CHARS", "1024"))
OVERLAP_DEFAULT = int(os.getenv("OVERLAP_CHARS", "64"))
MIN_CHUNK_CHARS_DEFAULT = int(os.getenv("MIN_CHUNK_CHARS", "50"))
EMBED_MODEL_DEFAULT = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-small")
EMBED_BATCH_DEFAULT = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))
METRICS_DIR_DEFAULT = Path("results")


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


@st.cache_data(show_spinner=False)
def _load_metrics_cached(base_dir: Path):
    df = load_metrics_from_results(base_dir)
    return df


@st.cache_resource(show_spinner=False)
def _llm_client_cached(model_name: str):
    return OpenRouterClient(model=model_name)


def main() -> None:
    st.set_page_config(page_title="RusLawOD Retrieval Explorer", layout="wide")
    st.title("RusLawOD Retrieval Explorer")
    st.caption("Офлайн исследование BM25 / Dense / Hybrid по локальному срезу RusLawOD")

    slice_path = SLICE_PATH_DEFAULT
    if not slice_path.exists():
        st.error("Файл среза не найден. Выполните `make hf-slice` и повторите.")
        return

    docs, doc_count = _load_docs_cached(slice_path)

    tab_retrieval, tab_rag, tab_diff, tab_metrics = st.tabs(
        ["Поиск (retrieval)", "RAG-ассистент", "Сравнение выдачи", "Метрики"]
    )

    with tab_retrieval:
        with st.sidebar:
            st.header("Параметры поиска")
            retriever = st.selectbox(
                "Ретривер", ["bm25", "dense", "hybrid"], index=0, key="retriever_main"
            )
            k = st.slider("Top-k", min_value=1, max_value=50, value=10, step=1, key="k_main")
            chunk_size = st.number_input(
                "Длина чанка (символы)",
                value=CHUNK_SIZE_DEFAULT,
                min_value=128,
                max_value=4096,
                step=64,
                key="chunk_size_main",
            )
            overlap = st.number_input(
                "Перекрытие (символы)",
                value=OVERLAP_DEFAULT,
                min_value=0,
                max_value=512,
                step=16,
                key="overlap_main",
            )
            min_chunk_chars = st.number_input(
                "Мин. длина чанка",
                value=MIN_CHUNK_CHARS_DEFAULT,
                min_value=0,
                max_value=512,
                step=10,
                key="min_chunk_main",
            )
            available_models = available_embedding_models()
            default_model = default_embedding_model()
            model_index = (
                available_models.index(default_model) if default_model in available_models else 0
            )
            embed_model = (
                st.selectbox(
                    "Модель эмбеддингов",
                    options=available_models,
                    index=model_index,
                    key="embed_model_main",
                )
                if retriever in {"dense", "hybrid"}
                else ""
            )
            embed_batch = st.number_input(
                "Размер batch для эмбеддингов",
                value=EMBED_BATCH_DEFAULT,
                min_value=1,
                max_value=128,
                step=1,
                key="embed_batch_main",
            )
            st.markdown("Нажмите «Поиск» в основной области для запроса.")

        query = st.text_input(
            "Введите запрос",
            value="Какая процентная ставка по налогу на доходы физических лиц?",
            help="Введите текст запроса на русском",
        )
        search = st.button("Поиск", type="primary")

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
            st.error(
                f"Не удалось построить retrievers: {exc}. "
                "Если модель не скачана, включите EMBEDDING_ALLOW_DOWNLOAD=true и повторите при наличии сети."
            )
            return

        st.info(
            f"Документов: {doc_count} • Чанков: {chunk_count} • Chunk={chunk_size} • Overlap={overlap} • Model={embed_model or 'none'}"
        )

        if search and query.strip():
            start = time.time()
            results = run_retrieval(query, retriever, k, retrievers=retrievers)
            duration = time.time() - start
            st.success(f"Найдено {len(results)} чанков за {duration:.2f} c")

            grouped: dict[str, list] = defaultdict(list)
            for r in results:
                grouped[r.doc_id].append(r)

            unique_docs = len(grouped)
            doc_counts = [len(v) for v in grouped.values()] or [0]
            coverage = unique_docs / max(1, k)
            dominance = max(doc_counts) / max(1, len(results))
            st.subheader("Быстрые метрики")
            cols = st.columns(3)
            cols[0].metric("Уникальные doc_id", unique_docs)
            cols[1].metric("Покрытие doc_id", f"{coverage:.2f}")
            cols[2].metric("Доля top doc", f"{dominance:.2f}")

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

    with tab_rag:
        st.subheader("RAG-ассистент")
        llm_cfg = get_llm_env_config()
        retriever_rag = st.selectbox(
            "Ретривер (RAG)", ["bm25", "dense", "hybrid"], index=0, key="retriever_rag"
        )
        k_rag = st.slider("Top-k (RAG)", min_value=1, max_value=20, value=5, step=1, key="k_rag")
        available_models = available_embedding_models()
        default_model = default_embedding_model()
        model_index = (
            available_models.index(default_model) if default_model in available_models else 0
        )
        embed_model_rag = (
            st.selectbox(
                "Модель эмбеддингов",
                options=available_models,
                index=model_index,
                key="embed_model_rag",
            )
            if retriever_rag in {"dense", "hybrid"}
            else None
        )
        llm_default = llm_cfg["model"]
        llm_options = llm_cfg["models"] or ([llm_default] if llm_default else [])
        llm_model = st.selectbox(
            "Модель LLM (OpenRouter)",
            options=llm_options or ["(заполните RAG_OPENROUTER_MODEL)"],
            index=0 if llm_options else 0,
        )
        question = st.text_area(
            "Вопрос", value="Как рассчитывается налог на доходы физических лиц?"
        )
        if st.button("Спросить", type="primary", key="ask_rag"):
            chunks_rag, _ = _chunk_docs_cached(docs, CHUNK_SIZE_DEFAULT, OVERLAP_DEFAULT)
            try:
                retrievers_rag = _build_retrievers_cached(
                    chunks_rag,
                    retriever_names=(retriever_rag,),
                    embedding_model=embed_model_rag,
                    embedding_batch_size=EMBED_BATCH_DEFAULT,
                    min_chunk_chars=MIN_CHUNK_CHARS_DEFAULT,
                )
            except Exception as exc:
                st.error(
                    f"Не удалось построить retrievers: {exc}. "
                    "Если модель эмбеддингов не скачана, включите EMBEDDING_ALLOW_DOWNLOAD=true при наличии сети."
                )
                return
            if retriever_rag not in retrievers_rag:
                st.error("Выбранный retriever недоступен")
                return
            if not llm_model:
                st.error("Укажите модель LLM (RAG_OPENROUTER_MODEL) и ключ в окружении.")
                return
            if not llm_cfg["api_key"]:
                st.error("Не найден API ключ (RAG_OPENROUTER_API_KEY или OPENROUTER_API_KEY).")
                return
            try:
                client = _llm_client_cached(llm_model)
            except Exception as exc:
                st.error(f"LLM недоступен: {exc}")
                return
            start = time.time()
            try:
                response = rag_answer(
                    question,
                    retriever_func=lambda q, k, emb: retrievers_rag[retriever_rag].retrieve(q, k),
                    k=k_rag,
                    embedding_model=embed_model_rag,
                    llm_client=client,
                    llm_model=llm_model,
                )
            except Exception as exc:
                st.error(f"Ошибка генерации: {exc}")
                return
            latency = time.time() - start
            st.info(f"LLM модель: {llm_model} • Время ответа: {latency:.2f} c")
            st.markdown("### Ответ")
            st.markdown(response.answer)
            if not response.citations:
                st.info("Недостаточно информации в предоставленном контексте.")
            else:
                st.markdown("### Цитаты")
                for c in response.citations:
                    st.write(
                        f"- doc_id={c['doc_id']}, title={c.get('title') or '—'}, "
                        f"date={c.get('docdate') or '—'}, type={c.get('doc_type') or '—'}, "
                        f"number={c.get('doc_number') or '—'}"
                    )
            st.markdown("### Найденные чанки")
            grouped_r = defaultdict(list)
            for ch in response.retrieved_chunks:
                grouped_r[ch["doc_id"]].append(ch)
            for doc_id, items in grouped_r.items():
                with st.expander(f"{doc_id} ({len(items)} чанков)"):
                    for item in sorted(items, key=lambda x: x["score"], reverse=True):
                        st.markdown(f"**chunk_id:** {item['chunk_id']} • score={item['score']:.4f}")
                        st.write(item["text_preview"])

    with tab_diff:
        st.subheader("Сравнение выдачи (A vs B)")
        col_a, col_b = st.columns(2)
        with col_a:
            retr_a = st.selectbox("Ретривер A", ["bm25", "dense", "hybrid"], key="retr_a")
            model_a = ""
            if retr_a in {"dense", "hybrid"}:
                model_a = st.selectbox(
                    "Модель эмбеддингов A", options=available_models, index=0, key="model_a"
                )
        with col_b:
            retr_b = st.selectbox("Ретривер B", ["bm25", "dense", "hybrid"], key="retr_b")
            model_b = ""
            if retr_b in {"dense", "hybrid"}:
                model_b = st.selectbox(
                    "Модель эмбеддингов B",
                    options=available_models,
                    index=min(1, len(available_models) - 1),
                    key="model_b",
                )
        chunk_size_diff = st.number_input(
            "Длина чанка (сравнение)",
            value=CHUNK_SIZE_DEFAULT,
            min_value=128,
            max_value=4096,
            step=64,
            key="chunk_size_diff",
        )
        overlap_diff = st.number_input(
            "Перекрытие (сравнение)",
            value=OVERLAP_DEFAULT,
            min_value=0,
            max_value=512,
            step=16,
            key="overlap_diff",
        )
        min_chunk_diff = st.number_input(
            "Мин. длина чанка (сравнение)",
            value=MIN_CHUNK_CHARS_DEFAULT,
            min_value=0,
            max_value=512,
            step=10,
            key="min_chunk_diff",
        )
        k_diff = st.slider("Top-k (сравнение)", 1, 20, 5, key="k_diff")
        query_diff = st.text_input("Запрос для сравнения", value="договор", key="query_diff")
        if st.button("Сравнить", key="compare_diff"):
            try:
                chunks_diff, _ = _chunk_docs_cached(docs, chunk_size_diff, overlap_diff)
                retrievers_a = _build_retrievers_cached(
                    chunks_diff,
                    retriever_names=(retr_a,),
                    embedding_model=model_a or None,
                    embedding_batch_size=int(embed_batch),
                    min_chunk_chars=int(min_chunk_diff),
                )
                retrievers_b = _build_retrievers_cached(
                    chunks_diff,
                    retriever_names=(retr_b,),
                    embedding_model=model_b or None,
                    embedding_batch_size=int(embed_batch),
                    min_chunk_chars=int(min_chunk_diff),
                )
                res_a = run_retrieval(query_diff, retr_a, k_diff, retrievers=retrievers_a)
                res_b = run_retrieval(query_diff, retr_b, k_diff, retrievers=retrievers_b)
            except Exception as exc:  # pragma: no cover - UI guard
                st.error(f"Не удалось выполнить сравнение: {exc}")
            else:
                docs_a = {r.doc_id for r in res_a}
                docs_b = {r.doc_id for r in res_b}
                inter = docs_a & docs_b
                only_a = docs_a - docs_b
                only_b = docs_b - docs_a
                st.write(f"Пересечение doc_id: {len(inter)}")
                st.write(f"Только в A ({len(only_a)}): {list(only_a)[:5]}")
                st.write(f"Только в B ({len(only_b)}): {list(only_b)[:5]}")

    with tab_metrics:
        st.header("Metrics Dashboard")
        metrics_df = _load_metrics_cached(METRICS_DIR_DEFAULT)
        if metrics_df.empty:
            st.warning(
                "Метрики не найдены. Сначала запустите Phase 7 (eval) и сохраните summary.csv."
            )
            return
        metrics_df["metric_name"] = metrics_df["metric_name"].apply(
            lambda n: n.replace("id_based_", "")
        )
        ks = sorted(metrics_df["k"].unique())
        retrs = sorted(metrics_df["retriever"].unique())
        metrics_available = sorted(metrics_df["metric_name"].unique())
        models_available = sorted(metrics_df["embedding_model"].unique())

        col1, col2, col3, col4 = st.columns(4)
        k_sel = col1.selectbox("k", ks, index=0)
        retr_sel = col2.multiselect("Retrievers", retrs, default=retrs)
        metric_sel = col3.multiselect(
            "Metrics",
            metrics_available,
            default=[m for m in ["context_precision", "context_recall"] if m in metrics_available],
        )
        model_sel = col4.multiselect("Embedding models", models_available, default=models_available)

        filtered = metrics_df[
            (metrics_df["k"] == k_sel)
            & (metrics_df["retriever"].isin(retr_sel))
            & (metrics_df["metric_name"].isin(metric_sel))
            & (metrics_df["embedding_model"].isin(model_sel))
        ]
        if filtered.empty:
            st.warning("Фильтр не вернул данных")
        else:
            st.subheader("Таблица метрик")
            pivot_mode = st.radio("View mode", ["By retriever", "By model"], horizontal=True)

            def _fmt(row):
                std = "" if pd.isna(row["std"]) else f" ± {row['std']:.3f}"
                return f"{row['mean']:.3f}{std}"

            if pivot_mode == "By retriever":
                table = (
                    filtered.assign(value=filtered.apply(_fmt, axis=1))
                    .pivot_table(
                        index=["retriever", "embedding_model"],
                        columns="metric_name",
                        values="value",
                        aggfunc="first",
                    )
                    .reset_index()
                )
            else:
                table = (
                    filtered.assign(value=filtered.apply(_fmt, axis=1))
                    .pivot_table(
                        index="embedding_model",
                        columns="metric_name",
                        values="value",
                        aggfunc="first",
                    )
                    .reset_index()
                )
            st.dataframe(table)

            st.subheader("Metric Comparison")
            comp_metric = st.selectbox("Metric", metrics_available, index=0, key="comp_metric")
            compare_mode = st.radio(
                "Comparison mode", ["Compare retrievers", "Compare models"], horizontal=True
            )
            if compare_mode == "Compare retrievers":
                model_fixed = st.selectbox(
                    "Embedding model", models_available, index=0, key="comp_model_fixed"
                )
                data = filtered[
                    (filtered["metric_name"] == comp_metric)
                    & (filtered["embedding_model"] == model_fixed)
                ]
                if data.empty:
                    st.warning("Нет данных для выбранных фильтров")
                else:
                    fig, ax = plt.subplots(figsize=(6, 4))
                    ax.bar(data["retriever"], data["mean"], yerr=data["std"], capsize=5)
                    ax.set_ylabel(comp_metric)
                    ax.set_xlabel("retriever")
                    ax.set_title(f"{comp_metric} @k={k_sel}, model={model_fixed}")
                    st.pyplot(fig)
            else:
                retr_fixed = st.selectbox(
                    "Retriever", [r for r in retrs if r != "bm25"], index=0, key="comp_retr"
                )
                data = filtered[
                    (filtered["metric_name"] == comp_metric) & (filtered["retriever"] == retr_fixed)
                ]
                if data.empty:
                    st.warning("Нет данных для выбранных фильтров")
                else:
                    fig, ax = plt.subplots(figsize=(6, 4))
                    ax.bar(data["embedding_model"], data["mean"], yerr=data["std"], capsize=5)
                    ax.set_ylabel(comp_metric)
                    ax.set_xlabel("embedding_model")
                    ax.set_title(f"{comp_metric} @k={k_sel}, retriever={retr_fixed}")
                    st.pyplot(fig)


if __name__ == "__main__":
    main()
