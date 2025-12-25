from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path
from time import monotonic

from rag.cli.retrieve_cli import build_retrievers
from rag.embeddings.config import get_available_models, get_default_model, normalize_model_id
from rag.embeddings.st import SentenceTransformerEmbeddings
from rag.index.contracts import RetrievedChunk
from rag.ingest.chunking import chunk_documents
from rag.ingest.load_dataset import load_documents_from_slice
from rag.ingest.schema import Chunk, Document


def load_documents(slice_path: Path) -> list[Document]:
    """
    Загружает нормализованный срез RusLawOD.

    Parameters
    ----------
    slice_path : Path
        Путь до .jsonl.gz среза

    Returns
    -------
    list[Document]
        Список документов
    """
    return load_documents_from_slice(slice_path)


def chunk_loaded_documents(
    docs: Iterable[Document],
    *,
    chunk_size_chars: int,
    overlap_chars: int,
) -> list[Chunk]:
    """
    Выполняет чанкинг документов.

    Parameters
    ----------
    docs : Iterable[Document]
        Документы
    chunk_size_chars : int
        Размер чанка в символах
    overlap_chars : int
        Перекрытие

    Returns
    -------
    list[Chunk]
        Полученные чанки
    """
    return chunk_documents(docs, chunk_size_chars=chunk_size_chars, overlap_chars=overlap_chars)


def build_retrievers_for_ui(
    chunks: list[Chunk],
    *,
    retriever_names: Iterable[str],
    embedding_model: str | None,
    embedding_batch_size: int,
    min_chunk_chars: int,
):
    """
    Строит нужные retrievers на основе списка чанков.

    Parameters
    ----------
    chunks : list[Chunk]
        Корпус чанков
    retriever_names : Iterable[str]
        Список нужных retrievers (bm25/dense/hybrid)
    embedding_model : str | None
        Имя модели эмбеддингов
    embedding_batch_size : int
        Размер батча при кодировании
    min_chunk_chars : int
        Минимальная длина чанка для dense индекса

    Returns
    -------
    dict[str, object]
        Построенные retrievers
    """
    need_dense = any(name in {"dense", "hybrid"} for name in retriever_names)
    need_hybrid = "hybrid" in retriever_names
    embedder = None
    if need_dense:
        embedder = SentenceTransformerEmbeddings(
            model_name=embedding_model,
            batch_size=embedding_batch_size,
        )
    return build_retrievers(
        chunks,
        use_dense=need_dense,
        use_hybrid=need_hybrid,
        embedder=embedder,
        min_chunk_chars=min_chunk_chars,
    )


def run_retrieval(
    query: str,
    retriever_name: str,
    k: int,
    retrievers: dict[str, object],
) -> list[RetrievedChunk]:
    """
    Выполняет поиск по выбранному retriever.

    Parameters
    ----------
    query : str
        Поисковый запрос
    retriever_name : str
        Имя retriever'а
    k : int
        Размер top-k
    retrievers : dict[str, object]
        Готовые retrievers

    Returns
    -------
    list[RetrievedChunk]
        Найденные чанки
    """
    if retriever_name not in retrievers:
        raise ValueError(f"Retriever {retriever_name} не построен")
    start = monotonic()
    results = retrievers[retriever_name].retrieve(query, k=k)
    _ = monotonic() - start
    return results


def available_embedding_models() -> list[str]:
    """
    Возвращает список доступных embedding-моделей.
    """
    return get_available_models()


def default_embedding_model() -> str:
    """
    Возвращает модель по умолчанию.
    """
    return os.getenv("EMBEDDING_MODEL") or get_default_model()


def embedding_model_id(model_name: str) -> str:
    """
    Возвращает slug модели.
    """
    return normalize_model_id(model_name)
