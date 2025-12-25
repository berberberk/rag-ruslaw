from __future__ import annotations

from collections.abc import Iterable

from rag.index.contracts import RetrievedChunk


def compute_hit_at_k(results: Iterable[RetrievedChunk], gold_chunk_ids: set[str], k: int) -> float:
    """
    Hit@k по chunk_id.

    Parameters
    ----------
    results : Iterable[RetrievedChunk]
        Результаты retrieval в порядке убывания score
    gold_chunk_ids : Set[str]
        Набор референсных chunk_id
    k : int
        Порог k

    Returns
    -------
    float
        1.0 если найден хотя бы один gold chunk в top-k, иначе 0.0
    """
    top = list(results)[:k]
    if not gold_chunk_ids:
        return 0.0
    return 1.0 if any(r.chunk_id in gold_chunk_ids for r in top) else 0.0


def compute_doc_hit_at_k(
    results: Iterable[RetrievedChunk], gold_doc_ids: set[str], k: int
) -> float:
    """
    Hit@k по doc_id.

    Parameters
    ----------
    results : Iterable[RetrievedChunk]
        Результаты retrieval
    gold_doc_ids : Set[str]
        Референсные документы
    k : int
        Порог k

    Returns
    -------
    float
        1.0 если найден хотя бы один gold doc в top-k, иначе 0.0
    """
    top = list(results)[:k]
    if not gold_doc_ids:
        return 0.0
    return 1.0 if any(r.doc_id in gold_doc_ids for r in top) else 0.0


def compute_precision_at_k(
    results: Iterable[RetrievedChunk], gold_chunk_ids: set[str], k: int
) -> float | None:
    """
    Precision@k по chunk_id.

    Parameters
    ----------
    results : Iterable[RetrievedChunk]
        Результаты retrieval
    gold_chunk_ids : Set[str]
        Референсные chunk_id
    k : int
        Порог k

    Returns
    -------
    float | None
        Доля релевантных чанков в top-k или None, если gold_chunk_ids пуст
    """
    if not gold_chunk_ids:
        return None
    top = list(results)[:k]
    if not top:
        return 0.0
    hits = sum(1 for r in top if r.chunk_id in gold_chunk_ids)
    return hits / len(top)


def compute_recall_at_k(
    results: Iterable[RetrievedChunk], gold_chunk_ids: set[str], k: int
) -> float | None:
    """
    Recall@k по chunk_id.

    Parameters
    ----------
    results : Iterable[RetrievedChunk]
        Результаты retrieval
    gold_chunk_ids : Set[str]
        Референсные chunk_id
    k : int
        Порог k

    Returns
    -------
    float | None
        Доля покрытых gold чанков в top-k или None, если gold_chunk_ids пуст
    """
    if not gold_chunk_ids:
        return None
    top_ids = {r.chunk_id for r in list(results)[:k]}
    return len(top_ids & gold_chunk_ids) / len(gold_chunk_ids)
