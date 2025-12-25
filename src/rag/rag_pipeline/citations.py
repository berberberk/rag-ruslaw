from __future__ import annotations

from rag.ingest.schema import Chunk


def build_citations(chunks: list[Chunk]) -> list[dict]:
    """
    Формирует список ссылок (citations) по чанкам.

    Parameters
    ----------
    chunks : List[Chunk]
        Список чанков в порядке убывания релевантности

    Returns
    -------
    list[dict]
        Уникальные ссылки по doc_id, отсортированные по наибольшему score
    """
    best_by_doc: dict[str, Chunk] = {}
    for ch in chunks:
        prev = best_by_doc.get(ch.doc_id)
        if prev is None or ch.score > prev.score:
            best_by_doc[ch.doc_id] = ch
    ordered = sorted(best_by_doc.values(), key=lambda c: c.score, reverse=True)

    citations = []
    for ch in ordered:
        md = ch.metadata or {}
        citations.append(
            {
                "doc_id": ch.doc_id,
                "title": md.get("title") or md.get("heading") or md.get("headingIPS"),
                "docdate": md.get("docdate") or md.get("docdateIPS"),
                "doc_type": md.get("doc_type") or md.get("doc_typeIPS"),
                "doc_number": md.get("doc_number") or md.get("docNumberIPS"),
            }
        )
    return citations
