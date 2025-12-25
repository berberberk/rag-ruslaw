from __future__ import annotations

from typing import Any

from rag.ingest.schema import Document


def normalize_ruslawod_record(rec: dict[str, Any]) -> Document:
    """
    Приведение записи RusLawOD к внутреннему типу `Document`.

    Parameters
    ----------
    rec : Dict[str, Any]
        Сырая запись RusLawOD (поля pravogovruNd, headingIPS, textIPS и др.)

    Returns
    -------
    Document
        Нормализованное представление с выделенными метаданными
    """
    doc_id = str(
        rec.get("pravogovruNd") or rec.get("docNumberIPS") or rec.get("id") or "unknown"
    ).strip()
    title = (rec.get("headingIPS") or "").strip()
    text = (rec.get("textIPS") or "").strip()

    metadata = {
        "source": "RusLawOD",
        "doc_type": rec.get("doc_typeIPS"),
        "docdate": rec.get("docdateIPS"),
        "doc_number": rec.get("docNumberIPS"),
        "author": rec.get("doc_author_normal_formIPS"),
        "signed": rec.get("signedIPS"),
        "status": rec.get("statusIPS"),
        "keywords": rec.get("keywordsByIPS"),
        "classifier": rec.get("classifierByIPS"),
        "actual_datetime": rec.get("actual_datetimeIPS"),
        "actual_datetime_human": rec.get("actual_datetime_humanIPS"),
        "raw_id": rec.get("pravogovruNd"),
    }

    return Document(doc_id=doc_id, title=title, text=text, metadata=metadata)
