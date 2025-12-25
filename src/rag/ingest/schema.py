from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Document:
    """
    Представление исходного документа RusLawOD в нормализованном виде.

    Parameters
    ----------
    doc_id : str
        Уникальный идентификатор документа
    title : str
        Заголовок документа
    text : str
        Полный текст документа
    metadata : Dict[str, Any]
        Дополнительные атрибуты документа (тип, дата, номер и т.д.)
    """

    doc_id: str
    title: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    """
    Единица retrieval — фрагмент текста с оценкой похожести.

    Parameters
    ----------
    doc_id : str
        Идентификатор исходного документа
    chunk_id : str
        Локальный идентификатор чанка
    text : str
        Текст чанка
    metadata : Dict[str, Any]
        Метаданные, унаследованные от документа
    score : float
        Вес/оценка релевантности чанка
    char_start : int
        Начальная позиция чанка в тексте (включительно)
    char_end : int
        Конечная позиция чанка в тексте (исключительно)
    """

    doc_id: str
    chunk_id: str
    text: str
    metadata: dict[str, Any]
    score: float
    char_start: int = 0
    char_end: int = 0
