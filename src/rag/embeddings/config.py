from __future__ import annotations

import os
import re

# Базовый список лёгких моделей (офлайн, CPU)
DEFAULT_MODELS = [
    "intfloat/multilingual-e5-small",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L6-v2",
]


def get_available_models() -> list[str]:
    """
    Возвращает список доступных embedding-моделей с учётом ENV.

    Returns
    -------
    list[str]
        Список имён моделей
    """
    env_val = os.getenv("EMBEDDING_MODELS")
    if env_val:
        return [m.strip() for m in env_val.split(",") if m.strip()]
    return DEFAULT_MODELS


def get_default_model() -> str:
    """
    Возвращает модель по умолчанию: ENV EMBEDDING_MODEL_DEFAULT или первая из списка.

    Returns
    -------
    str
        Имя модели по умолчанию
    """
    return os.getenv("EMBEDDING_MODEL_DEFAULT") or get_available_models()[0]


def normalize_model_id(model_name: str) -> str:
    """
    Нормализует имя модели в короткий slug.

    Parameters
    ----------
    model_name : str
        Полное имя модели

    Returns
    -------
    str
        Слаг без пробелов/слэшей
    """
    base = model_name.rsplit("/", maxsplit=1)[-1]
    slug = re.sub(r"[^A-Za-z0-9_-]+", "-", base).strip("-")
    return slug or base
