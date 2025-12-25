from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
from huggingface_hub import snapshot_download
from sentence_transformers import SentenceTransformer

from rag.embeddings.config import get_default_model

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "intfloat/multilingual-e5-base"


def _get_device() -> str:
    """
    Возвращает устройство для инференса sentence-transformers (CPU по умолчанию).

    Returns
    -------
    str
        Устройство, передаваемое в SentenceTransformer
    """
    return "cpu"


def _cache_dir() -> Path:
    """
    Папка для кэша эмбеддингов в пределах репозитория или ENV.
    """
    env_cache = os.getenv("SENTENCE_TRANSFORMERS_HOME") or os.getenv("TRANSFORMERS_CACHE")
    if env_cache:
        return Path(env_cache)
    return Path("data/cache/embeddings")


@lru_cache(maxsize=4)
def _load_model(model_name: str) -> SentenceTransformer:
    """
    Загружает и кеширует модель sentence-transformers.

    Parameters
    ----------
    model_name : str
        Имя модели

    Returns
    -------
    SentenceTransformer
        Загруженная модель
    """
    cache_path = _cache_dir()
    cache_path.mkdir(parents=True, exist_ok=True)
    logger.info("Загрузка модели эмбеддингов: %s (cache=%s)", model_name, cache_path)
    try:
        return SentenceTransformer(
            model_name,
            device=_get_device(),
            cache_folder=str(cache_path),
            local_files_only=True,
        )
    except Exception as exc:
        allow_download = os.getenv("EMBEDDING_ALLOW_DOWNLOAD", "").lower() == "true"
        if not allow_download:
            raise RuntimeError(
                f"Модель {model_name} не найдена локально. "
                "Скачайте её заранее при доступе к сети или установите EMBEDDING_ALLOW_DOWNLOAD=true."
            ) from exc
        # разрешаем однократную загрузку в кэш при явном разрешении
        snapshot_download(repo_id=model_name, cache_dir=cache_path)
        return SentenceTransformer(
            model_name,
            device=_get_device(),
            cache_folder=str(cache_path),
            local_files_only=True,
        )


class SentenceTransformerEmbeddings:
    """
    Обёртка над SentenceTransformer для расчёта эмбеддингов запросов и документов.

    Поддерживает e5-префиксы (query:/passage:), батчинг и L2-нормализацию.
    """

    def __init__(
        self,
        model_name: str | None = None,
        *,
        batch_size: int | None = None,
        normalize: bool = True,
    ) -> None:
        """
        Инициализация эмбеддера.

        Parameters
        ----------
        model_name : str, optional
            Имя модели sentence-transformers (по умолчанию intfloat/multilingual-e5-base)
        batch_size : int, optional
            Размер батча при кодировании (по умолчанию 32 или EMBEDDING_BATCH_SIZE)
        normalize : bool, optional
            Применять ли L2-нормализацию к вектору
        """
        self.model_name = model_name or os.getenv("EMBEDDING_MODEL") or get_default_model()
        self.batch_size = batch_size or int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))
        self.normalize = normalize
        self._model: SentenceTransformer | None = None

    @property
    def model(self) -> SentenceTransformer:
        """
        Lazy-загрузка модели с кешированием.

        Returns
        -------
        SentenceTransformer
            Экземпляр модели
        """
        if self._model is None:
            self._model = _load_model(self.model_name)
        return self._model

    def _encode(self, texts: list[str]) -> np.ndarray:
        vectors = self.model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
            show_progress_bar=False,
        )
        return vectors.astype(np.float32)

    def encode_passages(self, texts: list[str]) -> np.ndarray:
        """
        Кодирование чанков/документов с e5-префиксом passage.

        Parameters
        ----------
        texts : list[str]
            Список текстов

        Returns
        -------
        np.ndarray
            Матрица эмбеддингов
        """
        prefixed = [f"passage: {t}" for t in texts]
        return self._encode(prefixed)

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        """
        Кодирование поисковых запросов с e5-префиксом query.

        Parameters
        ----------
        texts : list[str]
            Список запросов

        Returns
        -------
        np.ndarray
            Матрица эмбеддингов
        """
        prefixed = [f"query: {t}" for t in texts]
        return self._encode(prefixed)
