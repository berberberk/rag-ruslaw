from __future__ import annotations

import logging
import os
import random
import time
from typing import Any

import requests

logger = logging.getLogger(__name__)


class OpenRouterClient:
    """
    Клиент OpenRouter с ретраями.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        *,
        base_url: str | None = None,
        timeout_s: float | None = None,
        max_retries: int = 4,
        backoff_base: float = 1.8,
    ) -> None:
        self.api_key = (
            api_key or os.getenv("RAG_OPENROUTER_API_KEY") or os.getenv("OPENROUTER_API_KEY")
        )
        self.model = model or os.getenv("RAG_OPENROUTER_MODEL")
        self.base_url = (
            base_url or os.getenv("RAG_OPENROUTER_BASE_URL") or "https://openrouter.ai/api/v1"
        )
        self.timeout_s = timeout_s or float(os.getenv("RAG_OPENROUTER_TIMEOUT_S", "45"))
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        if not self.api_key or not self.model:
            raise RuntimeError(
                "Отсутствуют ключ или модель OpenRouter (RAG_OPENROUTER_API_KEY/RAG_OPENROUTER_MODEL)."
            )

    def _post(self, url: str, headers: dict[str, str], json: dict, timeout: float):
        return requests.post(url, headers=headers, json=json, timeout=timeout)

    def chat(
        self, messages: list[dict[str, Any]], *, temperature: float = 0.0, max_tokens: int = 512
    ) -> str:
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}

        for attempt in range(1, self.max_retries + 1):
            resp = self._post(url, headers=headers, json=payload, timeout=self.timeout_s)
            if resp.status_code == 200:
                data = resp.json()
                return data["choices"][0]["message"]["content"]
            if resp.status_code in {429, 500, 502, 503, 504} and attempt < self.max_retries:
                wait = self.backoff_base * (2 ** (attempt - 1))
                wait = wait * (1 + random.random() * 0.1)
                logger.warning(
                    "OpenRouter retry %s/%s status=%s wait=%.2fs",
                    attempt,
                    self.max_retries,
                    resp.status_code,
                    wait,
                )
                time.sleep(wait)
                continue
            msg = ""
            try:
                msg = resp.json().get("error", {}).get("message")
            except Exception:  # pragma: no cover - защитный блок
                msg = getattr(resp, "text", "")
            raise RuntimeError(f"OpenRouter error status={resp.status_code} message={msg}")
        raise RuntimeError("Не удалось получить ответ от OpenRouter")
