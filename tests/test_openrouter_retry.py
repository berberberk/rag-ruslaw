import pytest

from rag.llm.openrouter import OpenRouterClient


class _Resp:
    def __init__(self, status_code: int, data: dict):
        self.status_code = status_code
        self._data = data

    def json(self):
        return self._data


def test_openrouter_retries_on_429(monkeypatch):
    calls = {"n": 0}

    def _post(url, headers, json=None, timeout=None):
        calls["n"] += 1
        if calls["n"] < 3:
            return _Resp(429, {})
        return _Resp(200, {"choices": [{"message": {"content": "ok"}}]})

    sleeps = []

    def _sleep(seconds):
        sleeps.append(seconds)

    monkeypatch.setattr("time.sleep", _sleep)
    client = OpenRouterClient(api_key="x", model="m", max_retries=4, backoff_base=0.0)
    monkeypatch.setattr(client, "_post", _post)

    content = client.chat([{"role": "user", "content": "hi"}])
    assert content == "ok"
    assert calls["n"] == 3
    assert len(sleeps) == 2


def test_openrouter_no_retry_on_400(monkeypatch):
    def _post(url, headers, json=None, timeout=None):
        return _Resp(400, {"error": {"message": "bad"}})

    client = OpenRouterClient(api_key="x", model="m", max_retries=3, backoff_base=0.0)
    monkeypatch.setattr(client, "_post", _post)
    with pytest.raises(RuntimeError):
        client.chat([{"role": "user", "content": "hi"}])
