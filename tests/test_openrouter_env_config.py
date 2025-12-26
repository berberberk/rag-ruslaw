from rag.llm.openrouter import get_llm_env_config


def test_get_llm_env_config_reads_env(monkeypatch):
    monkeypatch.setenv("RAG_OPENROUTER_API_KEY", "key")
    monkeypatch.setenv("RAG_OPENROUTER_MODEL", "model-a")
    monkeypatch.setenv("RAG_OPENROUTER_MODELS", "model-a,model-b")

    cfg = get_llm_env_config()
    assert cfg["api_key"] == "key"
    assert cfg["model"] == "model-a"
    assert cfg["models"] == ["model-a", "model-b"]
