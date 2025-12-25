from rag.rag_pipeline.generate import FALLBACK_TEXT, rag_answer


class _FakeClient:
    def chat(self, messages, *, temperature, max_tokens):
        raise AssertionError("LLM не должен вызываться")


def test_rag_pipeline_fallback_without_chunks():
    def _retriever(query: str, k: int, embedding_model=None):
        return []

    resp = rag_answer("вопрос", retriever_func=_retriever, k=3, llm_client=_FakeClient())
    assert resp.answer == FALLBACK_TEXT
    assert resp.citations == []
    assert resp.retrieved_chunks == []
