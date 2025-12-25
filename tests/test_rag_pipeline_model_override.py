from rag.rag_pipeline.generate import rag_answer


class _DummyClient:
    def __init__(self):
        self.model = "default"
        self.called_with = None

    def chat(self, messages, *, temperature, max_tokens):
        self.called_with = self.model
        return "ok"


def test_rag_answer_model_override():
    def _retriever(query: str, k: int, embedding_model=None):
        from rag.ingest.schema import Chunk

        return [
            Chunk(
                doc_id="d1",
                chunk_id="c1",
                text="text",
                metadata={},
                score=1.0,
            )
        ]

    client = _DummyClient()
    resp = rag_answer(
        "q",
        retriever_func=_retriever,
        k=1,
        llm_client=client,
        llm_model="override-model",
    )
    assert resp.answer == "ok"
    assert client.called_with == "override-model"
