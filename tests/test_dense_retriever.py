import numpy as np
import pytest

from rag.index.faiss_dense import DenseRetriever, build_faiss_index
from rag.ingest.schema import Chunk


def _make_chunk(doc_id: str, text: str, idx: int = 0) -> Chunk:
    return Chunk(
        doc_id=doc_id,
        chunk_id=f"{doc_id}_chunk_{idx}",
        text=text,
        metadata={"source": "test"},
        score=0.0,
        char_start=0,
        char_end=len(text),
    )


class _FakeEmbedder:
    def __init__(self, dim: int):
        self.dim = dim

    def encode(self, texts: list[str]) -> np.ndarray:
        vecs = []
        for t in texts:
            arr = np.zeros(self.dim, dtype=np.float32)
            arr[0] = float(len(t))
            arr[1] = float(len(t.split()))
            vecs.append(arr)
        return np.stack(vecs, axis=0)


def test_dense_retrieve_returns_relevant_doc_id():
    chunks = [
        _make_chunk("tax", "налог это платеж"),
        _make_chunk("civil", "договор это соглашение"),
        _make_chunk("other", "другое содержание"),
    ]
    embedder = _FakeEmbedder(dim=64)

    index, ordered_chunks = build_faiss_index(chunks, embedder.encode)
    retriever = DenseRetriever(
        index=index,
        chunks=ordered_chunks,
        embed_passage=embedder.encode,
        embed_query=embedder.encode,
    )

    results = retriever.retrieve("налог", k=2)
    doc_ids = [r.doc_id for r in results]
    assert "tax" in doc_ids
    assert results[0].score >= results[-1].score


def test_dense_retrieve_is_deterministic():
    chunks = [_make_chunk("a", "foo"), _make_chunk("b", "bar")]
    embedder = _FakeEmbedder(dim=64)
    index, ordered_chunks = build_faiss_index(chunks, embedder.encode)
    retriever = DenseRetriever(
        index=index,
        chunks=ordered_chunks,
        embed_passage=embedder.encode,
        embed_query=embedder.encode,
    )

    res1 = retriever.retrieve("foo bar", k=2)
    res2 = retriever.retrieve("foo bar", k=2)
    assert res1 == res2


@pytest.mark.network
@pytest.mark.slow
def test_dense_real_model_small_corpus(mocker):
    """
    Мини интеграционный тест: модель загружается один раз.
    Пропускается, если нет сети/HF кэша.
    """
    from rag.embeddings.st import SentenceTransformerEmbeddings

    try:
        embedder = SentenceTransformerEmbeddings(
            model_name="intfloat/multilingual-e5-small", batch_size=4, normalize=True
        )
    except Exception:
        pytest.skip("Модель недоступна или нет сети/кэша")

    chunks = [
        _make_chunk("tax", "Налог на доходы физических лиц регулируется законом."),
        _make_chunk("family", "Брачный договор определяет имущественные права супругов."),
        _make_chunk("other", "Произвольный текст без темы."),
    ]
    index, ordered_chunks = build_faiss_index(chunks, embedder.encode_passages)
    retriever = DenseRetriever(
        index=index,
        chunks=ordered_chunks,
        embed_passage=embedder.encode_passages,
        embed_query=embedder.encode_queries,
    )

    results = retriever.retrieve("налог", k=2)
    assert results
    assert results[0].doc_id in {"tax", "family", "other"}
