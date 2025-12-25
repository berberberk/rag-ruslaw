import numpy as np

from rag.index.faiss_dense import DenseRetriever
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
    def __init__(self, mapping: dict[str, np.ndarray]):
        self.mapping = mapping

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.stack([self.mapping[t] for t in texts], axis=0)


class _FakeIndex:
    def __init__(self, chunks: list[Chunk], embed_func):
        self.chunks = chunks
        self._embed = embed_func

    def search(self, query_vec: np.ndarray, k: int):
        corpus = self._embed([c.text for c in self.chunks]).astype(np.float32)
        q = query_vec.astype(np.float32).reshape(-1)
        scores = corpus @ q
        order = np.argsort(-scores)
        top = order[:k]
        return scores[top][None, :], top.astype(np.int64)[None, :]


def test_dense_retrieve_returns_relevant_doc_id():
    chunks = [
        _make_chunk("tax", "налог это платеж"),
        _make_chunk("civil", "договор это соглашение"),
        _make_chunk("other", "другое содержание"),
    ]
    mapping = {
        "налог это платеж": np.array([1.0, 0.0], dtype=np.float32),
        "договор это соглашение": np.array([0.0, 1.0], dtype=np.float32),
        "другое содержание": np.array([0.0, 0.0], dtype=np.float32),
        "налог": np.array([0.9, 0.1], dtype=np.float32),
    }
    embedder = _FakeEmbedder(mapping)

    index = _FakeIndex(chunks, embedder.embed)
    retriever = DenseRetriever(index=index, chunks=chunks, embed_func=embedder.embed)

    results = retriever.retrieve("налог", k=2)
    doc_ids = [r.doc_id for r in results]
    assert "tax" in doc_ids


def test_dense_retrieve_is_deterministic():
    chunks = [_make_chunk("a", "foo"), _make_chunk("b", "bar")]
    mapping = {
        "foo": np.array([1.0], dtype=np.float32),
        "bar": np.array([0.0], dtype=np.float32),
        "foo bar": np.array([0.5], dtype=np.float32),
    }
    embedder = _FakeEmbedder(mapping)
    index = _FakeIndex(chunks, embedder.embed)
    retriever = DenseRetriever(index=index, chunks=chunks, embed_func=embedder.embed)

    res1 = retriever.retrieve("foo bar", k=2)
    res2 = retriever.retrieve("foo bar", k=2)
    assert res1 == res2
