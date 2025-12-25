from rag.ingest.chunking import chunk_documents
from rag.ingest.schema import Document


def _doc(text: str, metadata=None) -> Document:
    return Document(doc_id="doc1", title="t", text=text, metadata=metadata or {"source": "test"})


def test_chunking_preserves_metadata():
    doc = _doc("abcdef", {"source": "s", "doc_type": "law"})
    chunks = chunk_documents([doc], chunk_size_chars=3, overlap_chars=1)
    assert chunks
    for idx, ch in enumerate(chunks):
        assert ch.doc_id == doc.doc_id
        assert ch.metadata["source"] == "s"
        assert ch.metadata["chunk_index"] == idx


def test_chunking_nonempty_chunks():
    doc = _doc("a" * 50)
    chunks = chunk_documents([doc], chunk_size_chars=20, overlap_chars=5)
    assert all(ch.text for ch in chunks)


def test_chunking_overlap_correctness():
    text = "abcdefghij" * 10
    doc = _doc(text)
    chunks = chunk_documents([doc], chunk_size_chars=20, overlap_chars=5)
    for i in range(len(chunks) - 1):
        assert chunks[i].text[-5:] == chunks[i + 1].text[:5]


def test_structural_split_creates_more_readable_chunks():
    text = "1. Введение <img src='x'>\n\n2. Описание\nСтатья 1. Общие положения <table><tr><td>data</td></tr></table>"
    doc = _doc(text)
    chunks = chunk_documents([doc], chunk_size_chars=50, overlap_chars=0)
    assert all("<" not in ch.text for ch in chunks)
    assert any(ch.text.strip().startswith(("1.", "2.", "Статья")) for ch in chunks)
