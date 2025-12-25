from rag.ingest.chunking import chunk_document, chunk_documents
from rag.ingest.schema import Document


def test_chunk_count_simple_text():
    doc = Document(doc_id="doc1", title="", text="abcdef", metadata={"source": "test"})
    chunks = chunk_document(doc, chunk_size_chars=2, overlap_chars=0)

    assert len(chunks) == 3
    assert [c.text for c in chunks] == ["ab", "cd", "ef"]
    assert [c.chunk_id for c in chunks] == ["doc1_chunk_0", "doc1_chunk_1", "doc1_chunk_2"]


def test_chunk_overlap_correctness():
    doc = Document(doc_id="doc2", title="", text="abcdefgh", metadata={"source": "test"})
    chunks = chunk_document(doc, chunk_size_chars=4, overlap_chars=2)

    assert len(chunks) == 3  # "abcd", "cdef", "efgh"
    for idx in range(len(chunks) - 1):
        current = chunks[idx]
        nxt = chunks[idx + 1]
        assert nxt.char_start == current.char_end - 2
        assert current.text[-2:] == nxt.text[:2]


def test_chunk_metadata_preserved():
    doc = Document(
        doc_id="doc3",
        title="t",
        text="123456",
        metadata={"source": "RusLawOD", "doc_type": "law"},
    )

    chunks = chunk_document(doc, chunk_size_chars=3, overlap_chars=1)
    assert len(chunks) == 3

    for idx, chunk in enumerate(chunks):
        assert chunk.doc_id == doc.doc_id
        assert chunk.metadata["source"] == "RusLawOD"
        assert chunk.metadata["doc_type"] == "law"
        assert chunk.metadata["chunk_index"] == idx
        assert chunk.char_start >= 0
        assert chunk.char_end > chunk.char_start
        assert chunk.metadata["char_start"] == chunk.char_start
        assert chunk.metadata["char_end"] == chunk.char_end

    # batch helper preserves order and counts
    batch_chunks = chunk_documents([doc], chunk_size_chars=3, overlap_chars=1)
    assert [c.chunk_id for c in batch_chunks] == [c.chunk_id for c in chunks]
