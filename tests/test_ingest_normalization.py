from rag.ingest.load_dataset import load_jsonl
from rag.ingest.preprocess import normalize_ruslawod_record
from rag.ingest.schema import Document


def test_load_jsonl_reads_all_rows_from_fixture():
    rows = load_jsonl("data/fixtures/mini_docs.jsonl")

    assert len(rows) == 3
    assert all("pravogovruNd" in row for row in rows)


def test_normalize_ruslawod_record_maps_core_fields():
    rows = load_jsonl("data/fixtures/mini_docs.jsonl")
    doc = normalize_ruslawod_record(rows[1])  # doc_id 123456789

    assert isinstance(doc, Document)
    assert doc.doc_id == "123456789"
    assert "налог" in doc.text.lower()
    assert doc.metadata["doc_type"] == "Федеральный закон"
    assert doc.metadata["doc_number"] == "1-ФЗ"
    assert doc.metadata["author"] == "Президент Российской Федерации"
    assert doc.metadata["source"] == "RusLawOD"


def test_normalize_ruslawod_record_handles_missing_fields():
    raw = {"textIPS": "Пример без идентификатора", "headingIPS": " Заголовок "}
    doc = normalize_ruslawod_record(raw)

    assert doc.doc_id == "unknown"
    assert doc.title == "Заголовок"
    assert doc.text == "Пример без идентификатора"
