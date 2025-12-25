from fastapi.testclient import TestClient

from rag.api.main import create_app


def test_retrieve_endpoint_returns_k_chunks_from_ruslawod_fixture():
    app = create_app(test_mode=True)
    client = TestClient(app)

    resp = client.get("/retrieve", params={"query": "что такое налог", "k": 3, "retriever": "bm25"})

    assert resp.status_code == 200
    data = resp.json()
    assert data["retriever"] == "bm25"
    assert "results" in data
    assert len(data["results"]) == 3

    doc_ids = [r["doc_id"] for r in data["results"]]
    assert "123456789" in doc_ids  # налоговый учебный документ из фикстур

    first = data["results"][0]
    assert "doc_id" in first
    assert "text" in first
    assert "score" in first
