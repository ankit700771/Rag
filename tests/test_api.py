import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.generation.gemini import GenerationUnavailable
from app import main as main_module
from app.main import app
from app.ingestion.pipeline import ingest_directory

pytestmark = pytest.mark.integration


class FakeGenerator:
    def __init__(self, reply="Use BackgroundTasks [1]."):
        self.reply = reply
        self.calls = []

    def generate(self, system_instruction, prompt):
        self.calls.append((system_instruction, prompt))
        return self.reply


@pytest.fixture()
def client(engine, session):
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture()
def loaded(session, docs_tree, monkeypatch):
    monkeypatch.setattr(settings, "docs_dir", docs_tree)
    monkeypatch.setattr(settings, "docs_src_dir", docs_tree / "missing")
    ingest_directory(session, docs_dir=docs_tree, docs_src_dir=None)


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "db": "connected"}


def test_stats_empty(client):
    assert client.get("/stats").json() == {"documents": 0, "chunks": 0}


def test_ingest_endpoint_then_stats(client, session, docs_tree, monkeypatch):
    monkeypatch.setattr(settings, "docs_dir", docs_tree)
    monkeypatch.setattr(settings, "docs_src_dir", docs_tree / "missing")
    response = client.post("/ingest")
    assert response.status_code == 200
    body = response.json()
    assert body["documents"] == 3
    stats = client.get("/stats").json()
    assert stats == {"documents": 3, "chunks": body["chunks"]}


def test_search_endpoint(client, loaded):
    response = client.post(
        "/search", json={"query": "serve static files", "top_k": 2, "mode": "hybrid", "rerank": True}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["results"][0]["source"] == "tutorial/static-files.md"
    assert len(body["results"]) == 2
    assert body["latency_ms"] > 0


def test_search_validation_errors(client):
    assert client.post("/search", json={"query": ""}).status_code == 422
    assert client.post("/search", json={"query": "x", "top_k": 0}).status_code == 422
    assert client.post("/search", json={"query": "x", "mode": "bogus"}).status_code == 422
    assert client.post("/search", json={}).status_code == 422


def test_ask_returns_answer_and_citations(client, loaded, monkeypatch):
    generator = FakeGenerator("Use BackgroundTasks [1].")
    monkeypatch.setattr(main_module, "get_generator", lambda: generator)
    response = client.post("/ask", json={"question": "How do I run work after the response is sent?"})
    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Use BackgroundTasks [1]."
    assert body["sources"][0]["number"] == 1
    assert body["sources"][0]["cited"] is True
    assert body["sources"][0]["source"] == "tutorial/background-tasks.md"
    assert all(s["cited"] is False for s in body["sources"][1:])
    system_instruction, prompt = generator.calls[0]
    assert "Context passages" in prompt
    assert "How do I run work after the response is sent?" in prompt


def test_ask_with_empty_database_skips_generation(client, session, monkeypatch):
    generator = FakeGenerator()
    monkeypatch.setattr(main_module, "get_generator", lambda: generator)
    response = client.post("/ask", json={"question": "anything at all"})
    assert response.status_code == 200
    assert "don't know" in response.json()["answer"]
    assert response.json()["sources"] == []
    assert generator.calls == []


def test_ask_without_api_key_returns_503(client, loaded, monkeypatch):
    monkeypatch.setattr(settings, "gemini_api_key", None)
    response = client.post("/ask", json={"question": "How do I enable CORS?"})
    assert response.status_code == 503
    assert "GEMINI_API_KEY" in response.json()["detail"]


def test_ask_validation(client):
    assert client.post("/ask", json={"question": ""}).status_code == 422
    assert client.post("/ask", json={"question": "x", "top_k": 50}).status_code == 422


def test_ask_returns_503_when_generation_fails(client, loaded, monkeypatch):
    class Failing:
        def generate(self, system_instruction, prompt):
            raise GenerationUnavailable("Gemini request failed: 503 UNAVAILABLE")

    monkeypatch.setattr(main_module, "get_generator", lambda: Failing())
    response = client.post("/ask", json={"question": "How do I enable CORS?"})
    assert response.status_code == 503
    assert "503" in response.json()["detail"]
