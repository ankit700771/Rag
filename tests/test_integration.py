import pytest
from sqlalchemy import func, select

from app.ingestion.pipeline import ingest_directory
from app.models import Chunk, Document
from app.retrieval.search import search

pytestmark = pytest.mark.integration


def ingest(session, docs_tree):
    return ingest_directory(session, docs_dir=docs_tree, docs_src_dir=None)


def count(session, model):
    return session.scalar(select(func.count()).select_from(model))


def test_ingest_stores_documents_and_chunks(session, docs_tree):
    report = ingest(session, docs_tree)
    assert report.documents == 3
    assert report.chunks >= 3
    assert count(session, Document) == 3
    assert count(session, Chunk) == report.chunks


def test_excluded_pages_are_not_ingested(session, docs_tree):
    ingest(session, docs_tree)
    sources = set(session.scalars(select(Document.source)))
    assert "release-notes.md" not in sources
    assert not any(s.startswith("reference/") for s in sources)


def test_ingest_is_idempotent(session, docs_tree):
    first = ingest(session, docs_tree)
    second = ingest(session, docs_tree)
    assert first.chunks == second.chunks
    assert count(session, Document) == 3
    assert count(session, Chunk) == first.chunks


def test_reingest_replaces_changed_content(session, docs_tree):
    ingest(session, docs_tree)
    (docs_tree / "tutorial" / "cors.md").write_text(
        "# CORS\n\nCompletely new text about origins and preflight requests.\n", encoding="utf-8"
    )
    ingest(session, docs_tree)
    contents = session.scalars(
        select(Chunk.content).join(Document).where(Document.source == "tutorial/cors.md")
    ).all()
    assert len(contents) == 1
    assert "preflight" in contents[0]


def test_removed_files_are_deleted_on_full_ingest(session, docs_tree):
    ingest(session, docs_tree)
    (docs_tree / "tutorial" / "static-files.md").unlink()
    ingest(session, docs_tree)
    sources = set(session.scalars(select(Document.source)))
    assert "tutorial/static-files.md" not in sources
    assert count(session, Document) == 2


def test_limited_ingest_keeps_other_documents(session, docs_tree):
    ingest(session, docs_tree)
    ingest_directory(session, docs_dir=docs_tree, docs_src_dir=None, limit=1)
    assert count(session, Document) == 3


def test_deleting_document_cascades_to_chunks(session, docs_tree):
    ingest(session, docs_tree)
    document = session.scalars(select(Document)).first()
    session.delete(document)
    session.commit()
    orphans = session.scalar(
        select(func.count()).select_from(Chunk).where(Chunk.document_id == document.id)
    )
    assert orphans == 0


def test_embeddings_have_expected_dimension(session, docs_tree):
    ingest(session, docs_tree)
    chunk = session.scalars(select(Chunk)).first()
    assert len(chunk.embedding) == 384


def test_tsv_column_is_generated(session, docs_tree):
    ingest(session, docs_tree)
    matches = session.scalar(
        select(func.count())
        .select_from(Chunk)
        .where(Chunk.tsv.op("@@")(func.plainto_tsquery("english", "background tasks")))
    )
    assert matches >= 1


@pytest.mark.parametrize("mode", ["vector", "keyword", "hybrid"])
def test_search_finds_the_right_document(session, docs_tree, mode):
    ingest(session, docs_tree)
    hits = search(session, "how do I allow cross origin requests from a browser", top_k=3, mode=mode, use_rerank=False)
    assert hits
    if mode == "keyword":
        assert hits[0].source == "tutorial/cors.md"
    else:
        assert hits[0].source == "tutorial/cors.md"


def test_vector_search_scores_are_descending(session, docs_tree):
    ingest(session, docs_tree)
    hits = search(session, "serving static files", top_k=3, mode="vector", use_rerank=False)
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)
    assert hits[0].source == "tutorial/static-files.md"


def test_keyword_search_without_matches_returns_empty(session, docs_tree):
    ingest(session, docs_tree)
    assert search(session, "zzzqqq xxyyzz", top_k=3, mode="keyword", use_rerank=False) == []


def test_keyword_search_with_only_punctuation(session, docs_tree):
    ingest(session, docs_tree)
    assert search(session, "?!?", top_k=3, mode="keyword", use_rerank=False) == []


def test_rerank_returns_top_k_sorted(session, docs_tree):
    ingest(session, docs_tree)
    hits = search(session, "run something after the response is sent", top_k=2, mode="hybrid", use_rerank=True)
    assert len(hits) == 2
    assert hits[0].score >= hits[1].score
    assert hits[0].source == "tutorial/background-tasks.md"


def test_search_on_empty_database(session):
    assert search(session, "anything", top_k=3, mode="hybrid", use_rerank=True) == []
