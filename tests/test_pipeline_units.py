from app.ingestion.pipeline import discover_documents, extract_title, is_excluded


def test_is_excluded_by_prefix_and_exact():
    assert is_excluded("reference/fastapi.md", ["reference/"])
    assert is_excluded("release-notes.md", ["release-notes.md"])
    assert not is_excluded("tutorial/release-notes.md", ["release-notes.md"])
    assert not is_excluded("tutorial/body.md", ["reference/", "release-notes.md"])


def test_extract_title():
    assert extract_title("intro\n\n# Real Title\n\n## Sub\n", "fb") == "Real Title"
    assert extract_title("no heading", "fb") == "fb"


def test_discover_documents_applies_exclusions(docs_tree):
    names = [p.relative_to(docs_tree).as_posix() for p in discover_documents(docs_tree)]
    assert names == [
        "tutorial/background-tasks.md",
        "tutorial/cors.md",
        "tutorial/static-files.md",
    ]
