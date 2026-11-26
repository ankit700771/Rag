from app.ingestion.chunker import chunk_markdown, count_tokens

DOC = (
    "# Guide\n\nIntro paragraph about the guide.\n\n"
    "## First\n\n" + " ".join(f"First sentence number {i}." for i in range(80)) + "\n\n"
    "## Second\n\nShort text.\n\n"
    "```python\nprint('hello')\n```\n"
)


def build(max_tokens=120, overlap=20, min_tokens=10):
    return chunk_markdown(DOC, "Guide", max_tokens=max_tokens, overlap_tokens=overlap, min_tokens=min_tokens)


def test_chunks_respect_max_tokens():
    chunks = build()
    assert len(chunks) > 1
    assert all(c.token_count <= 120 for c in chunks)


def test_every_chunk_starts_with_heading_path():
    chunks = build()
    assert all(c.content.startswith("Guide") for c in chunks)
    assert any(c.content.startswith("Guide > First") for c in chunks)


def test_chunk_indexes_are_sequential():
    chunks = build()
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_code_fence_is_never_split():
    code = "\n".join(f"line_{i} = {i}" for i in range(10))
    text = f"# T\n\nIntro.\n\n```python\n{code}\n```\n"
    chunks = chunk_markdown(text, "T", max_tokens=400, overlap_tokens=0, min_tokens=0)
    assert any(code in c.content for c in chunks)
    assert all(c.content.count("```") % 2 == 0 for c in chunks)


def test_long_code_block_is_split_into_balanced_fences():
    code = "\n".join(f"value_{i} = {i} + {i}" for i in range(300))
    text = f"# T\n\n```python\n{code}\n```\n"
    chunks = chunk_markdown(text, "T", max_tokens=150, overlap_tokens=0, min_tokens=0)
    assert len(chunks) > 1
    assert all(c.token_count <= 150 for c in chunks)
    assert all(c.content.count("```") % 2 == 0 for c in chunks)


def test_hash_inside_code_fence_is_not_a_heading():
    text = "# T\n\n```python\n# not a heading\nx = 1\n```\n"
    chunks = chunk_markdown(text, "T", max_tokens=200, overlap_tokens=0, min_tokens=0)
    assert len(chunks) == 1
    assert "# not a heading" in chunks[0].content


def test_tiny_sections_are_merged():
    text = "# T\n\nOne.\n\n## A\n\nTwo.\n\n## B\n\nThree.\n"
    chunks = chunk_markdown(text, "T", max_tokens=200, overlap_tokens=0, min_tokens=60)
    assert len(chunks) == 1
    assert "Two." in chunks[0].content and "Three." in chunks[0].content


def test_overlap_repeats_trailing_content():
    paragraphs = [f"Paragraph {i} " + "word " * 25 for i in range(8)]
    text = "# T\n\n" + "\n\n".join(p.strip() for p in paragraphs) + "\n"
    chunks = chunk_markdown(text, "T", max_tokens=120, overlap_tokens=45, min_tokens=0)
    assert len(chunks) > 2
    shared = [p for p in paragraphs if sum(p.strip() in c.content for c in chunks) > 1]
    assert shared


def test_token_count_matches_content():
    assert all(c.token_count == count_tokens(c.content) for c in build())


def test_fallback_title_used_without_headings():
    chunks = chunk_markdown("Just text.\n", "Fallback")
    assert chunks[0].content.startswith("Fallback")


def test_empty_document_yields_no_chunks():
    assert chunk_markdown("", "T") == []


def test_single_oversized_code_line_is_cut_by_tokens():
    blob = "+".join(f"Ab{i}Zq" for i in range(400))
    text = f'# T\n\n```python\nvalue = "{blob}"\n```\n'
    chunks = chunk_markdown(text, "T", max_tokens=150, overlap_tokens=0, min_tokens=0)
    assert len(chunks) > 1
    assert all(c.token_count <= 150 for c in chunks)
    assert all(c.content.count("```") % 2 == 0 for c in chunks)
