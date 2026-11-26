from app.ingestion.cleaner import clean_markdown


def test_strips_frontmatter_and_anchor():
    cleaned = clean_markdown("---\ntitle: x\n---\n# Title { #title }\n\nBody\n", None)
    assert cleaned == "# Title\n\nBody\n"


def test_admonitions_become_labels():
    cleaned = clean_markdown("/// tip\n\nUse this.\n\n///\n", None)
    assert "Tip:" in cleaned
    assert "///" not in cleaned


def test_admonition_with_title():
    cleaned = clean_markdown("/// note | Technical Details\n\ntext\n\n///\n", None)
    assert "Note - Technical Details:" in cleaned


def test_links_images_and_html_are_reduced():
    cleaned = clean_markdown("See [the docs](https://x.y) ![img](a.png) <abbr>ok</abbr>\n", None)
    assert cleaned.strip() == "See the docs  ok"


def test_include_is_inlined_with_line_selection(tmp_path):
    src = tmp_path / "ex"
    src.mkdir()
    (src / "a.py").write_text("one = 1\ntwo = 2\nthree = 3\n", encoding="utf-8")
    cleaned = clean_markdown('{* ../../docs_src/ex/a.py ln[2:3] title["a.py"] *}\n', tmp_path)
    assert "File: a.py" in cleaned
    assert "```python" in cleaned
    assert "two = 2" in cleaned and "three = 3" in cleaned
    assert "one = 1" not in cleaned


def test_include_dropped_when_source_missing(tmp_path):
    cleaned = clean_markdown("before\n\n{* ../../docs_src/none.py *}\n\nafter\n", tmp_path)
    assert "docs_src" not in cleaned
    assert "before" in cleaned and "after" in cleaned


def test_code_fence_content_is_untouched():
    text = "```python\nx = [1](2)\n/// tip\n```\n"
    cleaned = clean_markdown(text, None)
    assert "x = [1](2)" in cleaned
    assert "/// tip" in cleaned


def test_console_terminal_tags_removed():
    text = '```console\n<span style="color: green;">INFO</span> started\n```\n'
    cleaned = clean_markdown(text, None)
    assert "<span" not in cleaned
    assert "INFO started" in cleaned


def test_blank_runs_collapsed():
    assert clean_markdown("a\n\n\n\n\nb\n", None) == "a\n\nb\n"
