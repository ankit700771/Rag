import re
from pathlib import Path

DEFAULT_DOCS_SRC = Path(__file__).resolve().parents[2] / "data" / "docs_src"

FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")

INCLUDE_RE = re.compile(r"^\s*\{[*!]>?.*[*!]\}\s*$")
INCLUDE_PATH_RE = re.compile(r"docs_src/(\S+?)(?=\s|!\}|\*\})")
INCLUDE_LINES_RE = re.compile(r"\bln\[([^\]]*)\]")
INCLUDE_TITLE_RE = re.compile(r'\btitle\["([^"]*)"\]')
CODE_LANGS = {".py": "python", ".js": "javascript", ".html": "html", ".css": "css"}
ADMONITION_OPEN_RE = re.compile(r"^\s*/{3,}\s*(\w+)\s*(?:\|\s*(.*))?$")
ADMONITION_CLOSE_RE = re.compile(r"^\s*/{3,}\s*$")
HEADING_ANCHOR_RE = re.compile(r"^(#{1,6}\s.*?)\s*\{\s*#[^}]*\}\s*$")
IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]*\)")
HTML_TAG_RE = re.compile(r"</?[a-zA-Z][^>]*>")
TERMINAL_TAG_RE = re.compile(r"</?(?:font|span|u|b|i)(?:\s[^>]*)?>")
BLANK_RUN_RE = re.compile(r"\n{3,}")


def _strip_frontmatter(text: str) -> str:
    if text.startswith("---\n"):
        end = text.find("\n---", 4)
        if end != -1:
            return text[end + 4 :].lstrip("\n")
    return text


def _admonition_label(kind: str, title: str | None) -> str:
    label = kind.capitalize()
    return f"{label} - {title.strip()}:" if title else f"{label}:"


def _select_lines(lines: list[str], spec: str) -> list[str]:
    picked: list[str] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        start, _, end = part.partition(":")
        first = int(start)
        last = int(end) if end else first
        picked.extend(lines[first - 1 : last])
    return picked


def _resolve_include(line: str, docs_src_dir: Path | None) -> tuple[str, str, str] | None:
    if docs_src_dir is None:
        return None
    path_match = INCLUDE_PATH_RE.search(line)
    if not path_match:
        return None
    path = docs_src_dir / path_match.group(1)
    if not path.is_file():
        return None

    lines = path.read_text(encoding="utf-8").rstrip("\n").split("\n")
    lines_match = INCLUDE_LINES_RE.search(line)
    if lines_match:
        try:
            lines = _select_lines(lines, lines_match.group(1))
        except ValueError:
            pass
    title_match = INCLUDE_TITLE_RE.search(line)
    title = title_match.group(1) if title_match else ""
    return CODE_LANGS.get(path.suffix, ""), "\n".join(lines), title


def clean_markdown(text: str, docs_src_dir: Path | None = DEFAULT_DOCS_SRC) -> str:
    text = _strip_frontmatter(text.replace("\r\n", "\n"))

    out: list[str] = []
    in_code = False
    in_console = False

    for line in text.split("\n"):
        fence = FENCE_RE.match(line)
        if fence:
            in_code = not in_code
            in_console = in_code and line.strip().lstrip("`~").lower().startswith("console")
            out.append(line)
            continue

        if in_code:
            if INCLUDE_RE.match(line):
                included = _resolve_include(line, docs_src_dir)
                if included:
                    out.append(included[1])
                continue
            if in_console:
                line = TERMINAL_TAG_RE.sub("", line)
            out.append(line)
            continue

        if INCLUDE_RE.match(line):
            included = _resolve_include(line, docs_src_dir)
            if included:
                lang, code, title = included
                if title:
                    out.append(f"File: {title}")
                out.extend([f"```{lang}", code, "```"])
            continue
        if ADMONITION_CLOSE_RE.match(line):
            continue

        m = ADMONITION_OPEN_RE.match(line)
        if m:
            out.append(_admonition_label(m.group(1), m.group(2)))
            continue

        line = HEADING_ANCHOR_RE.sub(r"\1", line)
        line = IMAGE_RE.sub("", line)
        line = LINK_RE.sub(r"\1", line)
        line = HTML_TAG_RE.sub("", line)
        out.append(line)

    cleaned = "\n".join(out)
    cleaned = BLANK_RUN_RE.sub("\n\n", cleaned)
    return cleaned.strip() + "\n"
