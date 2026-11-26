import re
from dataclasses import dataclass
from functools import lru_cache
from transformers import AutoTokenizer
from app.config import settings
from app.ingestion.cleaner import FENCE_RE

HEADING_RE = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n")
SPLIT_LEVELS = (1, 2, 3)


@dataclass
class ChunkData:
    chunk_index: int
    heading_path: list[str]
    content: str
    token_count: int


@dataclass
class _Section:
    path: list[str]
    heading_line: str
    blocks: list[str]


@lru_cache(maxsize=1)
def get_tokenizer():
    tok = AutoTokenizer.from_pretrained(settings.embedding_model)
    tok.model_max_length = 10**9
    return tok


def count_tokens(text: str) -> int:
    return len(get_tokenizer().encode(text, add_special_tokens=False))


def _parse_sections(text: str, fallback_title: str) -> list[_Section]:
    sections: list[_Section] = []
    path: list[str] = []
    current = _Section(path=[fallback_title], heading_line="", blocks=[])
    block: list[str] = []
    in_code = False

    def flush_block() -> None:
        if block:
            current.blocks.append("\n".join(block).strip("\n"))
            block.clear()

    for line in text.split("\n"):
        if FENCE_RE.match(line):
            if not in_code:
                flush_block()
                in_code = True
                block.append(line)
            else:
                block.append(line)
                in_code = False
                flush_block()
            continue

        if in_code:
            block.append(line)
            continue

        heading = HEADING_RE.match(line)
        if heading and len(heading.group(1)) in SPLIT_LEVELS:
            flush_block()
            sections.append(current)
            level = len(heading.group(1))
            path = path[: level - 1] + [heading.group(2)]
            current = _Section(path=list(path), heading_line=line, blocks=[])
            continue

        if line.strip() == "":
            flush_block()
        else:
            block.append(line)

    flush_block()
    sections.append(current)
    return [s for s in sections if s.blocks or s.heading_line]


def _split_long_text(text: str, budget: int) -> list[str]:
    tok = get_tokenizer()
    parts: list[str] = []
    cur: list[str] = []
    cur_tokens = 0
    for unit in filter(None, (u.strip() for u in SENTENCE_RE.split(text))):
        ut = count_tokens(unit)
        if ut > budget:
            if cur:
                parts.append(" ".join(cur))
                cur, cur_tokens = [], 0
            ids = tok.encode(unit, add_special_tokens=False)
            for i in range(0, len(ids), budget):
                parts.append(tok.decode(ids[i : i + budget]))
            continue
        if cur and cur_tokens + ut > budget:
            parts.append(" ".join(cur))
            cur, cur_tokens = [], 0
        cur.append(unit)
        cur_tokens += ut
    if cur:
        parts.append(" ".join(cur))
    return parts


def _split_long_code(block: str, budget: int) -> list[str]:
    lines = block.split("\n")
    opener, closer = lines[0], "```"
    body = lines[1:-1] if lines[-1].strip().startswith(("```", "~~~")) else lines[1:]
    overhead = count_tokens(opener + "\n" + closer)

    parts: list[str] = []
    cur: list[str] = []
    cur_tokens = overhead
    tok = get_tokenizer()
    for line in body:
        lt = count_tokens(line) + 1
        if overhead + lt > budget:
            if cur:
                parts.append("\n".join([opener, *cur, closer]))
                cur, cur_tokens = [], overhead
            ids = tok.encode(line, add_special_tokens=False)
            step = max(1, budget - overhead - 8)
            for i in range(0, len(ids), step):
                piece = tok.decode(ids[i : i + step])
                parts.append("\n".join([opener, piece, closer]))
            continue
        if cur and cur_tokens + lt > budget:
            parts.append("\n".join([opener, *cur, closer]))
            cur, cur_tokens = [], overhead
        cur.append(line)
        cur_tokens += lt
    if cur:
        parts.append("\n".join([opener, *cur, closer]))
    return parts


def _fit_blocks(blocks: list[str], budget: int) -> list[str]:
    fitted: list[str] = []
    for block in blocks:
        if count_tokens(block) <= budget:
            fitted.append(block)
        elif FENCE_RE.match(block.split("\n")[0]):
            fitted.extend(_split_long_code(block, budget))
        else:
            fitted.extend(_split_long_text(block, budget))
    return fitted


def _pack(blocks: list[str], budget: int, overlap: int) -> list[list[str]]:
    sized = [(b, count_tokens(b)) for b in blocks]
    groups: list[list[tuple[str, int]]] = []
    cur: list[tuple[str, int]] = []
    cur_tokens = 0

    for block, bt in sized:
        if cur and cur_tokens + bt > budget:
            groups.append(cur)
            carry: list[tuple[str, int]] = []
            carry_tokens = 0
            for prev, pt in reversed(cur):
                if carry_tokens + pt > overlap:
                    break
                carry.insert(0, (prev, pt))
                carry_tokens += pt
            if carry_tokens + bt > budget:
                carry, carry_tokens = [], 0
            cur, cur_tokens = carry, carry_tokens
        cur.append((block, bt))
        cur_tokens += bt

    if cur:
        groups.append(cur)
    return [[b for b, _ in g] for g in groups]


def _prefix(path: list[str]) -> str:
    return " > ".join(path) + "\n\n"


def chunk_markdown(
    text: str,
    title: str,
    max_tokens: int | None = None,
    overlap_tokens: int | None = None,
    min_tokens: int | None = None,
) -> list[ChunkData]:
    max_tokens = max_tokens or settings.chunk_max_tokens
    overlap_tokens = settings.chunk_overlap_tokens if overlap_tokens is None else overlap_tokens
    min_tokens = settings.chunk_min_tokens if min_tokens is None else min_tokens

    sections = _parse_sections(text, fallback_title=title)

    groups: list[tuple[list[str], list[str]]] = []
    for sec in sections:
        if groups:
            g_path, g_blocks = groups[-1]
            budget = max_tokens - count_tokens(_prefix(g_path))
            g_tokens = sum(count_tokens(b) for b in g_blocks)
            sec_blocks = ([sec.heading_line] if sec.heading_line else []) + sec.blocks
            sec_tokens = sum(count_tokens(b) for b in sec_blocks)
            if g_tokens < min_tokens and g_tokens + sec_tokens <= budget:
                g_blocks.extend(sec_blocks)
                continue
        groups.append((sec.path, list(sec.blocks)))

    chunks: list[ChunkData] = []
    for path, blocks in groups:
        if not blocks:
            continue
        prefix = _prefix(path)
        budget = max_tokens - count_tokens(prefix)
        for packed in _pack(_fit_blocks(blocks, budget), budget, overlap_tokens):
            content = prefix + "\n\n".join(packed)
            chunks.append(
                ChunkData(
                    chunk_index=len(chunks),
                    heading_path=path,
                    content=content,
                    token_count=count_tokens(content),
                )
            )
    return chunks
