import re
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.config import settings
from app.embeddings import embed_documents
from app.ingestion.chunker import chunk_markdown
from app.ingestion.cleaner import clean_markdown
from app.models import Chunk, Document

TITLE_RE = re.compile(r"^#\s+(.*\S)\s*$", re.MULTILINE)


@dataclass
class IngestReport:
    documents: int
    chunks: int
    skipped: int


def is_excluded(relative_path: str, excluded: list[str] | None = None) -> bool:
    patterns = settings.excluded_docs if excluded is None else excluded
    for pattern in patterns:
        if pattern.endswith("/"):
            if relative_path.startswith(pattern):
                return True
        elif relative_path == pattern:
            return True
    return False


def discover_documents(docs_dir: Path | None = None) -> list[Path]:
    root = docs_dir or settings.docs_dir
    files = sorted(root.rglob("*.md"))
    return [
        f for f in files if not is_excluded(f.relative_to(root).as_posix())
    ]


def extract_title(cleaned: str, fallback: str) -> str:
    match = TITLE_RE.search(cleaned)
    return match.group(1) if match else fallback


def prepare_document(path: Path, root: Path, docs_src_dir: Path | None) -> tuple[str, str, list]:
    source = path.relative_to(root).as_posix()
    raw = path.read_text(encoding="utf-8")
    cleaned = clean_markdown(raw, docs_src_dir)
    title = extract_title(cleaned, path.stem.replace("-", " ").title())
    chunks = chunk_markdown(cleaned, title)
    return source, title, chunks


def store_document(session: Session, source: str, title: str, chunks: list) -> int:
    session.execute(delete(Document).where(Document.source == source))
    session.flush()
    document = Document(source=source, title=title)
    session.add(document)
    session.flush()
    vectors = embed_documents([c.content for c in chunks])
    session.add_all(
        Chunk(
            document_id=document.id,
            chunk_index=c.chunk_index,
            content=c.content,
            token_count=c.token_count,
            embedding=vector,
        )
        for c, vector in zip(chunks, vectors)
    )
    return len(chunks)


def ingest_directory(
    session: Session,
    docs_dir: Path | None = None,
    docs_src_dir: Path | None = None,
    limit: int | None = None,
) -> IngestReport:
    root = docs_dir or settings.docs_dir
    src = docs_src_dir if docs_src_dir is not None else settings.docs_src_dir
    files = discover_documents(root)
    if limit is not None:
        files = files[:limit]

    total_chunks = 0
    skipped = 0
    stored_sources: list[str] = []
    for path in files:
        source, title, chunks = prepare_document(path, root, src)
        if not chunks:
            skipped += 1
            continue
        total_chunks += store_document(session, source, title, chunks)
        stored_sources.append(source)
        session.commit()

    if limit is None:
        stale = session.scalars(
            select(Document).where(Document.source.not_in(stored_sources))
        ).all()
        for document in stale:
            session.delete(document)
        session.commit()

    return IngestReport(documents=len(stored_sources), chunks=total_chunks, skipped=skipped)
