import re

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models import Chunk, Document
from app.retrieval.types import Hit

WORD_RE = re.compile(r"[A-Za-z0-9_]+")


def build_or_query(query: str) -> str:
    words = list(dict.fromkeys(w.lower() for w in WORD_RE.findall(query)))
    return " | ".join(words)


def keyword_search(session: Session, query: str, limit: int) -> list[Hit]:
    or_query = build_or_query(query)
    if not or_query:
        return []
    ts_query = func.to_tsquery("english", or_query)
    rank = func.ts_rank_cd(Chunk.tsv, ts_query)
    statement = (
        select(Chunk, Document, rank.label("rank"))
        .join(Document, Document.id == Chunk.document_id)
        .where(Chunk.tsv.op("@@")(ts_query))
        .order_by(rank.desc(), Chunk.id)
        .limit(limit)
    )
    return [
        Hit(
            chunk_id=chunk.id,
            source=document.source,
            title=document.title,
            chunk_index=chunk.chunk_index,
            content=chunk.content,
            score=float(score),
        )
        for chunk, document, score in session.execute(statement)
    ]
