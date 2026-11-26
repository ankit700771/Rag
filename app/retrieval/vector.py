from sqlalchemy import select
from sqlalchemy.orm import Session

from app.embeddings import embed_query
from app.models import Chunk, Document
from app.retrieval.types import Hit


def vector_search(session: Session, query: str, limit: int) -> list[Hit]:
    query_vector = embed_query(query)
    distance = Chunk.embedding.cosine_distance(query_vector)
    statement = (
        select(Chunk, Document, distance.label("distance"))
        .join(Document, Document.id == Chunk.document_id)
        .order_by(distance)
        .limit(limit)
    )
    return [
        Hit(
            chunk_id=chunk.id,
            source=document.source,
            title=document.title,
            chunk_index=chunk.chunk_index,
            content=chunk.content,
            score=1.0 - float(dist),
        )
        for chunk, document, dist in session.execute(statement)
    ]
