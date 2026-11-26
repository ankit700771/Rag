from typing import Literal

from sqlalchemy.orm import Session

from app.config import settings
from app.retrieval.fusion import reciprocal_rank_fusion
from app.retrieval.keyword import keyword_search
from app.retrieval.reranker import rerank
from app.retrieval.types import Hit
from app.retrieval.vector import vector_search

Mode = Literal["vector", "keyword", "hybrid"]


def search(
    session: Session,
    query: str,
    top_k: int | None = None,
    mode: Mode = "hybrid",
    use_rerank: bool | None = None,
) -> list[Hit]:
    top_k = top_k or settings.top_k
    use_rerank = settings.rerank_enabled if use_rerank is None else use_rerank
    candidates = max(settings.retrieval_candidates, top_k)

    if mode == "vector":
        pool = vector_search(session, query, candidates)
    elif mode == "keyword":
        pool = keyword_search(session, query, candidates)
    else:
        pool = reciprocal_rank_fusion(
            [
                vector_search(session, query, candidates),
                keyword_search(session, query, candidates),
            ],
            k=settings.rrf_k,
        )

    if use_rerank:
        return rerank(query, pool[: max(settings.rerank_candidates, top_k)], top_k)
    return pool[:top_k]
