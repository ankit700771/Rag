from dataclasses import replace
from functools import lru_cache

from sentence_transformers import CrossEncoder

from app.config import settings
from app.retrieval.types import Hit


@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoder:
    return CrossEncoder(settings.reranker_model)


def rerank(query: str, hits: list[Hit], top_k: int) -> list[Hit]:
    if not hits:
        return []
    scores = get_reranker().predict(
        [(query, hit.content) for hit in hits], show_progress_bar=False
    )
    scored = sorted(zip(hits, scores), key=lambda pair: -float(pair[1]))
    return [replace(hit, score=float(score)) for hit, score in scored[:top_k]]
