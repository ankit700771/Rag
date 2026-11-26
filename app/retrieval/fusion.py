from dataclasses import replace

from app.retrieval.types import Hit


def reciprocal_rank_fusion(rankings: list[list[Hit]], k: int = 60) -> list[Hit]:
    scores: dict[int, float] = {}
    first_seen: dict[int, Hit] = {}
    for ranking in rankings:
        for position, hit in enumerate(ranking, start=1):
            scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + 1.0 / (k + position)
            first_seen.setdefault(hit.chunk_id, hit)
    ordered = sorted(scores, key=lambda chunk_id: (-scores[chunk_id], chunk_id))
    return [replace(first_seen[chunk_id], score=scores[chunk_id]) for chunk_id in ordered]
