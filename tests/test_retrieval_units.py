from app.retrieval.fusion import reciprocal_rank_fusion
from app.retrieval.keyword import build_or_query
from app.retrieval.types import Hit


def make_hit(chunk_id: int) -> Hit:
    return Hit(chunk_id, f"s{chunk_id}.md", "T", 0, f"c{chunk_id}", 0.0)


def test_rrf_prefers_items_ranked_high_in_both_lists():
    a = [make_hit(1), make_hit(2), make_hit(3)]
    b = [make_hit(3), make_hit(1), make_hit(4)]
    fused = reciprocal_rank_fusion([a, b], k=60)
    assert [h.chunk_id for h in fused][:2] == [1, 3]
    assert {h.chunk_id for h in fused} == {1, 2, 3, 4}


def test_rrf_scores_are_summed():
    fused = reciprocal_rank_fusion([[make_hit(1)], [make_hit(1)]], k=60)
    assert fused[0].score == 2 / 61


def test_rrf_handles_empty_lists():
    assert reciprocal_rank_fusion([[], []]) == []


def test_rrf_is_deterministic_on_ties():
    fused = reciprocal_rank_fusion([[make_hit(5)], [make_hit(2)]])
    assert [h.chunk_id for h in fused] == [2, 5]


def test_or_query_dedupes_and_lowercases():
    assert build_or_query("How do I use CORS? cors!") == "how | do | i | use | cors"


def test_or_query_empty_for_punctuation():
    assert build_or_query("?!... --") == ""
