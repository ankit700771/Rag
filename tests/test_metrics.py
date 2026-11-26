from evals.metrics import (
    cited_expected_source,
    first_relevant_rank,
    is_refusal,
    mean_reciprocal_rank,
    percentile,
    recall_at_k,
)


def test_first_relevant_rank():
    assert first_relevant_rank(["a", "b", "c"], ["c", "x"]) == 3
    assert first_relevant_rank(["a"], ["z"]) is None


def test_recall_at_k():
    ranks = [1, 3, None, 6]
    assert recall_at_k(ranks, 1) == 0.25
    assert recall_at_k(ranks, 5) == 0.5


def test_mrr():
    assert mean_reciprocal_rank([1, 2, None]) == (1 + 0.5) / 3


def test_percentile():
    assert percentile([10, 20, 30, 40, 50], 50) == 30
    assert percentile([10, 20, 30, 40, 50], 95) == 50
    assert percentile([], 50) == 0.0


def test_is_refusal():
    assert is_refusal("I don't know based on the provided documentation.")
    assert is_refusal("I don’t know based on the provided documentation.")
    assert not is_refusal("Use CORSMiddleware [1].")


def test_cited_expected_source():
    sources = ["a.md", "b.md", "c.md"]
    assert cited_expected_source([2], sources, ["b.md"])
    assert not cited_expected_source([1, 3], sources, ["b.md"])
    assert not cited_expected_source([9], sources, ["b.md"])
    assert not cited_expected_source([], sources, ["a.md"])
