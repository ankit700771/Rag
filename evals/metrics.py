import math


def first_relevant_rank(retrieved: list[str], expected: list[str]) -> int | None:
    for position, source in enumerate(retrieved, start=1):
        if source in expected:
            return position
    return None


def recall_at_k(ranks: list[int | None], k: int) -> float:
    if not ranks:
        return 0.0
    return sum(1 for r in ranks if r is not None and r <= k) / len(ranks)


def mean_reciprocal_rank(ranks: list[int | None]) -> float:
    if not ranks:
        return 0.0
    return sum(1.0 / r for r in ranks if r is not None) / len(ranks)


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, math.ceil(pct / 100 * len(ordered)) - 1)
    return ordered[index]


REFUSAL_MARKER = "don't know based on the provided documentation"


def is_refusal(answer: str) -> bool:
    return REFUSAL_MARKER in answer.lower().replace("’", "'")


def cited_expected_source(
    cited_numbers: list[int], sources: list[str], expected: list[str]
) -> bool:
    return any(
        1 <= number <= len(sources) and sources[number - 1] in expected
        for number in cited_numbers
    )
