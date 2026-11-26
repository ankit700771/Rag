import re

from app.retrieval.types import Hit

SYSTEM_INSTRUCTION = (
    "You answer questions about the FastAPI framework using only the numbered "
    "context passages provided. Cite the passages you rely on with bracketed "
    "numbers such as [1] or [2][3] right after the claim they support. If the "
    "context does not contain the answer, reply exactly: I don't know based on "
    "the provided documentation. Do not use outside knowledge."
)

CITATION_RE = re.compile(r"\[(\d+)\]")


def build_prompt(question: str, hits: list[Hit]) -> str:
    passages = "\n\n".join(
        f"[{number}] (source: {hit.source})\n{hit.content}"
        for number, hit in enumerate(hits, start=1)
    )
    return f"Context passages:\n\n{passages}\n\nQuestion: {question}\n\nAnswer:"


def extract_citations(answer: str, total: int) -> list[int]:
    numbers = []
    for match in CITATION_RE.finditer(answer):
        number = int(match.group(1))
        if 1 <= number <= total and number not in numbers:
            numbers.append(number)
    return numbers
