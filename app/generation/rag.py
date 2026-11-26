from dataclasses import dataclass

from sqlalchemy.orm import Session

from app.config import settings
from app.generation.gemini import Generator
from app.generation.prompt import SYSTEM_INSTRUCTION, build_prompt, extract_citations
from app.retrieval.search import Mode, search
from app.retrieval.types import Hit

NO_CONTEXT_ANSWER = "I don't know based on the provided documentation."


@dataclass
class Answer:
    answer: str
    sources: list[Hit]
    cited: list[int]


def answer_question(
    session: Session,
    generator: Generator,
    question: str,
    top_k: int | None = None,
    mode: Mode = "hybrid",
    use_rerank: bool | None = None,
) -> Answer:
    limit = min(top_k or settings.max_context_chunks, settings.max_context_chunks)
    hits = search(session, question, top_k=limit, mode=mode, use_rerank=use_rerank)
    if not hits:
        return Answer(answer=NO_CONTEXT_ANSWER, sources=[], cited=[])
    text = generator.generate(SYSTEM_INSTRUCTION, build_prompt(question, hits))
    return Answer(answer=text, sources=hits, cited=extract_citations(text, len(hits)))
