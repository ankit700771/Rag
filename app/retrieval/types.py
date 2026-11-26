from dataclasses import dataclass


@dataclass
class Hit:
    chunk_id: int
    source: str
    title: str
    chunk_index: int
    content: str
    score: float
