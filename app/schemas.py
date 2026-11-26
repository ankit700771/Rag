from typing import Literal

from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=20)
    mode: Literal["vector", "keyword", "hybrid"] = "hybrid"
    rerank: bool = True


class HitOut(BaseModel):
    chunk_id: int
    source: str
    title: str
    chunk_index: int
    content: str
    score: float


class SearchResponse(BaseModel):
    query: str
    mode: str
    results: list[HitOut]
    latency_ms: float


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=10)
    mode: Literal["vector", "keyword", "hybrid"] = "hybrid"
    rerank: bool = True


class SourceOut(BaseModel):
    number: int
    source: str
    title: str
    chunk_index: int
    cited: bool
    content: str


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceOut]
    latency_ms: float


class IngestRequest(BaseModel):
    limit: int | None = Field(default=None, ge=1)


class IngestResponse(BaseModel):
    documents: int
    chunks: int
    skipped: int
    latency_ms: float


class StatsResponse(BaseModel):
    documents: int
    chunks: int
