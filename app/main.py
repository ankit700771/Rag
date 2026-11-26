import time
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException
from sqlalchemy import func, select, text
from sqlalchemy.orm import Session

from app import models
from app.db import Base, engine, get_db
from app.embeddings import get_model
from app.generation.gemini import GenerationUnavailable, get_generator
from app.generation.rag import answer_question
from app.ingestion.pipeline import ingest_directory
from app.retrieval.reranker import get_reranker
from app.retrieval.search import search
from app.schemas import (
    AskRequest,
    AskResponse,
    HitOut,
    IngestRequest,
    IngestResponse,
    SearchRequest,
    SearchResponse,
    SourceOut,
    StatsResponse,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    Base.metadata.create_all(engine)
    get_model()
    get_reranker()
    yield


app = FastAPI(title="RAG Project", lifespan=lifespan)


def elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 1)


@app.get("/health")
def health(db: Session = Depends(get_db)):
    try:
        db.execute(text("SELECT 1"))
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"database unavailable: {exc}")
    return {"status": "ok", "db": "connected"}


@app.get("/stats", response_model=StatsResponse)
def stats(db: Session = Depends(get_db)):
    documents = db.scalar(select(func.count()).select_from(models.Document))
    chunks = db.scalar(select(func.count()).select_from(models.Chunk))
    return StatsResponse(documents=documents or 0, chunks=chunks or 0)


@app.post("/ingest", response_model=IngestResponse)
def ingest(request: IngestRequest | None = None, db: Session = Depends(get_db)):
    started = time.perf_counter()
    limit = request.limit if request else None
    report = ingest_directory(db, limit=limit)
    return IngestResponse(
        documents=report.documents,
        chunks=report.chunks,
        skipped=report.skipped,
        latency_ms=elapsed_ms(started),
    )


@app.post("/search", response_model=SearchResponse)
def search_endpoint(request: SearchRequest, db: Session = Depends(get_db)):
    started = time.perf_counter()
    hits = search(
        db,
        request.query,
        top_k=request.top_k,
        mode=request.mode,
        use_rerank=request.rerank,
    )
    return SearchResponse(
        query=request.query,
        mode=request.mode,
        results=[HitOut(**vars(hit)) for hit in hits],
        latency_ms=elapsed_ms(started),
    )


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest, db: Session = Depends(get_db)):
    started = time.perf_counter()
    try:
        generator = get_generator()
    except GenerationUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    try:
        result = answer_question(
            db,
            generator,
            request.question,
            top_k=request.top_k,
            mode=request.mode,
            use_rerank=request.rerank,
        )
    except GenerationUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    sources = [
        SourceOut(
            number=number,
            source=hit.source,
            title=hit.title,
            chunk_index=hit.chunk_index,
            cited=number in result.cited,
            content=hit.content,
        )
        for number, hit in enumerate(result.sources, start=1)
    ]
    return AskResponse(
        question=request.question,
        answer=result.answer,
        sources=sources,
        latency_ms=elapsed_ms(started),
    )
