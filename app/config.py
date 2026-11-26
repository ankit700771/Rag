from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=PROJECT_ROOT / ".env", extra="ignore")

    database_url: str

    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_batch_size: int = 32
    query_instruction: str = "Represent this sentence for searching relevant passages: "

    chunk_max_tokens: int = 400
    chunk_overlap_tokens: int = 50
    chunk_min_tokens: int = 60

    docs_dir: Path = PROJECT_ROOT / "data" / "fastapi_docs"
    docs_src_dir: Path = PROJECT_ROOT / "data" / "docs_src"
    excluded_docs: list[str] = ["release-notes.md", "reference/", "_llm-test.md"]

    retrieval_candidates: int = 30
    rrf_k: int = 60
    top_k: int = 5

    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_enabled: bool = True
    rerank_candidates: int = 15

    gemini_api_key: str | None = None
    gemini_model: str = "gemini-3.5-flash"
    gemini_timeout_seconds: int = 45
    gemini_retry_attempts: int = 2
    max_context_chunks: int = 5


settings = Settings()
