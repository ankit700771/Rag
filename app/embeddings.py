from functools import lru_cache

from sentence_transformers import SentenceTransformer

from app.config import settings


@lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    return SentenceTransformer(settings.embedding_model)


def embed_documents(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    vectors = get_model().encode(
        texts,
        batch_size=settings.embedding_batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return vectors.tolist()


def embed_query(query: str) -> list[float]:
    vector = get_model().encode(
        settings.query_instruction + query,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return vector.tolist()
