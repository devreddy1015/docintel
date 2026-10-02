from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All configuration comes from environment variables (or a .env file)."""

    model_config = SettingsConfigDict(env_file=".env", env_prefix="DOCINTEL_", extra="ignore")

    database_url: str = "postgresql://tenderlens:tenderlens@localhost:5432/docintel"
    es_url: str = "http://localhost:9200"
    es_index: str = "docintel_chunks"
    pdf_dir: Path = Path("data/pdfs")
    ocr_cache_dir: Path = Path("data/ocr")
    # TenderLens API, for tender metadata (state, closing date, buyer) used as filters.
    tenderlens_url: str = "http://localhost:8080"

    # multilingual-e5-small: 384 dimensions, English + Hindi (some NITs are bilingual).
    embedding_model: str = "intfloat/multilingual-e5-small"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    embedding_dim: int = 384

    chunk_target_tokens: int = 500
    chunk_max_tokens: int = 800
    chunk_overlap_tokens: int = 60

    bm25_k: int = 50
    dense_k: int = 50
    rrf_k: int = 60
    rerank_candidates: int = 30
    final_k: int = 5

    llm_model: str = "claude-opus-5-5"
    llm_effort: str = "medium"
    llm_max_tokens: int = 4000


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
