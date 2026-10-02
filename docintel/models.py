"""Embedding and re-ranking models, loaded once per process on first use."""

from functools import lru_cache

import numpy as np

from docintel.config import get_settings


@lru_cache(maxsize=4)
def embedder(name: str | None = None):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(name or get_settings().embedding_model, device="cpu")


@lru_cache(maxsize=1)
def reranker():
    from sentence_transformers import CrossEncoder

    return CrossEncoder(get_settings().reranker_model, device="cpu")


# e5 models are trained with these prefixes; leaving them out costs several points of recall.
def _prefix(name: str, kind: str) -> str:
    return f"{kind}: " if "e5" in name.lower() else ""


def embed_passages(texts: list[str], model_name: str | None = None) -> np.ndarray:
    name = model_name or get_settings().embedding_model
    p = _prefix(name, "passage")
    return embedder(name).encode(
        [p + t for t in texts], batch_size=32, normalize_embeddings=True, show_progress_bar=False
    )


def embed_query(text: str, model_name: str | None = None) -> np.ndarray:
    name = model_name or get_settings().embedding_model
    return embedder(name).encode(
        [_prefix(name, "query") + text], normalize_embeddings=True, show_progress_bar=False
    )[0]


def rerank_scores(query: str, passages: list[str]) -> list[float]:
    if not passages:
        return []
    return [
        float(s)
        for s in reranker().predict([(query, p) for p in passages], show_progress_bar=False)
    ]
