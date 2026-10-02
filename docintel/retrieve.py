"""Hybrid retrieval: BM25 + dense, fused with Reciprocal Rank Fusion, then re-ranked.

Modes (all kept, because the evaluation compares them):
  bm25           Elasticsearch BM25 over chunk text
  dense          pgvector cosine search over e5 embeddings (HNSW index)
  hybrid         RRF(bm25 top 50, dense top 50), k = 60
  hybrid_rerank  hybrid top 30 -> cross-encoder -> top 5   (the default)

Why both: BM25 nails exact tokens - tender IDs, clause numbers, "EMD" - that embeddings
blur; embeddings match paraphrases ("bid security" ~ "earnest money") that BM25 misses.
"""

from dataclasses import dataclass, field
from datetime import datetime

from docintel import db, es, models
from docintel.config import get_settings

MODES = ("bm25", "dense", "hybrid", "hybrid_rerank")


@dataclass
class Filters:
    tender_id: str | None = None
    state: str | None = None
    closes_after: datetime | None = None
    document_ids: list[int] | None = None


@dataclass
class Hit:
    chunk_id: int
    document_id: int
    filename: str
    tender_id: str
    page_from: int
    page_to: int
    heading: str
    text: str
    score: float = 0.0
    ranks: dict[str, int] = field(default_factory=dict)

    @property
    def pages(self) -> str:
        return (
            f"p. {self.page_from}"
            if self.page_from == self.page_to
            else f"pp. {self.page_from}-{self.page_to}"
        )


def _es_filters(f: Filters) -> list[dict]:
    out = []
    if f.tender_id:
        out.append({"term": {"tender_id": f.tender_id}})
    if f.state:
        out.append({"term": {"state": f.state}})
    if f.closes_after:
        out.append({"range": {"closes_at": {"gte": f.closes_after.isoformat()}}})
    if f.document_ids:
        out.append({"terms": {"document_id": f.document_ids}})
    return out


def bm25(query: str, f: Filters, k: int, index: str | None = None) -> list[int]:
    resp = es.client().search(
        index=index or get_settings().es_index,
        size=k,
        source=False,
        query={
            "bool": {
                "must": [{"multi_match": {"query": query, "fields": ["text", "heading^2"]}}],
                "filter": _es_filters(f),
            }
        },
    )
    return [int(h["_id"]) for h in resp["hits"]["hits"]]


def dense(
    query: str, f: Filters, k: int, *, model_name: str | None = None, column: str = "embedding"
) -> list[int]:
    vec = models.embed_query(query, model_name)
    if column not in ("embedding", "embedding_ft"):  # interpolated into SQL: allow-list
        raise ValueError(f"unknown embedding column {column!r}")
    where, params = [f"c.{column} IS NOT NULL"], []
    if f.tender_id:
        where.append("d.tender_id = %s")
        params.append(f.tender_id)
    if f.state:
        where.append("d.state = %s")
        params.append(f.state)
    if f.closes_after:
        where.append("d.closes_at >= %s")
        params.append(f.closes_after)
    if f.document_ids:
        where.append("d.id = ANY(%s)")
        params.append(f.document_ids)
    sql = f"""
        SELECT c.id FROM chunk c JOIN document d ON d.id = c.document_id
        WHERE {" AND ".join(where)}
        ORDER BY c.{column} <=> %s LIMIT %s
    """
    with db.connection() as conn, conn.transaction():
        # Filters are applied after the HNSW walk; iterative scan keeps walking until it
        # has k rows that pass them, instead of returning fewer than k.
        conn.execute("SET LOCAL hnsw.ef_search = 100")
        conn.execute("SET LOCAL hnsw.iterative_scan = relaxed_order")
        rows = conn.execute(sql, [*params, vec, k]).fetchall()
    return [r[0] for r in rows]


def rrf(*ranked_lists: list[int], k: int = 60) -> list[tuple[int, float]]:
    """Reciprocal Rank Fusion: score = sum over lists of 1 / (k + rank), rank from 1.

    Uses only ranks, so BM25 scores and cosine similarities never need to be put on a
    common scale.
    """
    scores: dict[int, float] = {}
    for lst in ranked_lists:
        for rank, cid in enumerate(lst, start=1):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))


def load_hits(ids: list[int]) -> dict[int, Hit]:
    if not ids:
        return {}
    with db.connection() as conn:
        rows = conn.execute(
            """SELECT c.id, c.document_id, d.filename, d.tender_id, c.page_from, c.page_to, c.heading, c.text
               FROM chunk c JOIN document d ON d.id = c.document_id WHERE c.id = ANY(%s)""",
            (ids,),
        ).fetchall()
    return {r[0]: Hit(*r) for r in rows}


def search(
    query: str,
    *,
    mode: str = "hybrid_rerank",
    filters: Filters | None = None,
    k: int | None = None,
    model_name: str | None = None,
    column: str = "embedding",
    index: str | None = None,
) -> list[Hit]:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    s = get_settings()
    f = filters or Filters()
    k = k or s.final_k
    ranks: dict[int, dict[str, int]] = {}

    def note(name: str, lst: list[int]) -> None:
        for r, cid in enumerate(lst, start=1):
            ranks.setdefault(cid, {})[name] = r

    if mode == "bm25":
        ids = bm25(query, f, max(k, s.bm25_k), index)
        note("bm25", ids)
        scored = [(cid, 1.0 / r) for r, cid in enumerate(ids, start=1)]
    elif mode == "dense":
        ids = dense(query, f, max(k, s.dense_k), model_name=model_name, column=column)
        note("dense", ids)
        scored = [(cid, 1.0 / r) for r, cid in enumerate(ids, start=1)]
    else:
        b = bm25(query, f, s.bm25_k, index)
        d = dense(query, f, s.dense_k, model_name=model_name, column=column)
        note("bm25", b)
        note("dense", d)
        scored = rrf(b, d, k=s.rrf_k)

    if mode == "hybrid_rerank":
        cands = [cid for cid, _ in scored[: s.rerank_candidates]]
        hits = load_hits(cands)
        cands = [c for c in cands if c in hits]
        rs = models.rerank_scores(query, [f"{hits[c].heading}\n{hits[c].text}" for c in cands])
        scored = sorted(zip(cands, rs, strict=True), key=lambda kv: -kv[1])
        out = []
        for cid, score in scored[:k]:
            h = hits[cid]
            h.score, h.ranks = score, ranks.get(cid, {})
            out.append(h)
        return out

    top = scored[:k]
    hits = load_hits([cid for cid, _ in top])
    out = []
    for cid, score in top:
        if cid in hits:
            h = hits[cid]
            h.score, h.ranks = score, ranks.get(cid, {})
            out.append(h)
    return out
