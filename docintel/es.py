"""BM25 side of hybrid retrieval: one Elasticsearch document per chunk."""

from functools import lru_cache

from elasticsearch import Elasticsearch, helpers

from docintel.config import get_settings

MAPPING = {
    "settings": {"number_of_shards": 1, "number_of_replicas": 0},
    "mappings": {
        "dynamic": "strict",
        "properties": {
            "chunk_id": {"type": "long"},
            "document_id": {"type": "long"},
            "tender_id": {"type": "keyword"},
            "filename": {"type": "keyword"},
            "heading": {"type": "text"},
            # Standard analyser, not English: stemming would merge "deposit"/"deposits"
            # (fine) but also mangle clause numbers and abbreviations like "EMD", "LD".
            "text": {"type": "text"},
            "state": {"type": "keyword"},
            "closes_at": {"type": "date"},
            "page_from": {"type": "integer"},
            "page_to": {"type": "integer"},
        },
    },
}


@lru_cache(maxsize=1)
def client() -> Elasticsearch:
    return Elasticsearch(get_settings().es_url, request_timeout=15)


def ensure_index(recreate: bool = False, index: str | None = None) -> None:
    index = index or get_settings().es_index
    if recreate:
        client().indices.delete(index=index, ignore_unavailable=True)
    if not client().indices.exists(index=index):
        client().indices.create(index=index, body=MAPPING)


def index_chunks(rows: list[dict], index: str | None = None) -> None:
    index = index or get_settings().es_index
    helpers.bulk(
        client(),
        ({"_index": index, "_id": str(r["chunk_id"]), "_source": r} for r in rows),
        refresh="wait_for",
    )


def delete_document(document_id: int, index: str | None = None) -> None:
    client().delete_by_query(
        index=index or get_settings().es_index,
        query={"term": {"document_id": document_id}},
        refresh=True,
    )
