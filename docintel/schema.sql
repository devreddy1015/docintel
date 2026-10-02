-- Idempotent: safe to run on every start.
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS document (
    id            BIGSERIAL PRIMARY KEY,
    tender_id     TEXT NOT NULL DEFAULT '',     -- GePNIC Tender ID, e.g. 2026_AMU_926330_3
    filename      TEXT NOT NULL,
    path          TEXT NOT NULL,
    sha256        CHAR(64) NOT NULL UNIQUE,     -- the same file is never ingested twice
    pages         INTEGER NOT NULL,
    ocr_used      BOOLEAN NOT NULL DEFAULT FALSE,
    ocr_pages     INTEGER NOT NULL DEFAULT 0,
    bytes         BIGINT NOT NULL,
    -- Copied from TenderLens at ingest time; used as retrieval filters.
    state         TEXT NOT NULL DEFAULT '',
    buyer         TEXT NOT NULL DEFAULT '',
    tender_title  TEXT NOT NULL DEFAULT '',
    closes_at     TIMESTAMPTZ,
    added_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS document_tender_id ON document (tender_id);

CREATE TABLE IF NOT EXISTS chunk (
    id           BIGSERIAL PRIMARY KEY,
    document_id  BIGINT NOT NULL REFERENCES document(id) ON DELETE CASCADE,
    ord          INTEGER NOT NULL,              -- position within the document
    page_from    INTEGER NOT NULL,
    page_to      INTEGER NOT NULL,
    heading      TEXT NOT NULL DEFAULT '',
    text         TEXT NOT NULL,
    tokens       INTEGER NOT NULL,
    embedding    vector(384),
    UNIQUE (document_id, ord)
);
-- HNSW: approximate nearest neighbour; cosine distance because e5 vectors are normalised.
CREATE INDEX IF NOT EXISTS chunk_embedding_hnsw ON chunk USING hnsw (embedding vector_cosine_ops);
