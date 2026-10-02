"""Folder of PDFs -> document + chunk rows (Postgres/pgvector) + BM25 index (Elasticsearch).

Idempotent: a file is identified by its SHA-256, so re-running ingestion, renaming or
copying a file never creates a duplicate. Put PDFs at data/pdfs/<TENDER_ID>/<file>.pdf, or
name the file with the tender ID in it (e.g. 2026_AMU_926330_3_NIT.pdf).
"""

import hashlib
import logging
import re
from dataclasses import dataclass
from pathlib import Path

from docintel import db, es, models
from docintel.chunking import chunk_pages
from docintel.config import get_settings
from docintel.extract import extract
from docintel.tenderlens import TenderMeta, lookup

log = logging.getLogger(__name__)
TENDER_ID = re.compile(r"\d{4}_[A-Za-z0-9]+_\d+_\d+")


@dataclass
class IngestResult:
    path: Path
    status: str  # added | skipped | failed
    document_id: int | None = None
    chunks: int = 0
    ocr_pages: int = 0
    error: str = ""


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def tender_id_for(path: Path) -> str:
    for part in (path.parent.name, path.stem):
        m = TENDER_ID.search(part)
        if m:
            return m.group(0)
    return ""


def ingest_file(path: Path, *, meta: TenderMeta | None = None) -> IngestResult:
    s = get_settings()
    digest = sha256(path)
    with db.connection() as conn:
        row = conn.execute("SELECT id FROM document WHERE sha256 = %s", (digest,)).fetchone()
        if row:
            return IngestResult(path, "skipped", document_id=row[0])

    ex = extract(path, ocr_dir=s.ocr_cache_dir)
    chunks = chunk_pages(
        ex.pages,
        target=s.chunk_target_tokens,
        max_tokens=s.chunk_max_tokens,
        overlap=s.chunk_overlap_tokens,
    )
    if not chunks:
        return IngestResult(
            path, "failed", error="no text could be extracted (scanned PDF without OCR?)"
        )
    tid = meta.tender_id if meta else tender_id_for(path)
    meta = meta or lookup(tid)
    # Embed "heading + text": the heading often carries the meaning ("EMD", "Turnover").
    vectors = models.embed_passages(
        [
            f"{c.heading}\n{c.text}" if c.heading and not c.text.startswith(c.heading) else c.text
            for c in chunks
        ]
    )

    with db.connection() as conn, conn.transaction():
        doc_id = conn.execute(
            """INSERT INTO document (tender_id, filename, path, sha256, pages, ocr_used, ocr_pages, bytes,
                                     state, buyer, tender_title, closes_at)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING id""",
            (
                tid,
                path.name,
                str(path),
                digest,
                len(ex.pages),
                ex.ocr_pages > 0,
                ex.ocr_pages,
                path.stat().st_size,
                meta.state,
                meta.buyer,
                meta.title,
                meta.closes_at,
            ),
        ).fetchone()[0]
        ids = []
        with conn.cursor() as cur:
            for c, v in zip(chunks, vectors, strict=True):
                cur.execute(
                    """INSERT INTO chunk (document_id, ord, page_from, page_to, heading, text, tokens, embedding)
                       VALUES (%s,%s,%s,%s,%s,%s,%s,%s) RETURNING id""",
                    (doc_id, c.ord, c.page_from, c.page_to, c.heading, c.text, c.tokens, v),
                )
                ids.append(cur.fetchone()[0])
        # Index inside the transaction: if Elasticsearch fails, Postgres rolls back too, so
        # the two stores never disagree about which documents exist.
        es.index_chunks(
            [
                {
                    "chunk_id": cid,
                    "document_id": doc_id,
                    "tender_id": tid,
                    "filename": path.name,
                    "heading": c.heading,
                    "text": c.text,
                    "state": meta.state or None,
                    "closes_at": meta.closes_at.isoformat() if meta.closes_at else None,
                    "page_from": c.page_from,
                    "page_to": c.page_to,
                }
                for cid, c in zip(ids, chunks, strict=True)
            ]
        )
    return IngestResult(
        path, "added", document_id=doc_id, chunks=len(chunks), ocr_pages=ex.ocr_pages
    )


def ingest_folder(folder: Path | None = None) -> list[IngestResult]:
    folder = folder or get_settings().pdf_dir
    results = []
    for path in sorted(folder.rglob("*.pdf")):
        try:
            r = ingest_file(path)
        except Exception as exc:  # one bad PDF must not stop the batch
            log.exception("failed to ingest %s", path)
            r = IngestResult(path, "failed", error=f"{type(exc).__name__}: {exc}")
        log.info(
            "%s %s chunks=%d ocr_pages=%d %s", r.status, path.name, r.chunks, r.ocr_pages, r.error
        )
        results.append(r)
    return results


def delete_document(document_id: int) -> None:
    with db.connection() as conn:
        conn.execute("DELETE FROM document WHERE id = %s", (document_id,))
    es.delete_document(document_id)
