"""HTTP API.

POST /ask           Server-Sent Events: retrieval -> delta... -> citation... -> final
POST /ask?stream=0  the same answer as one JSON object
GET  /search        retrieval only (any mode), for debugging and the eval UI
GET  /documents     what has been ingested
GET  /health

The streamed text is a draft until the 'final' event: if the grounding check rejects it,
'final' says so and carries the refusal, and clients must replace what they showed.
"""

import json
import logging
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from docintel import db, es
from docintel.answer import Answer, ask_stream
from docintel.retrieve import MODES, Filters, search

log = logging.getLogger(__name__)
app = FastAPI(
    title="DocIntel",
    version="0.1.0",
    description="Grounded answers about Indian tender documents, with citations.",
)
WEB = Path(__file__).resolve().parent.parent / "web"


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    tender_id: str | None = Field(default=None, max_length=64)
    state: str | None = Field(default=None, max_length=64)
    closes_after: datetime | None = None


def _filters(r: AskRequest) -> Filters:
    return Filters(
        tender_id=r.tender_id or None, state=r.state or None, closes_after=r.closes_after
    )


def answer_json(a: Answer) -> dict:
    return {
        "question": a.question,
        "status": a.status,
        "answer": a.text,
        "citations": [
            asdict(c) | {"document": c.filename, "page": c.page_from} for c in a.citations
        ],
        "grounding": asdict(a.grounding) if a.grounding else None,
        "model": a.model,
        "error": a.error or None,
    }


def _backend():
    """None means "use Claude"; a clear error if no API key is configured."""
    try:
        from docintel.answer import ClaudeBackend

        return ClaudeBackend()
    except Exception as exc:  # the SDK raises when no credentials can be found
        raise HTTPException(503, f"LLM not configured: {exc}. Set ANTHROPIC_API_KEY.") from exc


@app.post("/ask")
async def ask(req: AskRequest, stream: bool = True):
    backend = _backend()
    events = ask_stream(req.question, filters=_filters(req), backend=backend)
    if not stream:
        async for ev in events:
            if ev["type"] == "final":
                return answer_json(ev["answer"])
        raise HTTPException(500, "no answer produced")

    async def sse():
        async for ev in events:
            if ev["type"] == "final":
                ev = {"type": "final", **answer_json(ev["answer"])}
            yield f"event: {ev['type']}\ndata: {json.dumps(ev, ensure_ascii=False, default=str)}\n\n"

    return StreamingResponse(
        sse(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/search")
def search_endpoint(
    q: str = Query(min_length=2, max_length=500),
    mode: str = "hybrid_rerank",
    tender_id: str | None = None,
    k: int = Query(5, ge=1, le=20),
):
    if mode not in MODES:
        raise HTTPException(400, f"mode must be one of {MODES}")
    hits = search(q, mode=mode, filters=Filters(tender_id=tender_id), k=k)
    return [
        {
            "chunk_id": h.chunk_id,
            "filename": h.filename,
            "tender_id": h.tender_id,
            "pages": h.pages,
            "heading": h.heading,
            "text": h.text,
            "score": h.score,
            "ranks": h.ranks,
        }
        for h in hits
    ]


@app.get("/documents")
def documents(tender_id: str | None = None):
    sql = """SELECT d.id, d.tender_id, d.filename, d.pages, d.ocr_used, d.ocr_pages, d.state, d.tender_title,
                    count(c.id) FROM document d LEFT JOIN chunk c ON c.document_id = d.id
             WHERE (%s::text IS NULL OR d.tender_id = %s) GROUP BY d.id ORDER BY d.id"""
    with db.connection() as conn:
        rows = conn.execute(sql, (tender_id, tender_id)).fetchall()
    keys = [
        "id",
        "tender_id",
        "filename",
        "pages",
        "ocr_used",
        "ocr_pages",
        "state",
        "tender_title",
        "chunks",
    ]
    return [dict(zip(keys, r, strict=True)) for r in rows]


@app.get("/health")
def health():
    checks = {}
    try:
        with db.connection() as conn:
            checks["documents"] = conn.execute("SELECT count(*) FROM document").fetchone()[0]
        checks["database"] = "ok"
    except Exception as exc:
        checks["database"] = f"error: {type(exc).__name__}"
    try:
        checks["elasticsearch"] = "ok" if es.client().ping() else "unavailable"
    except Exception:
        checks["elasticsearch"] = "unavailable"
    ok = checks.get("database") == "ok" and checks["elasticsearch"] == "ok"
    return {"status": "ok" if ok else "degraded", **checks}


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(WEB / "index.html")
