"""Grounded answers: retrieved chunks -> Claude with citations -> post-check -> answer or refusal.

Each retrieved chunk is sent as its own plain-text document with citations enabled, so
every citation Claude returns points (by document index) at exactly one chunk, and its
cited_text is the sentence it relied on. Answers carry [document, page] from that.
"""

import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Protocol

import anthropic

from docintel import grounding as grounding_check
from docintel.config import get_settings
from docintel.grounding import GroundingReport
from docintel.retrieve import Filters, Hit, search

log = logging.getLogger(__name__)

ABSTAIN = "I couldn't find this in the tender documents."

SYSTEM = f"""You answer questions about Indian government tender documents (notices inviting \
tender, bid documents, corrigenda) for contractors deciding whether and how to bid.

Use only the documents provided in the user's message. Each document is one excerpt of a \
tender file; its title gives the file name and page.

- Support every statement with a citation to the documents. Do not add facts from memory.
- Copy numbers, amounts, percentages and dates exactly as the documents write them. Do not \
calculate, convert units, round, or combine values: a separate check rejects any number \
that does not appear in the passages you cite.
- If the documents do not answer the question, reply with exactly this sentence and \
nothing else: {ABSTAIN}
- If they answer only part of it, answer that part and say what is missing.
- Be brief: one to three sentences, plain language, no headings or lists."""


@dataclass
class TextDelta:
    text: str


@dataclass
class CitationDelta:
    document_index: int
    cited_text: str


@dataclass
class FinalBlock:
    text: str
    citations: list[CitationDelta]


@dataclass
class Final:
    blocks: list[FinalBlock]
    stop_reason: str
    model: str


StreamEvent = TextDelta | CitationDelta | Final


class LLMBackend(Protocol):
    def stream(self, system: str, content: list[dict]) -> AsyncIterator[StreamEvent]: ...


class ClaudeBackend:
    """Claude Opus 5.5 with citations, adaptive thinking (always on for this model) at an
    explicit effort, and server-side refusal fallbacks."""

    def __init__(self, client: anthropic.AsyncAnthropic | None = None):
        self.client = client or anthropic.AsyncAnthropic()
        s = get_settings()
        self.model, self.effort, self.max_tokens = s.llm_model, s.llm_effort, s.llm_max_tokens

    async def stream(self, system: str, content: list[dict]) -> AsyncIterator[StreamEvent]:
        async with self.client.beta.messages.stream(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": content}],
            output_config={"effort": self.effort},
            # On a safety decline, re-run on Anthropic's recommended fallback model.
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        ) as stream:
            async for event in stream:
                if event.type == "text":
                    yield TextDelta(event.text)
                elif event.type == "citation":
                    c = event.citation
                    if getattr(c, "document_index", None) is not None:
                        yield CitationDelta(c.document_index, c.cited_text)
            final = await stream.get_final_message()
        blocks = []
        for b in final.content:
            if b.type == "text":
                cites = [
                    CitationDelta(c.document_index, c.cited_text)
                    for c in (b.citations or [])
                    if getattr(c, "document_index", None) is not None
                ]
                blocks.append(FinalBlock(b.text, cites))
        yield Final(blocks, final.stop_reason or "", final.model)


def build_content(question: str, hits: list[Hit]) -> list[dict]:
    docs = [
        {
            "type": "document",
            "source": {"type": "text", "media_type": "text/plain", "data": h.text},
            "title": f"{h.filename}, {h.pages}",
            "context": f"Tender {h.tender_id}" if h.tender_id else "Tender document",
            "citations": {"enabled": True},
        }
        for h in hits
    ]
    return [*docs, {"type": "text", "text": f"Question: {question}"}]


@dataclass
class Citation:
    chunk_id: int
    filename: str
    tender_id: str
    page_from: int
    page_to: int
    cited_text: str


@dataclass
class Answer:
    question: str
    status: str  # answered | abstained | rejected | no_context | refused | error
    text: str
    citations: list[Citation] = field(default_factory=list)
    hits: list[Hit] = field(default_factory=list)
    grounding: GroundingReport | None = None
    model: str = ""
    error: str = ""

    @property
    def abstained(self) -> bool:
        return self.status in ("abstained", "no_context", "rejected", "refused")


def is_abstention(text: str) -> bool:
    t = " ".join(text.split()).strip().lower()
    return t.startswith(ABSTAIN.lower().rstrip(".")) or t.startswith(
        "i could not find this in the tender documents"
    )


def finalize(question: str, hits: list[Hit], final: Final) -> Answer:
    text = "".join(b.text for b in final.blocks).strip()
    if final.stop_reason == "refusal":
        return Answer(question, "refused", ABSTAIN, hits=hits, model=final.model)
    if is_abstention(text):
        return Answer(question, "abstained", ABSTAIN, hits=hits, model=final.model)

    citations, cited_chunks = [], {}
    for b in final.blocks:
        for c in b.citations:
            if 0 <= c.document_index < len(hits):
                h = hits[c.document_index]
                cited_chunks[h.chunk_id] = h
                citations.append(
                    Citation(
                        h.chunk_id, h.filename, h.tender_id, h.page_from, h.page_to, c.cited_text
                    )
                )
    report = grounding_check.check(
        [(b.text, [c.cited_text for c in b.citations]) for b in final.blocks],
        [h.text for h in cited_chunks.values()],
    )
    if not citations or not report.ok:
        log.warning("answer rejected by grounding check: %s", report)
        return Answer(
            question,
            "rejected",
            ABSTAIN,
            hits=hits,
            grounding=report,
            model=final.model,
            error="the draft answer contained values not found in its cited passages",
        )
    return Answer(
        question,
        "answered",
        text,
        citations=_dedupe(citations),
        hits=hits,
        grounding=report,
        model=final.model,
    )


def _dedupe(cs: list[Citation]) -> list[Citation]:
    seen, out = set(), []
    for c in cs:
        key = (c.chunk_id, c.cited_text)
        if key not in seen:
            seen.add(key)
            out.append(c)
    return out


async def ask_stream(
    question: str,
    *,
    filters: Filters | None = None,
    backend: LLMBackend | None = None,
    mode: str = "hybrid_rerank",
) -> AsyncIterator[dict]:
    """Events for the API: retrieval, delta, citation, then exactly one 'final'."""
    hits = search(question, mode=mode, filters=filters)
    yield {
        "type": "retrieval",
        "hits": [
            {
                "chunk_id": h.chunk_id,
                "filename": h.filename,
                "tender_id": h.tender_id,
                "pages": h.pages,
                "heading": h.heading,
                "score": round(h.score, 4),
            }
            for h in hits
        ],
    }
    if not hits:
        yield {"type": "final", "answer": Answer(question, "no_context", ABSTAIN)}
        return
    backend = backend or ClaudeBackend()
    try:
        async for ev in backend.stream(SYSTEM, build_content(question, hits)):
            if isinstance(ev, TextDelta):
                yield {"type": "delta", "text": ev.text}
            elif isinstance(ev, CitationDelta):
                if 0 <= ev.document_index < len(hits):
                    h = hits[ev.document_index]
                    yield {
                        "type": "citation",
                        "filename": h.filename,
                        "pages": h.pages,
                        "cited_text": ev.cited_text,
                    }
            elif isinstance(ev, Final):
                yield {"type": "final", "answer": finalize(question, hits, ev)}
    except anthropic.APIStatusError as exc:
        log.exception("LLM request failed")
        yield {
            "type": "final",
            "answer": Answer(
                question, "error", ABSTAIN, hits=hits, error=f"LLM error {exc.status_code}"
            ),
        }
    except anthropic.APIConnectionError:
        log.exception("LLM unreachable")
        yield {
            "type": "final",
            "answer": Answer(question, "error", ABSTAIN, hits=hits, error="LLM unreachable"),
        }


async def ask(question: str, **kw) -> Answer:
    answer = None
    async for ev in ask_stream(question, **kw):
        if ev["type"] == "final":
            answer = ev["answer"]
    return answer
