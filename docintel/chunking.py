"""Split extracted pages into chunks along the document's own structure.

Tender notices are organised as numbered clauses ("4.2 Financial turnover") under
uppercase headings ("ELIGIBILITY CRITERIA"). Chunking on those boundaries keeps a clause
and its numbers together, which matters more for answering "what is the EMD?" than any
fixed window size. Sections are packed into chunks of roughly target..max tokens; a
section longer than max is split on paragraph/sentence boundaries with overlap. Every
chunk records the pages it spans, so answers can cite [document, page].
"""

import re
from dataclasses import dataclass, field

from docintel.extract import Page

_NUMBERED = re.compile(r"^\s*(\d{1,2}(?:\.\d{1,2}){0,3})[.)]?\s+\S")
_KEYWORD = re.compile(r"^\s*(Section|Clause|Annexure|Schedule|Chapter|Part|Appendix)\b", re.I)
_SENTENCE = re.compile(r"(?<=[.;:])\s+(?=[A-Z0-9(])")


def approx_tokens(text: str) -> int:
    """~1.3 tokens per word for English procurement text (close enough for sizing)."""
    return max(1, round(len(text.split()) * 1.3))


def is_heading(line: str) -> bool:
    s = line.strip()
    if not s or len(s) > 90:
        return False
    if _NUMBERED.match(s) and len(s) < 90:
        return True
    if _KEYWORD.match(s):
        return True
    letters = [c for c in s if c.isalpha()]
    # Short, mostly-uppercase lines: "NOTICE INVITING TENDER", "ELIGIBILITY CRITERIA".
    return (
        len(letters) >= 6
        and sum(c.isupper() for c in letters) / len(letters) > 0.8
        and len(s.split()) <= 10
    )


@dataclass
class Section:
    heading: str
    lines: list[tuple[int, str]] = field(default_factory=list)  # (page, line)

    @property
    def text(self) -> str:
        return "\n".join(t for _, t in self.lines)


@dataclass
class Chunk:
    ord: int
    page_from: int
    page_to: int
    heading: str
    text: str
    tokens: int


_DIGITS = re.compile(r"\d+")


def running_lines(pages: list[Page]) -> set[str]:
    """Headers and footers: lines that repeat on most pages, digits ignored, so that
    "Page 2 of 9" counts as one line. Left in, they land in the middle of sentences that
    cross a page break and split them for retrieval and citation."""
    if len(pages) < 3:
        return set()
    counts: dict[str, int] = {}
    for page in pages:
        for key in {_DIGITS.sub("#", ln.strip()) for ln in page.text.splitlines() if ln.strip()}:
            counts[key] = counts.get(key, 0) + 1
    return {k for k, n in counts.items() if n >= max(3, 0.6 * len(pages))}


def sections(pages: list[Page]) -> list[Section]:
    out = [Section(heading="")]
    skip = running_lines(pages)
    for page in pages:
        for raw in page.text.splitlines():
            line = raw.rstrip()
            if not line.strip() or _DIGITS.sub("#", line.strip()) in skip:
                continue
            if is_heading(line):
                out.append(Section(heading=line.strip(), lines=[(page.number, line.strip())]))
            else:
                out[-1].lines.append((page.number, line.strip()))
    return [s for s in out if s.lines]


def _split_long(
    lines: list[tuple[int, str]], max_tokens: int, overlap: int
) -> list[list[tuple[int, str]]]:
    """Split one long section into pieces of <= max_tokens, sentence-aligned, with overlap."""
    units: list[tuple[int, str]] = []
    for page, line in lines:
        units.extend((page, s) for s in _SENTENCE.split(line) if s.strip())
    pieces, cur, cur_tokens = [], [], 0
    for unit in units:
        t = approx_tokens(unit[1])
        if cur and cur_tokens + t > max_tokens:
            pieces.append(cur)
            tail, tail_tokens = [], 0
            for u in reversed(cur):  # carry the last ~overlap tokens into the next piece
                tail_tokens += approx_tokens(u[1])
                tail.insert(0, u)
                if tail_tokens >= overlap:
                    break
            cur, cur_tokens = list(tail), tail_tokens
        cur.append(unit)
        cur_tokens += t
    if cur:
        pieces.append(cur)
    return pieces


def chunk_pages(
    pages: list[Page], *, target: int = 500, max_tokens: int = 800, overlap: int = 60
) -> list[Chunk]:
    chunks: list[Chunk] = []
    buf: list[tuple[int, str]] = []
    buf_heading = ""

    def emit(lines: list[tuple[int, str]], heading: str) -> None:
        text = "\n".join(t for _, t in lines).strip()
        if not text:
            return
        chunks.append(
            Chunk(
                ord=len(chunks),
                page_from=min(p for p, _ in lines),
                page_to=max(p for p, _ in lines),
                heading=heading,
                text=text,
                tokens=approx_tokens(text),
            )
        )

    for sec in sections(pages):
        sec_tokens = approx_tokens(sec.text)
        if sec_tokens > max_tokens:
            if buf:
                emit(buf, buf_heading)
                buf = []
            for piece in _split_long(sec.lines, max_tokens, overlap):
                # Continuation pieces repeat the heading so each chunk stands alone.
                if sec.heading and piece[0][1] != sec.heading:
                    piece = [(piece[0][0], f"{sec.heading} (continued)"), *piece]
                emit(piece, sec.heading)
            continue
        buf_tokens = approx_tokens("\n".join(t for _, t in buf)) if buf else 0
        if buf and buf_tokens + sec_tokens > max_tokens:
            emit(buf, buf_heading)
            buf = []
        if not buf:
            buf_heading = sec.heading
        buf.extend(sec.lines)
        if approx_tokens("\n".join(t for _, t in buf)) >= target:
            emit(buf, buf_heading)
            buf = []
    if buf:
        emit(buf, buf_heading)
    return chunks
