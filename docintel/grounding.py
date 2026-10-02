"""Post-check: every number, amount, percentage and date in an answer must appear in the
passages it cites. An answer that fails is rejected, never shown as if it were true.

This is the same idea as CONTROL ARM's grounded-language invariant: generation is not
trusted on its own; a deterministic check sits between the model and the user.

Values are compared after normalisation, so "₹16 lakh" matches "16,00,000",
"7 October 2026" matches "07-Oct-2026", and "2.0%" matches "2 %".
"""

import re
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal, InvalidOperation

MONTHS = {
    m: i
    for i, m in enumerate(
        ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"],
        start=1,
    )
}
_MON = r"(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"

DATE_PATTERNS = [
    # 07-Oct-2026, 7 Oct 2026, 7th October, 2026
    (
        re.compile(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?[\s\-./]+{_MON}[\s\-.,/]+(\d{{4}})\b", re.I),
        "dmy_name",
    ),
    # October 7, 2026
    (re.compile(rf"\b{_MON}\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+(\d{{4}})\b", re.I), "mdy_name"),
    # 07/10/2026, 07-10-2026, 07.10.2026 (Indian documents are day-first)
    (re.compile(r"\b(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{4})\b"), "dmy_num"),
    # 2026-10-07
    (re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b"), "ymd"),
]
MONEY = re.compile(
    r"(?:₹|rs\.?|inr)\s*([\d,]+(?:\.\d+)?)\s*(lakhs?|lacs?|crores?|cr\b|l\b)?"
    r"|([\d,]+(?:\.\d+)?)\s*(lakhs?|lacs?|crores?)\b",
    re.I,
)
PERCENT = re.compile(r"(\d+(?:\.\d+)?)\s*(?:%|per\s*cent|percent)", re.I)
NUMBER = re.compile(r"(?<![\w.])(\d[\d,]*(?:\.\d+)?)(?![\w])")
MULTIPLIER = {"lakh": 100_000, "lac": 100_000, "l": 100_000, "crore": 10_000_000, "cr": 10_000_000}


def _num(text: str) -> Decimal | None:
    try:
        return Decimal(text.replace(",", ""))
    except InvalidOperation:
        return None


def _mult(unit: str | None) -> int:
    if not unit:
        return 1
    u = unit.lower().rstrip("s")
    return MULTIPLIER.get(u, MULTIPLIER.get(u.rstrip("e"), 1))


@dataclass(frozen=True)
class Fact:
    kind: str  # date | money | percent | number
    value: object
    text: str


def _parse_date(kind: str, m: re.Match) -> date | None:
    try:
        if kind == "dmy_name":
            return date(int(m.group(3)), MONTHS[m.group(2)[:3].lower()], int(m.group(1)))
        if kind == "mdy_name":
            return date(int(m.group(3)), MONTHS[m.group(1)[:3].lower()], int(m.group(2)))
        if kind == "dmy_num":
            return date(int(m.group(3)), int(m.group(2)), int(m.group(1)))
        if kind == "ymd":
            return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    except (ValueError, KeyError):
        return None
    return None


def facts(text: str) -> list[Fact]:
    """Dates first, then money, percentages, then any remaining numbers, consuming
    the matched spans so "07-Oct-2026" is one date, not the numbers 7 and 2026."""
    out: list[Fact] = []
    taken = [False] * len(text)

    def free(m: re.Match) -> bool:
        return not any(taken[m.start() : m.end()])

    def take(m: re.Match) -> None:
        for i in range(m.start(), m.end()):
            taken[i] = True

    for rx, kind in DATE_PATTERNS:
        for m in rx.finditer(text):
            if free(m) and (d := _parse_date(kind, m)):
                out.append(Fact("date", d, m.group(0)))
                take(m)
    for m in MONEY.finditer(text):
        if not free(m):
            continue
        digits, unit = (m.group(1), m.group(2)) if m.group(1) else (m.group(3), m.group(4))
        n = _num(digits)
        if n is not None:
            out.append(Fact("money", n * _mult(unit), m.group(0)))
            take(m)
    for m in PERCENT.finditer(text):
        if free(m) and (n := _num(m.group(1))) is not None:
            out.append(Fact("percent", n.normalize(), m.group(0)))
            take(m)
    for m in NUMBER.finditer(text):
        if free(m) and (n := _num(m.group(1))) is not None:
            out.append(Fact("number", n.normalize(), m.group(0)))
            take(m)
    return out


def _values(source_text: str) -> set:
    """Every value in the sources, in all the forms an answer may use for it."""
    vals: set = set()
    for f in facts(source_text):
        if f.kind == "date":
            vals.add(("date", f.value))
        else:
            # A source "₹16,000" supports an answer "16,000" and vice versa; a source
            # "2.0%" supports "2%". Kinds other than dates are compared by value.
            vals.add(("num", Decimal(f.value).normalize()))
    return vals


@dataclass
class GroundingReport:
    ok: bool
    checked: int
    unsupported: list[str] = field(default_factory=list)
    uncited_claims: list[str] = field(default_factory=list)


def check(blocks: list[tuple[str, list[str]]], source_texts: list[str]) -> GroundingReport:
    """blocks: (answer text, cited passages) per answer block, as the Citations API returns
    them. source_texts: full text of every chunk the answer cited.

    A value must appear in what was cited (the cited sentences, or the chunks they come
    from). A block that states a value but cites nothing fails outright.
    """
    supported = _values("\n".join(source_texts))
    report = GroundingReport(ok=True, checked=0)
    for text, cited in blocks:
        fs = facts(text)
        if not fs:
            continue
        if not cited:
            report.uncited_claims.append(text.strip()[:120])
        local = _values("\n".join(cited)) | supported if cited else set()
        for f in fs:
            report.checked += 1
            key = ("date", f.value) if f.kind == "date" else ("num", Decimal(f.value).normalize())
            if key not in local:
                report.unsupported.append(f.text)
    report.ok = not report.unsupported and not report.uncited_claims
    return report
