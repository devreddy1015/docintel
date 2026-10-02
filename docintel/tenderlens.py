"""Tender metadata from the TenderLens API (state, buyer, closing date) for filters."""

import logging
from dataclasses import dataclass
from datetime import datetime

import httpx

from docintel.config import get_settings

log = logging.getLogger(__name__)


@dataclass
class TenderMeta:
    tender_id: str
    title: str = ""
    buyer: str = ""
    state: str = ""
    closes_at: datetime | None = None


def lookup(tender_id: str, *, client: httpx.Client | None = None) -> TenderMeta:
    """Best effort: DocIntel still works (without filters) if TenderLens is unreachable."""
    if not tender_id:
        return TenderMeta(tender_id="")
    own = client is None
    client = client or httpx.Client(base_url=get_settings().tenderlens_url, timeout=10)
    try:
        r = client.get(
            "/api/tenders",
            params={"q": tender_id, "page_size": 5, "closes_after": "2000-01-01T00:00:00Z"},
        )
        r.raise_for_status()
        for t in r.json()["results"]:
            if t["source_tender_id"] == tender_id:
                return TenderMeta(
                    tender_id=tender_id,
                    title=t["title"],
                    buyer=(t.get("buyer") or {}).get("canonical_name") or t["buyer_raw"],
                    state=t.get("state") or "",
                    closes_at=datetime.fromisoformat(t["closes_at"]),
                )
        log.info("tender %s not found in TenderLens", tender_id)
    except (httpx.HTTPError, KeyError, ValueError) as exc:
        log.warning("TenderLens lookup for %s failed: %s", tender_id, exc)
    finally:
        if own:
            client.close()
    return TenderMeta(tender_id=tender_id)
