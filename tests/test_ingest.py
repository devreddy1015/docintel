"""Integration: real Postgres (pgvector) + Elasticsearch + the e5 model. Uses a separate
database and index, and skips when the services aren't reachable."""

import shutil

import psycopg
import pymupdf
import pytest

from docintel import config

pytestmark = [pytest.mark.models, pytest.mark.es]

NIT = """NOTICE INVITING TENDER
Online bids are invited for Construction of boundary wall at Bhilai.
1. EARNEST MONEY DEPOSIT
Earnest Money Deposit (EMD): Rs. 16,000/- payable online.
2. ELIGIBILITY CRITERIA
Average annual financial turnover should be at least Rs. 2,40,000/-.
3. COMPENSATION FOR DELAY
Compensation at the rate of 0.5% of the tendered value per week of delay shall be levied."""


@pytest.fixture
def services(monkeypatch, tmp_path):
    url = "postgresql://tenderlens:tenderlens@localhost:5432/docintel_test"
    monkeypatch.setenv("DOCINTEL_DATABASE_URL", url)
    monkeypatch.setenv("DOCINTEL_ES_INDEX", "docintel_test_chunks")
    monkeypatch.setenv("DOCINTEL_TENDERLENS_URL", "http://127.0.0.1:9")  # unreachable: no metadata
    config.get_settings.cache_clear()
    from docintel import db, es

    db.pool.cache_clear()
    es.client.cache_clear()
    try:
        db.migrate(url)
        es.ensure_index(recreate=True)
    except (psycopg.OperationalError, Exception) as exc:  # noqa: BLE001
        pytest.skip(f"services not reachable: {exc}")
    with db.connection() as conn:
        conn.execute("TRUNCATE document CASCADE")
    yield tmp_path
    db.pool().close()
    db.pool.cache_clear()
    es.client().indices.delete(index="docintel_test_chunks", ignore_unavailable=True)
    config.get_settings.cache_clear()


def make_pdf(path, text):
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_textbox(pymupdf.Rect(50, 50, 550, 800), text, fontsize=10)
    doc.save(path)


def test_ingest_is_idempotent_and_searchable(services):
    from docintel import db, ingest
    from docintel.retrieve import Filters, search

    folder = services / "2026_TEST_000001_1"
    folder.mkdir()
    pdf = folder / "NIT.pdf"
    make_pdf(pdf, NIT)
    first = ingest.ingest_folder(services)
    assert [r.status for r in first] == ["added"]
    assert first[0].chunks >= 1

    # Same bytes under another name: skipped, not duplicated.
    shutil.copy(pdf, folder / "NIT copy.pdf")
    second = ingest.ingest_folder(services)
    assert sorted(r.status for r in second) == ["skipped", "skipped"]
    with db.connection() as conn:
        assert conn.execute("SELECT count(*) FROM document").fetchone()[0] == 1
        doc = conn.execute("SELECT tender_id, pages, ocr_used FROM document").fetchone()
    assert doc == ("2026_TEST_000001_1", 1, False)

    for mode in ("bm25", "dense", "hybrid", "hybrid_rerank"):
        hits = search(
            "How much bid security is payable?",
            mode=mode,
            filters=Filters(tender_id="2026_TEST_000001_1"),
            k=3,
        )
        assert hits and hits[0].tender_id == "2026_TEST_000001_1", mode
    top = search(
        "How much bid security is payable?", filters=Filters(tender_id="2026_TEST_000001_1"), k=1
    )[0]
    assert "16,000" in top.text


def test_delete_document_removes_both_stores(services):
    from docintel import db, es, ingest
    from docintel.config import get_settings

    make_pdf(services / "2026_TEST_000002_1.pdf", NIT)
    r = ingest.ingest_file(services / "2026_TEST_000002_1.pdf")
    ingest.delete_document(r.document_id)
    with db.connection() as conn:
        assert conn.execute("SELECT count(*) FROM chunk").fetchone()[0] == 0
    assert es.client().count(index=get_settings().es_index)["count"] == 0
