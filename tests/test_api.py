import json

import pytest
from fastapi.testclient import TestClient

from docintel import answer as A
from docintel import api
from tests.test_answer import HITS, Scripted


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(A, "search", lambda q, mode, filters: list(HITS))
    script = Scripted(
        [A.FinalBlock("The EMD is Rs. 9,38,100.", [A.CitationDelta(0, "Rs. 9,38,100/-")])]
    )
    monkeypatch.setattr(api, "_backend", lambda: script)
    return TestClient(api.app)


def test_ask_json(client):
    r = client.post(
        "/ask?stream=false", json={"question": "What is the EMD?", "tender_id": "2026_X_1_1"}
    )
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "answered"
    assert body["citations"][0]["document"] == "NIT.pdf" and body["citations"][0]["page"] == 1


def test_ask_streams_server_sent_events(client):
    with client.stream("POST", "/ask", json={"question": "What is the EMD?"}) as r:
        assert r.headers["content-type"].startswith("text/event-stream")
        events = [line.split(": ", 1)[1] for line in r.iter_lines() if line.startswith("event: ")]
    assert events[0] == "retrieval" and events[-1] == "final" and "delta" in events


def test_ask_validates_input(client):
    assert client.post("/ask", json={"question": "x"}).status_code == 422


def test_no_api_key_is_a_clear_503(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)

    def boom():
        raise api.HTTPException(503, "LLM not configured. Set ANTHROPIC_API_KEY.")

    monkeypatch.setattr(api, "_backend", boom)
    r = TestClient(api.app).post("/ask", json={"question": "What is the EMD?"})
    assert r.status_code == 503 and "ANTHROPIC_API_KEY" in r.json()["detail"]
    _ = json  # keep import used
