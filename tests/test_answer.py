import pytest

from docintel import answer as A
from docintel.retrieve import Hit, rrf


def hit(i, text, page=1):
    return Hit(
        chunk_id=i,
        document_id=1,
        filename="NIT.pdf",
        tender_id="2026_X_1_1",
        page_from=page,
        page_to=page,
        heading="",
        text=text,
    )


HITS = [
    hit(10, "Earnest Money Deposit (EMD): Rs. 9,38,100/- payable online.", 1),
    hit(11, "Performance Guarantee of 5% of the tendered value within 15 days.", 2),
]


class Scripted:
    """Stand-in LLM: replays fixed blocks, streaming them as Claude would."""

    def __init__(self, blocks, stop_reason="end_turn"):
        self.blocks, self.stop_reason, self.calls = blocks, stop_reason, []

    async def stream(self, system, content):
        self.calls.append((system, content))
        for b in self.blocks:
            yield A.TextDelta(b.text)
            for c in b.citations:
                yield c
        yield A.Final(self.blocks, self.stop_reason, "scripted")


@pytest.fixture
def fake_search(monkeypatch):
    monkeypatch.setattr(A, "search", lambda q, mode, filters: list(HITS))


async def test_grounded_answer(fake_search):
    b = Scripted(
        [
            A.FinalBlock(
                "The EMD is Rs. 9,38,100.",
                [A.CitationDelta(0, "Earnest Money Deposit (EMD): Rs. 9,38,100/-")],
            )
        ]
    )
    a = await A.ask("What is the EMD?", backend=b)
    assert a.status == "answered"
    assert a.citations[0].filename == "NIT.pdf" and a.citations[0].page_from == 1
    assert a.grounding.ok


async def test_wrong_number_is_rejected(fake_search):
    b = Scripted(
        [
            A.FinalBlock(
                "The guarantee is 3% of the value.",
                [A.CitationDelta(1, "Performance Guarantee of 5%")],
            )
        ]
    )
    a = await A.ask("Performance guarantee?", backend=b)
    assert a.status == "rejected"
    assert a.text == A.ABSTAIN
    assert a.grounding.unsupported == ["3%"]


async def test_abstention(fake_search):
    a = await A.ask("Price variation formula?", backend=Scripted([A.FinalBlock(A.ABSTAIN, [])]))
    assert a.status == "abstained" and a.abstained


async def test_refusal_stop_reason(fake_search):
    a = await A.ask("x?", backend=Scripted([A.FinalBlock("", [])], stop_reason="refusal"))
    assert a.status == "refused" and a.text == A.ABSTAIN


async def test_answer_without_citations_is_rejected(fake_search):
    a = await A.ask("EMD?", backend=Scripted([A.FinalBlock("The EMD is Rs. 9,38,100.", [])]))
    assert a.status == "rejected"


async def test_no_retrieval_hits_never_calls_llm(monkeypatch):
    monkeypatch.setattr(A, "search", lambda q, mode, filters: [])
    b = Scripted([])
    a = await A.ask("anything?", backend=b)
    assert a.status == "no_context" and b.calls == []


async def test_prompt_has_one_cited_document_per_chunk(fake_search):
    b = Scripted([A.FinalBlock(A.ABSTAIN, [])])
    await A.ask("EMD?", backend=b)
    system, content = b.calls[0]
    docs = [c for c in content if c["type"] == "document"]
    assert len(docs) == 2
    assert docs[0]["citations"] == {"enabled": True}
    assert docs[0]["title"] == "NIT.pdf, p. 1"
    assert content[-1]["text"] == "Question: EMD?"
    assert "Do not" in system and A.ABSTAIN in system


async def test_stream_events_order(fake_search):
    b = Scripted([A.FinalBlock("The EMD is Rs. 9,38,100.", [A.CitationDelta(0, "Rs. 9,38,100/-")])])
    types = [e["type"] async for e in A.ask_stream("EMD?", backend=b)]
    assert (
        types[0] == "retrieval"
        and types[-1] == "final"
        and "delta" in types
        and "citation" in types
    )


def test_rrf_rewards_agreement():
    fused = dict(rrf([1, 2, 3], [3, 1, 4], k=60))
    assert max(fused, key=fused.get) == 1  # rank 1 and 2 beats rank 3 and 1
    assert fused[3] > fused[2]  # in both lists beats one list
    assert 4 in fused


def test_is_abstention_tolerates_spacing():
    assert A.is_abstention("  I couldn't find this in the tender documents.  ")
    assert not A.is_abstention("The EMD is Rs. 100.")
