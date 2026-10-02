"""Evaluation harness.

Retrieval (no LLM needed): for each answerable question, is a chunk containing the
evidence in the top 5 (recall@5), and how high is the first one (MRR@10)? Compared across
bm25 / dense / hybrid / hybrid_rerank.

Answers (needs ANTHROPIC_API_KEY): exact match of the expected value, grounding rate
(answers that pass the number/date post-check), and abstention accuracy (says "I couldn't
find this" exactly when it should).

A chunk counts as relevant when it contains the question's evidence text after
whitespace/case normalisation, so the gold labels survive re-chunking.
"""

import json
import re
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from docintel import grounding
from docintel.retrieve import MODES, Filters, search


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).lower()
    return re.sub(r"\s+", " ", s).replace("- ", "-").strip()


def load_questions(path: Path, split: str | None = None) -> list[dict]:
    qs = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return [q for q in qs if split is None or q.get("split") == split]


def relevant(text: str, q: dict) -> bool:
    """Every evidence fragment appears in the chunk. Fragments, not one exact sentence,
    because PDF tables interleave wrapped labels with their values."""
    frags = q.get("evidence") or []
    if isinstance(frags, str):
        frags = [frags]
    t = norm(text)
    return bool(frags) and all(norm(f) in t for f in frags)


@dataclass
class RetrievalScores:
    mode: str
    n: int
    recall_at_5: float
    mrr_at_10: float
    ms_per_query: float


def eval_retrieval(
    questions: list[dict], modes=MODES, *, scoped: bool = True, **search_kw
) -> list[RetrievalScores]:
    answerable = [q for q in questions if q["answerable"]]
    out = []
    for mode in modes:
        hit5 = rr = 0.0
        t0 = time.perf_counter()
        for q in answerable:
            f = Filters(tender_id=q["tender_id"]) if scoped else Filters()
            question = q["question"] if scoped else q.get("global_question", q["question"])
            hits = search(question, mode=mode, filters=f, k=10, **search_kw)
            # Across all documents a chunk only counts if it is also the right tender's:
            # "Bid validity 90 days" appears in many notices.
            ranks = [
                i
                for i, h in enumerate(hits, start=1)
                if relevant(h.text, q) and h.tender_id == q["tender_id"]
            ]
            if ranks and ranks[0] <= 5:
                hit5 += 1
            if ranks:
                rr += 1.0 / ranks[0]
        n = len(answerable)
        out.append(
            RetrievalScores(mode, n, hit5 / n, rr / n, (time.perf_counter() - t0) * 1000 / n)
        )
    return out


def values_match(expected: str, answer: str) -> bool:
    """Every value in the expected answer appears in the model's answer (as a value,
    so "Rs. 9,38,100/-" matches "₹9,38,100")."""
    exp = {(f.kind == "date", str(f.value)) for f in grounding.facts(expected)}
    got = {(f.kind == "date", str(f.value)) for f in grounding.facts(answer)}
    if not exp:  # textual answers (officer name): substring match
        return norm(expected) in norm(answer)
    return exp <= got


@dataclass
class AnswerScores:
    n: int
    answerable: int
    exact_match: float
    grounding_rate: float
    abstention_accuracy: float
    false_abstentions: int
    confident_wrong: int


async def eval_answers(questions: list[dict], ask) -> tuple[AnswerScores, list[dict]]:
    rows, em = [], 0
    answered = grounded = correct_abstain = false_abstain = wrong = 0
    for q in questions:
        a = await ask(q["question"], filters=Filters(tender_id=q["tender_id"]))
        row = {"id": q["id"], "status": a.status, "answer": a.text, "expected": q["answer"]}
        if q["answerable"]:
            if a.abstained:
                false_abstain += 1
            else:
                answered += 1
                ok = values_match(q["answer"], a.text)
                em += ok
                wrong += not ok
                grounded += bool(a.grounding and a.grounding.ok)
                row["correct"] = ok
        else:
            correct_abstain += a.abstained
            if not a.abstained:
                wrong += 1
        rows.append(row)
    n_ans = sum(q["answerable"] for q in questions)
    abst_ok = correct_abstain + (n_ans - false_abstain)
    return (
        AnswerScores(
            n=len(questions),
            answerable=n_ans,
            exact_match=em / n_ans if n_ans else float("nan"),
            grounding_rate=grounded / answered if answered else float("nan"),
            abstention_accuracy=abst_ok / len(questions) if questions else float("nan"),
            false_abstentions=false_abstain,
            confident_wrong=wrong,
        ),
        rows,
    )


def retrieval_table(scores: list[RetrievalScores]) -> str:
    lines = ["| Retriever | Recall@5 | MRR@10 | ms / query |", "|---|---:|---:|---:|"]
    for s in scores:
        lines.append(
            f"| {s.mode} | {s.recall_at_5:.1%} | {s.mrr_at_10:.3f} | {s.ms_per_query:.0f} |"
        )
    return "\n".join(lines)
