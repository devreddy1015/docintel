"""Fine-tune the dense retriever (multilingual-e5-small) on tender question -> chunk pairs.

Honest by construction:
  * split by TENDER: training uses the 'train' tenders, the before/after comparison
    uses only the 'test' tenders, so no test document is ever seen in training;
  * different WORDING: training uses each fact's train_question ("What is the EMD
    amount?"); the test uses the eval question ("How much bid security has to be paid?");
  * HARD NEGATIVES: for every pair, a chunk from the same tender that BM25 ranks highly
    but that does not contain the answer.

Loss: MultipleNegativesRankingLoss (in-batch negatives + the hard negative).
The fine-tuned vectors go into chunk.embedding_ft, so before and after are measured on
exactly the same chunks.
"""

import json
import logging
import random
from pathlib import Path

from docintel import db, evaluate, models, retrieve

log = logging.getLogger(__name__)


def build_triplets(questions: list[dict], seed: int = 7) -> list[dict]:
    rng = random.Random(seed)
    triplets = []
    with db.connection() as conn:
        rows = conn.execute(
            "SELECT c.id, d.tender_id, c.heading, c.text FROM chunk c JOIN document d ON d.id = c.document_id"
        ).fetchall()
    by_tender: dict[str, list[tuple[int, str]]] = {}
    for cid, tid, heading, text in rows:
        by_tender.setdefault(tid, []).append((cid, f"{heading}\n{text}" if heading else text))
    for q in questions:
        chunks = by_tender.get(q["tender_id"], [])
        pos = [t for _, t in chunks if evaluate.relevant(t, q)]
        if not pos:
            continue
        bm25_ids = retrieve.bm25(
            q["train_question"], retrieve.Filters(tender_id=q["tender_id"]), 10
        )
        text_by_id = dict(chunks)
        negs = [
            text_by_id[i]
            for i in bm25_ids
            if i in text_by_id and not evaluate.relevant(text_by_id[i], q)
        ]
        if not negs:
            negs = [t for _, t in chunks if not evaluate.relevant(t, q)]
        if not negs:
            continue
        triplets.append(
            {
                "anchor": "query: " + q["train_question"],
                "positive": "passage: " + pos[0],
                "negative": "passage: " + negs[0]
                if rng.random() < 0.8
                else "passage: " + rng.choice(negs),
            }
        )
    return triplets


def embed_column(model_path: str, column: str = "embedding_ft") -> int:
    with db.connection() as conn:
        conn.execute(f"ALTER TABLE chunk ADD COLUMN IF NOT EXISTS {column} vector(384)")
        rows = conn.execute("SELECT id, heading, text FROM chunk ORDER BY id").fetchall()
    vecs = models.embed_passages(
        [f"{h}\n{t}" if h else t for _, h, t in rows], model_name=model_path
    )
    with db.connection() as conn, conn.cursor() as cur:
        for (cid, _, _), v in zip(rows, vecs, strict=True):
            cur.execute(f"UPDATE chunk SET {column} = %s WHERE id = %s", (v, cid))
    return len(rows)


def run(questions_path: Path, out: Path, *, epochs: int = 2, batch_size: int = 16) -> dict:
    from datasets import Dataset
    from sentence_transformers import (
        SentenceTransformer,
        SentenceTransformerTrainer,
        SentenceTransformerTrainingArguments,
    )
    from sentence_transformers.losses import MultipleNegativesRankingLoss
    from sentence_transformers.training_args import BatchSamplers

    from docintel.config import get_settings

    qs = evaluate.load_questions(questions_path)
    train_q = [
        q for q in qs if q["split"] == "train" and q["answerable"] and q.get("train_question")
    ]
    test_q = [q for q in qs if q["split"] == "test"]
    triplets = build_triplets(train_q)
    log.info("training on %d triplets from %d train questions", len(triplets), len(train_q))

    base = get_settings().embedding_model
    before = {
        "scoped": evaluate.eval_retrieval(test_q, ("dense",), scoped=True)[0],
        "global": evaluate.eval_retrieval(test_q, ("dense",), scoped=False)[0],
    }

    model = SentenceTransformer(base, device="cpu")
    args = SentenceTransformerTrainingArguments(
        output_dir=str(out / "checkpoints"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        # No duplicate texts in a batch: a duplicate positive would be an in-batch
        # "negative" for its own query.
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        save_strategy="no",
        logging_steps=10,
        report_to=[],
        seed=7,
    )
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=Dataset.from_list(triplets),
        loss=MultipleNegativesRankingLoss(model),
    )
    trainer.train()
    out.mkdir(parents=True, exist_ok=True)
    model.save(str(out))

    n = embed_column(str(out))
    after = {
        "scoped": evaluate.eval_retrieval(
            test_q, ("dense",), scoped=True, model_name=str(out), column="embedding_ft"
        )[0],
        "global": evaluate.eval_retrieval(
            test_q, ("dense",), scoped=False, model_name=str(out), column="embedding_ft"
        )[0],
    }
    result = {
        "triplets": len(triplets),
        "test_questions": sum(q["answerable"] for q in test_q),
        "chunks_reembedded": n,
        **{
            f"{k}_before": {"recall@5": round(v.recall_at_5, 4), "mrr@10": round(v.mrr_at_10, 4)}
            for k, v in before.items()
        },
        **{
            f"{k}_after": {"recall@5": round(v.recall_at_5, 4), "mrr@10": round(v.mrr_at_10, 4)}
            for k, v in after.items()
        },
    }
    (out / "result.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))
    return result
