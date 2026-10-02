"""Command line: python -m docintel.cli <command> ...

migrate                         create the database tables (idempotent)
ingest [FOLDER]                 ingest every PDF under FOLDER (default data/pdfs)
search QUERY [--mode M] [--tender ID]
ask QUESTION [--tender ID]      grounded answer (needs ANTHROPIC_API_KEY)
eval [--split S] [--answers] [--global]
finetune                        fine-tune the retriever (see docintel/finetune.py)
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

from docintel import db, es


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    for noisy in ("httpx", "elastic_transport", "sentence_transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    ap = argparse.ArgumentParser(prog="docintel")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("migrate")
    p = sub.add_parser("ingest")
    p.add_argument("folder", nargs="?", type=Path)
    p.add_argument("--reset", action="store_true", help="delete all documents first")
    p = sub.add_parser("search")
    p.add_argument("query")
    p.add_argument("--mode", default="hybrid_rerank")
    p.add_argument("--tender")
    p.add_argument("-k", type=int, default=5)
    p = sub.add_parser("ask")
    p.add_argument("question")
    p.add_argument("--tender")
    p = sub.add_parser("eval")
    p.add_argument("--questions", type=Path, default=Path("data/eval/questions.jsonl"))
    p.add_argument("--split")
    p.add_argument(
        "--answers", action="store_true", help="also evaluate LLM answers (costs API calls)"
    )
    p.add_argument("--limit", type=int, help="answer evaluation: first N questions only")
    p.add_argument(
        "--global",
        dest="global_",
        action="store_true",
        help="search all documents, not just the question's tender",
    )
    p.add_argument("--model", help="embedding model to evaluate (e.g. a fine-tuned one)")
    p.add_argument(
        "--column", default="embedding", help="chunk column holding that model's vectors"
    )
    p = sub.add_parser("finetune")
    p.add_argument("--questions", type=Path, default=Path("data/eval/questions.jsonl"))
    p.add_argument("--out", type=Path, default=Path("data/models/e5-small-tenders"))
    p.add_argument("--epochs", type=int, default=2)
    args = ap.parse_args(argv)

    if args.cmd == "migrate":
        db.migrate()
        es.ensure_index()
        print("ok")
    elif args.cmd == "ingest":
        from docintel import ingest

        db.migrate()
        es.ensure_index(recreate=args.reset)
        if args.reset:
            with db.connection() as conn:
                conn.execute("TRUNCATE document CASCADE")
        res = ingest.ingest_folder(args.folder)
        by = {}
        for r in res:
            by[r.status] = by.get(r.status, 0) + 1
        print(
            json.dumps(by),
            f"chunks={sum(r.chunks for r in res)}",
            f"ocr_pages={sum(r.ocr_pages for r in res)}",
        )
        for r in res:
            if r.status == "failed":
                print(f"  FAILED {r.path}: {r.error}", file=sys.stderr)
    elif args.cmd == "search":
        from docintel.retrieve import Filters, search

        for h in search(
            args.query, mode=args.mode, filters=Filters(tender_id=args.tender), k=args.k
        ):
            print(
                f"{h.score:8.3f}  {h.filename} {h.pages}  [{h.heading[:40]}]  {h.text[:110]!r}  ranks={h.ranks}"
            )
    elif args.cmd == "ask":
        from docintel.answer import ask
        from docintel.retrieve import Filters

        a = asyncio.run(ask(args.question, filters=Filters(tender_id=args.tender)))
        print(f"[{a.status}] {a.text}")
        for c in a.citations:
            print(f"  - {c.filename}, p. {c.page_from}: {c.cited_text!r}")
        if a.grounding and not a.grounding.ok:
            print(f"  grounding check failed: {a.grounding}")
        if a.error:
            print(f"  {a.error}")
    elif args.cmd == "eval":
        from docintel import evaluate

        qs = evaluate.load_questions(args.questions, args.split)
        kw = {"model_name": args.model, "column": args.column} if args.model else {}
        modes = ("dense",) if args.model else evaluate.MODES
        scores = evaluate.eval_retrieval(qs, modes, scoped=not args.global_, **kw)
        print(
            f"{len(qs)} questions ({sum(q['answerable'] for q in qs)} answerable), "
            f"{'all documents' if args.global_ else 'scoped to each question’s tender'}"
        )
        print(evaluate.retrieval_table(scores))
        if args.answers:
            from docintel.answer import ask

            subset = qs[: args.limit] if args.limit else qs
            s, rows = asyncio.run(evaluate.eval_answers(subset, ask))
            Path("data/eval/answers.jsonl").write_text(
                "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
            )
            print(json.dumps(s.__dict__, indent=1))
    elif args.cmd == "finetune":
        from docintel import finetune

        finetune.run(args.questions, args.out, epochs=args.epochs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
