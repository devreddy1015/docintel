# Evaluation questions

`questions.jsonl` has one JSON object per line. The current file is generated from the
synthetic notices (`make corpus`). For real documents, write your own file in the same
format (the build plan asks for 50-100 questions plus ~10 unanswerable ones) and run:

```
python -m docintel.cli eval --questions data/eval/real.jsonl
python -m docintel.cli eval --questions data/eval/real.jsonl --answers   # needs ANTHROPIC_API_KEY
```

## Format

```json
{"id": "2026_AMU_926330_3:emd", "tender_id": "2026_AMU_926330_3", "split": "test",
 "question": "How much bid security has to be paid?",
 "answerable": true,
 "answer": "Rs. 16,000/-",
 "evidence": ["EMD", "16,000"]}
```

| Field | Meaning |
|---|---|
| `tender_id` | Folder name under `data/pdfs/`. The scoped benchmark searches only this tender's documents |
| `question` | Phrase it the way a contractor would, not copied from the document |
| `answerable` | `false` for questions the documents don't answer (the right response is to say so) |
| `answer` | The expected value(s). Answer exact-match compares the amounts/dates/percentages in it |
| `evidence` | Short fragments that **all** appear in the passage that answers it. Retrieval counts a chunk as correct when it contains every fragment (after lower-casing and whitespace clean-up), so keep them short: a label word plus the value |
| `global_question` | Optional: the same question naming the work, for `eval --global` |
| `split` | Optional: `train` / `test`, used by `finetune` |

## Tips for writing good questions

* Open the PDF, find a fact a bidder needs (EMD, turnover, experience, completion time,
  penalties, pre-bid meeting, documents required, payment terms) and ask for it in
  your own words.
* Include questions whose answer sits in a table, on a scanned page, and in a
  corrigendum, since those are where retrieval fails.
* For unanswerable questions, pick things plausible for that tender that the documents
  really don't mention (check by searching the PDF).
