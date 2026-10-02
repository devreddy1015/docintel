COMPOSE ?= docker compose
RUN = $(COMPOSE) run --rm api python -m docintel.cli

.PHONY: up down ingest reingest eval eval-answers finetune corpus test lint

up:            ## start the API (TenderLens must be running: it provides Postgres + Elasticsearch)
	$(COMPOSE) up -d --build

down:
	$(COMPOSE) down

ingest:        ## ingest every PDF under data/pdfs (idempotent; OCR runs inside the container)
	$(RUN) ingest

reingest:      ## drop everything and ingest again
	$(RUN) ingest --reset

eval:          ## retrieval benchmark: bm25 vs dense vs hybrid vs hybrid+rerank
	$(RUN) eval
	$(RUN) eval --global

eval-answers:  ## answer benchmark (needs ANTHROPIC_API_KEY; makes API calls)
	$(RUN) eval --answers

finetune:      ## fine-tune the retriever and report before/after on held-out tenders
	$(COMPOSE) run --rm api sh -c "uv sync --frozen && python -m docintel.cli finetune"

corpus:        ## regenerate the synthetic test notices from TenderLens data
	uv run python scripts/make_synthetic_corpus.py --n 30

test:
	uv run pytest

lint:
	uv run ruff check .
	uv run ruff format --check .
