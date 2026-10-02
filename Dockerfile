FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv PATH=/opt/venv/bin:$PATH HF_HOME=/opt/hf
# OCR for scanned pages and broken Hindi text layers: Tesseract (English + Hindi) via ocrmypdf.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ocrmypdf tesseract-ocr tesseract-ocr-eng tesseract-ocr-hin ghostscript \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.8 /uv /usr/local/bin/uv
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
# Bake the models into the image so the API starts without network access.
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('intfloat/multilingual-e5-small'); CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"
COPY . .
RUN useradd --system --create-home --uid 1001 app && chown -R app /app /opt/hf
USER app
EXPOSE 8100
CMD ["sh", "-c", "python -m docintel.cli migrate && uvicorn docintel.api:app --host 0.0.0.0 --port 8100 --workers 2"]
