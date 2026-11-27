# Nimbus RAG: Streamlit UI with local embedding + cross-encoder models baked in.
FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false \
    HF_HOME=/opt/hf-cache

# 1) Dependencies only (cached unless pyproject.toml / uv.lock change)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# 2) Download both local models at build time so the demo works offline
RUN uv run --no-sync python -c "\
from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); \
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

# 3) Application code and the internal knowledge base
COPY src ./src
COPY docs ./docs
COPY main.py calibrate.py README.md ./
RUN uv sync --frozen --no-dev

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"

CMD ["uv", "run", "--no-sync", "streamlit", "run", "src/app.py", \
     "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
