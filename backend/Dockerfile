# The backend: FastAPI on port 8000. Build from the repo root (it needs data/ too):
#   docker build -f backend/Dockerfile .
# LAYA=1 adds the Laya runtime, for DECIDER=laya.
FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:0.9.5 /uv /usr/local/bin/uv
ENV UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1 PYTHONUNBUFFERED=1

WORKDIR /app/backend
ARG LAYA=0
COPY backend/pyproject.toml backend/uv.lock ./
RUN if [ "$LAYA" = "1" ]; then uv sync --frozen --no-dev --group laya; \
    else uv sync --frozen --no-dev; fi

# The backend reads the decision schema, wards and templates from the repo layout.
COPY data /app/data
COPY backend /app/backend

# Tickets and transcripts live on a volume; mount one at /var/lib/dhwani.
ENV DATABASE_URL=sqlite:////var/lib/dhwani/dhwani.db
RUN mkdir -p /var/lib/dhwani && useradd --system --home /app dhwani \
    && chown -R dhwani /var/lib/dhwani
USER dhwani

EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s --start-period=60s --retries=3 \
    CMD ["/app/backend/.venv/bin/python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"]
CMD ["/app/backend/.venv/bin/uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]
