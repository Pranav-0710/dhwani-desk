# Backend

FastAPI service: audio proxy, Gnani clients, dialogue state machine, Laya decisions, tickets.

## Setup

Requires [uv](https://docs.astral.sh/uv/). It installs Python 3.11 for this project if needed.

```bash
cd backend
uv sync
```

## Run

```bash
uv run uvicorn app:app --reload
```

Health check: <http://localhost:8000/health>

## Checks (same as CI)

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest -q
```
