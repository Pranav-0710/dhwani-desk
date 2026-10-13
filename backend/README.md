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

## Speech-to-text (Prisma)

`/ws/stt` relays browser audio to Gnani Prisma and transcripts back.

- Client → server: binary frames of 16-bit mono PCM at `GNANI_STT_SAMPLE_RATE` (8000 by default), any size; the text frame `flush` pushes out a trailing partial frame.
- Server → client: JSON events — `config` (sample rate), `connected`, `processing`, `transcript` (`text`, `segment_index`, `audio_duration_ms`, `latency_ms`), `error`.

Smoke test against the live API (needs `GNANI_API_KEY` in the repo-root `.env`):

```bash
uv run python -m scripts.stt_smoke                    # silence: checks key + handshake
uv run python -m scripts.stt_smoke clip.wav           # 16-bit mono WAV, 8 or 16 kHz
uv run python -m scripts.stt_smoke clip.wav --format verbatim
```

Boost words live in `gnani/stt_bias_words.json` (Kannada script, single words).
