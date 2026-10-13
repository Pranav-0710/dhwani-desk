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

Session contract:

1. Server sends `config` (`sample_rate`) once Prisma has accepted the connection, then forwards Prisma's `connected`.
2. **Start sending audio only after `connected`.** Send binary frames of 16-bit mono PCM at `config.sample_rate`, any size, **in real time**. Up to 3 s of backlog is drained slightly faster than real time; more than that ends the session with an `error`.
3. Optional text frame `flush` pushes out a trailing partial frame.
4. Server → client events: `processing`, `transcript` (`text`, `segment_index`, `audio_duration_ms`, `latency_ms`), `settings_updated`.
5. The session ends with either:
   - `closed` (`by: "prisma"`, `code`, `reason`) and a 1000 close — Prisma ended it cleanly (e.g. its 15-minute cap); reconnect for a new session.
   - `error` (`message`) and a 1011 close — missing key, rejected key, Prisma unreachable or failed, or audio not sent in real time.

   A Prisma `error` event that is *not* followed by a close is recoverable; the session continues.

Smoke test against the live API (needs `GNANI_API_KEY` in the repo-root `.env`):

```bash
uv run python -m scripts.stt_smoke                    # silence: checks key + handshake
uv run python -m scripts.stt_smoke clip.wav           # 16-bit mono WAV, 8 or 16 kHz
uv run python -m scripts.stt_smoke clip.wav --format verbatim
```

Boost words live in `gnani/stt_bias_words.json` (Kannada script, single words).
