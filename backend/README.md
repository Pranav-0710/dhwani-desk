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

## Text-to-speech (Timbre)

`gnani/timbre.py` wraps Timbre v2.5. Both calls return 16-bit mono PCM at `GNANI_TTS_SAMPLE_RATE`:

- `synthesize(text)` — the whole utterance in one response (`/inference`)
- `stream(text)` — PCM chunks as they are produced (`/sse`), for low time to first audio

Settings: `GNANI_TTS_VOICE` (default `Saanvi`), `GNANI_TTS_LANG` (`kn-IN`), `GNANI_TTS_SAMPLE_RATE` (16000), `GNANI_TTS_SPEED` (0.85–1.15).

Voice audition (writes WAVs to `samples/tts/`):

```bash
uv run python -m scripts.tts_voices
uv run python -m scripts.tts_voices --voices Kavin Saanvi --rate 8000
```

Why Saanvi: `docs/decisions/tts-voice.md`.

## Call (`/ws/call`)

Everything `/ws/stt` does, plus a spoken agent reply to each transcript. For now the agent echoes the caller ("ನೀವು ಹೇಳಿದ್ದು: …"); the dialogue state machine replaces it in #9.

Extra server → browser messages, per agent line, in order:

1. `agent_text` (`text`)
2. `agent_audio_start` (`sample_rate`, from `GNANI_TTS_SAMPLE_RATE`)
3. binary frames of 16-bit mono PCM (whole samples)
4. `agent_audio_end`

A Timbre failure sends `tts_error` (`message`) before `agent_audio_end`; the call continues. Every `agent_audio_start` is followed by an `agent_audio_end`, because the phone page mutes its mic in between. When the call ends (browser leaves, or Prisma closes or fails), a line being spoken is cut off and queued lines are dropped. `config` also carries `tts_sample_rate`.

The phone page in `console/` is the browser side of this route.
