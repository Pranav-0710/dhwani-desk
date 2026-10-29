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

Everything `/ws/stt` does, plus the agent. It greets the caller once Prisma is connected, then answers every caller turn. A turn is the transcript segments that arrive less than 0.6 s apart (`dialogue/turns.py`).

What the agent says comes from the state machine in `dialogue/machine.py`. It is greet → listen → decide → one of these:
- hand off to an officer: a safety risk, the caller asks for a person, the caller is distressed or abusive, the complaint can't be routed, or the decider failed
- redirect a non-civic request and close
- ask for what is missing
- read back the ticket number and close

A safety risk wins over everything. A risk outside the corporation's remit, such as chest pain or a fight, gets "call 112" and an officer, not a redirect.

The lines are in `dialogue/templates/kn.yaml`. Until Laya is served (#15), the decisions come from keyword rules in `dialogue/standin.py`. Fixed lines are synthesised once per process and replayed from memory (`audio/tts_cache.py`).

Before the lines of each reply, the browser gets a `dialogue` message:
- `keys`: the template keys, in speaking order
- `state`: `listening`, `handed_off` or `closed`
- `reason`: why it was handed off or closed
- `slots`: department, ward, and whether a location was given
- `ticket`: when one was filed
- `decision`: every answer behind the reply

Then, per agent line, in order:

1. `agent_text` (`text`)
2. `agent_audio_start` (`sample_rate`, from `GNANI_TTS_SAMPLE_RATE`)
3. binary frames of 16-bit mono PCM (whole samples)
4. `agent_audio_end`

When the agent closes the call (ticket filed or request redirected), the server waits until the browser has played the last line, sends `call_ended` (`reason`), and closes the socket with 1000. A handed-off call stays open for the officer.

A Timbre failure sends `tts_error` (`message`) before `agent_audio_end`; the call continues. Every `agent_audio_start` is followed by an `agent_audio_end`, because the phone page mutes its mic in between. When the call ends (browser leaves, or Prisma closes or fails), a line being spoken is cut off and queued lines are dropped. `config` also carries `tts_sample_rate`.

The phone page in `console/` is the browser side of this route. To try a call without a microphone, `uv run python -m scripts.call_smoke` phones the running server with synthetic caller speech and prints the transcripts, decisions and agent lines with timings.

## Tickets (`tickets/`, `/api/tickets`)

When the agent closes a call with a ticket, it is stored in SQLite (`DATABASE_URL`, default `backend/dhwani.db`, ignored by git). It uses Python's built-in `sqlite3`, so there is no extra dependency. The schema is in `tickets/store.py`, and it is a shared contract with the console (#18) and the eval harness (#19).

| Field | Meaning |
| --- | --- |
| `number` | Ticket number read back to the caller, from 1001 |
| `department`, `ward`, `urgency` | The routing decision. `ward` is a ward id or `unknown` |
| `location_text` | The caller's own words for where the problem is (the turns that gave the place or named the ward), masked |
| `repeat_complaint`, `status` | Repeat flag; `open` until someone closes it |
| `decision`, `decider_revision` | Every answer behind the ticket with its probability, and which decider made it (`keyword-standin-1` until Laya, #15) |
| `call_id`, `created_at` | Links the ticket to its caller turns; UTC timestamp |

Two rules protect callers:
- **Masking:** caller turns are stored masked in `transcripts`. Before anything is written, every run of 10 or more digits (phone, Aadhaar, account) is replaced with X. When the decider thinks a personal ID was said, runs of 4 or more digits are masked too.
- **Retention:** turns older than `TRANSCRIPT_RETENTION_DAYS` (default 7) are deleted at startup and every hour after. Tickets are kept.

Read-only API for the console:
- `GET /api/tickets?limit=50`: newest first, `limit` from 1 to 500.
- `GET /api/tickets/{number}`: returns 404 when no such ticket exists.

The console's Vite dev server proxies only `/ws` today, so `/api` needs adding there.
