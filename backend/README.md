# Backend

FastAPI service: audio proxy, Gnani clients, dialogue state machine, Laya decisions, tickets.

## Setup

Requires [uv](https://docs.astral.sh/uv/). It installs Python 3.11 for this project if needed.

```bash
cd backend
uv sync
```

## Run

In Docker, with the console in front: `docker compose up --build` from the repo root (`docs/runbook.md`). For development:

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

Who may connect, to `/ws/stt` and `/ws/call` alike (every session spends Gnani credits):

- **Only pages on `ALLOWED_ORIGINS`.** Browsers apply no CORS to WebSockets, so the server checks the `Origin` header itself before accepting. Any other page, or no `Origin` at all, is refused with HTTP 403 at the handshake. The default lets in the Vite dev server (`http://localhost:5173`, `http://127.0.0.1:5173`); add the tunnel's address when the phone page is served from it, e.g. `ALLOWED_ORIGINS=http://localhost:5173,https://hosapura.example.com`.
- **At most `MAX_STT_SESSIONS` at once** (default 2), counted across both sockets. One more gets `error` ("The line is busy…") and a 1013 close, before Prisma is opened. The count is per process: `uvicorn --workers N` would allow N times as many.
- **A session that sends no audio for 30 s is ended** with `error` and a 1008 close, so a socket left open and silent cannot hold a place. The phone page streams all the time, silence included. A phone that loses signal without closing keeps its place until uvicorn's ping gives up, about 40 s.

A script can forge `Origin`, so the first rule only stops other web pages; the cap is what bounds a script that has the URL.

Session contract:

1. Server sends `config` (`sample_rate`) once Prisma has accepted the connection, then forwards Prisma's `connected`.
2. **Start sending audio only after `connected`.** Send binary frames of 16-bit mono PCM at `config.sample_rate`, any size, **in real time**. Up to 3 s of backlog is drained slightly faster than real time; more than that ends the session with an `error`.
3. Optional text frame `flush` pushes out a trailing partial frame.
4. Server → client events: `processing`, `transcript` (`text`, `segment_index`, `audio_duration_ms`, `latency_ms`), `settings_updated`.
5. The session ends with one of:
   - `closed` (`by: "prisma"`, `code`, `reason`) and a 1000 close — Prisma ended it cleanly (e.g. its 15-minute cap); reconnect for a new session.
   - `error` (`message`) and a 1011 close — missing key, rejected key, Prisma unreachable or failed, or audio not sent in real time.
   - `error` (`message`) and a 1013 close — the line is busy (`MAX_STT_SESSIONS`); try again later.
   - `error` (`message`) and a 1008 close — no audio for 30 s.

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

The lines are in `dialogue/templates/kn.yaml`. Fixed lines are synthesised once per process and replayed from memory (`audio/tts_cache.py`).

`DECIDER` picks who decides. `keyword` (the default) uses the rules in `dialogue/standin.py`. `laya` uses Laya through `decisions/laya.py`, loaded once at startup from `LAYA_CHECKPOINT_REPO` at `LAYA_CHECKPOINT_REVISION`, and needs `uv sync --group laya`. Laya answers 8 of the decision schema's questions; code answers the ward and personal numbers (`decisions/extractors.py`).

Each turn is decided in two steps, safety first:
1. The four questions that can end the call at once: danger, how urgent (an emergency counts as danger), a request for a person, not a civic matter.
2. The rest.

If step 1 calls for a hand-off, the hand-off line is said at once, and step 2 still runs for the officer. Otherwise, if step 2 takes over 0.3 s, the agent says "ಸರಿ, ಒಂದು ಕ್ಷಣ" ("okay, one moment") while it runs. Timings: `docs/decisions/laya-latency.md`.

Before the lines of each reply, the browser gets a `dialogue` message:
- `keys`: the template keys, in speaking order
- `state`: `listening`, `handed_off` or `closed`
- `reason`: why it was handed off or closed
- `slots`: department, ward, and whether a location was given
- `ticket`: when one was filed
- `decision`: every answer behind the reply
- `screening`: set instead of `decision` when the call was handed off after the safety step. Once the full decision is in, a second `dialogue` message follows with no `keys`, the `decision` and the updated `slots`; nothing more is said.

Then, per agent line, in order:

1. `agent_text` (`text`)
2. `agent_audio_start` (`sample_rate`, from `GNANI_TTS_SAMPLE_RATE`)
3. binary frames of 16-bit mono PCM (whole samples)
4. `agent_audio_end`

When the agent closes the call (ticket filed or request redirected), the server waits until the browser has played the last line, sends `call_ended` (`reason`), and closes the socket with 1000. A handed-off call stays open for the officer.

A Timbre failure sends `tts_error` (`message`) before `agent_audio_end`; the call continues. Every `agent_audio_start` is followed by an `agent_audio_end`, because the phone page mutes its mic in between. When the call ends (browser leaves, or Prisma closes or fails), a line being spoken is cut off and queued lines are dropped. `config` also carries `tts_sample_rate`.

The phone page in `console/` is the browser side of this route. To try a call without a microphone, `uv run python -m scripts.call_smoke` phones the running server with synthetic caller speech and prints the transcripts, decisions and agent lines with timings. It calls as the first page on `ALLOWED_ORIGINS` (`--origin` to change).

## Supervisor console (`/ws/console`, `/api/thresholds`)

The console watches every live call and lets an officer step in (`dialogue/live.py`). It is off until `CONSOLE_TOKEN` is set, because it can speak to callers.

1. The console's first message is `{"type": "hello", "token": "<CONSOLE_TOKEN>"}`, within 5 s. A wrong or missing token gets `error` and a close with 1008.
2. It then gets `calls`: every live call (`call_id`, `started_at`, `state`, `reason`, `officer`, `slots`) and the escalation `thresholds` for the decision bars (also at `GET /api/thresholds`, from `decisions/thresholds.py`).
3. As things happen: `call_started` (the same fields), `call_event` (`call_id`, and `event`: every JSON message that call's browser got, as listed above; no audio) and `call_closed` (`call_id`).

Commands, each with a `call_id` and an optional `id` that is echoed back:

| Command | Does | Allowed when |
| --- | --- | --- |
| `take_over` | The officer has the call; the agent stays quiet. A call the agent was still handling is handed off with reason `taken_over`. | the call is not closed |
| `say` (`text`) | The agent's voice says the officer's text (up to 300 characters) | the officer has taken the call |
| `hand_back` | The agent says "ಕಾಯ್ದಿದ್ದಕ್ಕೆ ಧನ್ಯವಾದ…" and decides on everything the caller said, including to the officer | the call is handed off |

Each command is answered with `done` or `error` (`message`), plus `command`, `call_id` and `id`. A command the console stops waiting for (after 10 s, or because it disconnected) is cancelled and never carried out later, so an officer who tries again is not heard twice.

After a hand-back, only what the caller says from then on can hand the call off again: danger, an emergency or a request for an officer, judged on those words alone. The officer has dealt with everything before, and the decider, rereading the whole call, would otherwise hand it straight back. Routing and the ticket still come from the whole call, so the ticket keeps an emergency's urgency. Caller turns said while the call is handed off are stored too, masked as if they held a personal number. `calls` and `call_started` also carry the caller's `turns` so far, for a console that joins mid-call.

Every `dialogue` message carries `officer`: whether an officer has taken the call. A console that stops reading falls behind after 1000 messages and is cut off; it reconnects and starts again from `calls`.

## Tickets (`tickets/`, `/api/tickets`)

When the agent closes a call with a ticket, it is stored in SQLite (`DATABASE_URL`, default `backend/dhwani.db`, ignored by git). It uses Python's built-in `sqlite3`, so there is no extra dependency. The schema is in `tickets/store.py`, and it is a shared contract with the console (#18) and the eval harness (#19).

| Field | Meaning |
| --- | --- |
| `number` | Ticket number read back to the caller, from 1001 |
| `department`, `ward`, `urgency` | The routing decision. `ward` is a ward id or `unknown` |
| `location_text` | The caller's own words for where the problem is (the turns that gave the place or named the ward), masked |
| `repeat_complaint`, `status` | Repeat flag; `open` until someone closes it |
| `decision`, `decider_revision` | Every answer behind the ticket with its probability, and which decider made it (`keyword-standin-1`, or `laya:<repo>@<commit>`) |
| `call_id`, `created_at` | Links the ticket to its caller turns; UTC timestamp |

Two rules protect callers:
- **Masking:** caller turns are stored masked in `transcripts`. Before anything is written, every run of 10 or more digits (phone, Aadhaar, account) is replaced with X. When the decider thinks a personal ID was said, runs of 4 or more digits are masked too.
- **Retention:** turns older than `TRANSCRIPT_RETENTION_DAYS` (default 7) are deleted at startup and every hour after. Tickets are kept.

Read-only API for the console. Each request needs the console's token, `Authorization: Bearer <CONSOLE_TOKEN>` (401 without it, 503 while `CONSOLE_TOKEN` is unset), as does `GET /api/thresholds`:
- `GET /api/tickets?limit=50`: newest first, `limit` from 1 to 500.
- `GET /api/tickets/{number}`: returns 404 when no such ticket exists.

