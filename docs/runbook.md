# Runbook: run Dhwani Desk, and share it through a tunnel

Two containers: the backend (FastAPI, not published) and the console (nginx on port 8080, serving the phone page and the supervisor console, and forwarding `/ws` and `/api` to the backend). The browser only ever talks to port 8080, so one address is all a tunnel has to expose.

## 1. Settings

Copy `.env.example` to `.env` at the repo root and fill in:

| Variable | Needed for |
| --- | --- |
| `GNANI_API_KEY` | every call (Prisma and Timbre) |
| `CONSOLE_TOKEN` | the supervisor console and the tickets API; long and random, e.g. `python -c "import secrets; print(secrets.token_urlsafe(24))"` |
| `HF_TOKEN` (read) | the fine-tuned Laya, which `.env.example` turns on (`DECIDER=laya`, `LAYA=1`, `LAYA_CHECKPOINT_REVISION=kn-ft-v1`); set `DECIDER=keyword` and `LAYA=0` to run without it |

`.env` is never copied into an image: compose passes it to the backend when it starts.

## 2. Run it

```bash
docker compose up --build
```

- Phone: http://localhost:8080. Press **Call** and speak Kannada. A browser only allows the microphone on `localhost` or `https`.
- Console: http://localhost:8080/supervisor, with `CONSOLE_TOKEN`.
- Logs: `docker compose logs -f backend`.
- Stop: `docker compose down`. Tickets stay in the `tickets` volume. `docker compose down -v` deletes them.

With `DECIDER=laya` the first start downloads the checkpoint (about 1.3 GB, kept on the `tickets` volume) and loads it for about 15 s before the backend reports healthy. The console container waits up to 10 minutes for that; later starts only load it.

Needs Docker Compose 2.24 or newer (for the optional `.env`).

## 3. Share it with a Cloudflare quick tunnel

A quick tunnel gives a temporary `https://<random>.trycloudflare.com` address for port 8080, with no Cloudflare account.

1. Install `cloudflared` (https://developers.cloudflare.com/cloudflare-one/networks/connectors/cloudflare-tunnel/downloads/). On Windows: `winget install --id Cloudflare.cloudflared`.
2. With the app running:

   ```bash
   cloudflared tunnel --url http://localhost:8080
   ```

   It prints the address, e.g. `https://quiet-river-1234.trycloudflare.com`.
3. Allow that page to open calls. In `.env`:

   ```
   ALLOWED_ORIGINS=http://localhost:8080,https://quiet-river-1234.trycloudflare.com
   ```

   Then restart the backend so it reads it: `docker compose up -d backend`. A value exported in your shell wins over `.env`, so unset `ALLOWED_ORIGINS` there if it does not take.
4. Open the address on a phone or another laptop: the phone page at `/`, the console at `/supervisor`.
5. When the demo is over, stop `cloudflared` (Ctrl+C). The address stops working at once, and the next tunnel gets a new one: update `ALLOWED_ORIGINS` again.

## What protects a public demo

Anyone with the address can reach the app, and every call spends Gnani credits, so:

- **Only allowed pages open calls.** `/ws/stt` and `/ws/call` refuse any page not on `ALLOWED_ORIGINS` (#29).
- **Two calls at a time** (`MAX_STT_SESSIONS`). One more is told the line is busy, before Prisma is opened.
- **Silent sockets are hung up** after 30 s without audio, so nobody can hold the line open for free.
- **The console and the tickets API need `CONSOLE_TOKEN`.** The console sends it in its first message. `/api/tickets` and `/api/thresholds` need it as `Authorization: Bearer <token>`, because tickets hold where callers are.
- **The backend's port is not published.** Only the console container reaches it.

A script can forge the `Origin` header. The call cap and the silence limit are what bound it, so keep the tunnel up only while it is being used.

Also:
- The tunnel's address is hard to guess but is not a secret: anyone it is shared with can call.
- Cloudflare ends TLS for the tunnel, so the console token passes through it. Change `CONSOLE_TOKEN` after a public demo.

## When something is wrong

| Symptom | Likely cause |
| --- | --- |
| The phone page can't connect; the backend log says "Refused a call socket from origin …" | The page's address is not in `ALLOWED_ORIGINS`, or the backend was not restarted after changing it |
| "The line is busy" | Two calls are already open. Hang one up, or raise `MAX_STT_SESSIONS` |
| Console says "Token refused" | `CONSOLE_TOKEN` differs from what was typed, or is empty in `.env` |
| Console shows the calls but no tickets, "HTTP 401" | The same: the tickets API checks the token too |
| No agent voice; `tts_error` in the transcript | `GNANI_API_KEY` is wrong or out of credits |
| `docker compose up` waits on the backend | Laya is still downloading or loading. Watch `docker compose logs -f backend` |
