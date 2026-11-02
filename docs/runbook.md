# Runbook: run Dhwani Desk, and share it through a tunnel

Two containers: the backend (FastAPI, not published) and the console (nginx on port 8080, serving the phone page and the supervisor console, and forwarding `/ws` and `/api` to the backend). The browser only ever talks to port 8080, so one address is all a tunnel has to expose.

## 1. Settings

Copy `.env.example` to `.env` at the repo root and fill in:

| Variable | Needed for |
| --- | --- |
| `GNANI_API_KEY` | every call (Prisma and Timbre) |
| `CONSOLE_TOKEN` | the supervisor console and the tickets API; long and random (the backend refuses one under 20 characters), e.g. `python -c "import secrets; print(secrets.token_urlsafe(24))"` |
| `HF_TOKEN` (read, limited to the checkpoint repo) | the fine-tuned Laya, which `.env.example` turns on (`DECIDER=laya`, `LAYA=1`, `LAYA_CHECKPOINT_REVISION=kn-ft-v1`); set `DECIDER=keyword` and `LAYA=0` to run without it |

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
3. Allow that page to open calls and the console. In `.env`:

   ```
   ALLOWED_ORIGINS=http://localhost:8080,https://quiet-river-1234.trycloudflare.com
   ```

   Then restart the backend so it reads it: `docker compose up -d backend`. A value exported in your shell wins over `.env`, so unset `ALLOWED_ORIGINS` there if it does not take.
4. Open the address on a phone or another laptop: the phone page at `/`. Open the console on this machine, at http://localhost:8080/supervisor, unless an officer has to be elsewhere (see "Operating the console").
5. When the demo is over, stop `cloudflared` (Ctrl+C). The address stops working at once, and the next tunnel gets a new one: update `ALLOWED_ORIGINS` again.

## What protects a public demo

Anyone with the address can reach the app, and every call spends Gnani credits, so (threats and what is left: `docs/decisions/security-model.md`):

- **Only allowed pages open calls or the console.** `/ws/call` and `/ws/console` refuse any page not on `ALLOWED_ORIGINS` (#29).
- **`/ws/stt` is off.** The raw transcription relay answers nothing unless `STT_RELAY_ENABLED=true`. The phone page does not use it.
- **Two calls at a time** (`MAX_STT_SESSIONS`). One more is told the line is busy, before Prisma is opened.
- **A credit ceiling.** Once the lines have been in use for `MAX_CALL_MINUTES_PER_DAY` minutes (240) in the last 24 hours, new calls are refused; calls in progress finish. One caller may start `MAX_CALLS_PER_CLIENT_PER_HOUR` calls (30) an hour, and nginx allows one new call socket every 2 s per client (bursts of 10). Set the ceiling from the credits you can spare: 240 minutes is about 120 two-minute calls, well above a day of rehearsing and filming.
- **Silent sockets are hung up.** A session needs to send at least `MIN_AUDIO_SHARE` (a quarter) of real time in each 30 s. The phone page sends all of it, silence included, so nobody can hold the line with a byte now and then.
- **Port 8080 is bound to localhost.** Other devices reach the app through the tunnel, not over the local network.
- **One call per caller, ten minutes at most.** Each IP address (Cloudflare's client IP, passed by nginx as `X-Real-IP`; all of an IPv6 /64 counts as one) may hold one of the `MAX_STT_SESSIONS` lines at a time (`MAX_SESSIONS_PER_CLIENT`), and a call ends after 10 minutes, so one script cannot hold every line.
- **The token routes are rate-limited.** nginx allows 5 requests a second to `/api/` (bursts of 20) and one console connection a second (bursts of 5) per client, keyed on the IP Cloudflare passes in `CF-Connecting-IP`; more get HTTP 429. Sockets yet to give the token are capped too (`MAX_CONSOLE_HELLOS`, `MAX_CONSOLE_HELLOS_PER_CLIENT`), apart from consoles signed in (`MAX_CONSOLES`), so they cannot lock an officer out.
- **The console and the tickets API need `CONSOLE_TOKEN`.** The console sends it in its first message. `/api/tickets` and `/api/thresholds` need it as `Authorization: Bearer <token>`, because tickets hold where callers are.
- **The backend's port is not published.** Only the console container reaches it.

A script can forge the `Origin` header, and several IP addresses get past every per-caller limit. The line cap and the credit ceiling are what bound it, so keep the tunnel up only while it is being used.

Reaching the ceiling refuses real callers too. If it happens during a demo, `docker compose restart backend` starts the counts afresh; all of them are in memory. Stop the tunnel first if someone is draining it.

Also:
- The tunnel's address is hard to guess but is not a secret: anyone it is shared with can call.
- Cloudflare ends TLS for the tunnel, so the console token passes through it. Change `CONSOLE_TOKEN` after a public demo.
- `docker compose logs backend` shows a line for each console that signs in, with its address. One you do not recognise means the token has leaked: change it and restart the backend.

## Operating the console

The phone page and the console share one public address, and `CONSOLE_TOKEN` is all that separates them. In order of preference:

1. **Keep officers on this machine.** Set `CONSOLE_LOCAL_ONLY=true` and open http://localhost:8080/supervisor. The console socket and the API then refuse anything that came through the tunnel, right token or not, so a token seen on a shared screen is useless from elsewhere. Callers are not affected. It goes by the `CF-Connecting-IP` header, so it only means this with a Cloudflare tunnel. Use this for filming and for any demo run from one laptop.
2. **If an officer must be remote, use a named tunnel with Cloudflare Access.** A quick tunnel's `trycloudflare.com` address belongs to no account of ours, so Access cannot sit in front of it. With a domain on a Cloudflare account, create a named tunnel and an Access policy for `/supervisor`, `/ws/console` and `/api/` that admits only the officers' emails. Leave `/` and `/ws/call` open for callers. Keep the token as well.
3. **Otherwise, a remote console over the quick tunnel on the token alone.** Leave `CONSOLE_LOCAL_ONLY=false`, use a freshly generated token, send it to the officer directly, never show it on a shared screen, and change it afterwards.

The `/supervisor` page itself holds no data, so leaving it reachable gives nothing away.

## When something is wrong

| Symptom | Likely cause |
| --- | --- |
| The phone page or console can't connect; the backend log says "Refused a socket from origin …" | The page's address is not in `ALLOWED_ORIGINS`, or the backend was not restarted after changing it. `http://127.0.0.1:8080` and `http://localhost:8080` are different addresses |
| "The line is busy" | Two calls are already open. Hang one up, or raise `MAX_STT_SESSIONS` |
| "The helpline has taken all the calls it can for now"; the backend log says "the credit ceiling is reached" | `MAX_CALL_MINUTES_PER_DAY` or `MAX_CALLS_PER_CLIENT_PER_HOUR` was reached. The log says which. `docker compose restart backend` starts them afresh |
| The console says "Disconnected from the backend" through the tunnel, with the right token | `CONSOLE_LOCAL_ONLY=true` refuses the tunnel on purpose. Open it on this machine |
| "Too many consoles are open" | `MAX_CONSOLES` are signed in. Close one, or raise it |
| Console says "Token refused" | `CONSOLE_TOKEN` differs from what was typed, or is empty in `.env` |
| Console shows the calls but no tickets, "HTTP 401" | The same: the tickets API checks the token too |
| No agent voice; `tts_error` in the transcript | `GNANI_API_KEY` is wrong or out of credits |
| `docker compose up` waits on the backend | Laya is still downloading or loading. Watch `docker compose logs -f backend` |
