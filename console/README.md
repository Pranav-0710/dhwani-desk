# Console

Browser front end for Dhwani Desk: the phone page (talk to the agent from the browser mic) at `/`, and the supervisor console for officers at `/supervisor`.

## Run

Start the backend first (`cd ../backend && uv run uvicorn app:app --reload`), then:

```bash
npm install
npm run dev
```

Open the URL Vite prints (http://localhost:5173), press **Call**, allow the microphone and speak Kannada. The dev server forwards `/ws/*` and `/api/*` to the backend on port 8000.

## Checks (same as CI)

```bash
npm run lint
npm run typecheck
npm test
npm run build
```

## How the phone page works

- `src/audio/mic.ts` — captures the mic (echo cancellation on), low-pass filters and downsamples to the backend's rate (8 kHz) and sends 16-bit PCM in ~40 ms batches.
- `src/audio/player.ts` — plays the agent's streamed PCM gap-free.
- `src/call/call-client.ts` — one call over `/ws/call`. Half duplex for now: the mic sends silence while the agent speaks and until its audio has finished playing (plus 300 ms), so the echo bot never answers itself. Barge-in comes in #10.

## Supervisor console (`/supervisor`)

Open http://localhost:5173/supervisor and enter the console token (`CONSOLE_TOKEN` in the backend's `.env`; the console is off while it is empty). The token is kept for the browser tab only.

- **Live calls:** every call in progress, newest first. A call waiting for an officer is marked red, and a banner says why (safety risk, caller asked for an officer, distressed, could not route…). The console opens on that call.
- **Transcript:** the caller's words as Prisma hears them, the agent's lines, and what an officer typed, as it is spoken.
- **Decision bars:** Laya's probability for each question, with the backend's threshold marked (`/ws/console` sends them with the calls). Safety risk, a request for an officer and "not a civic matter" turn red past their threshold, because the call is then handed off or redirected. When a call is handed off on the safety step, the bars show that step until the rest of the decision arrives.
- **Officer:** **Take over** gives the officer the call. **Say** speaks the typed line in the agent's voice (up to 300 characters). **Hand back to the agent** returns the call. Each waits for the backend's answer and shows its error, e.g. a call that has just ended.
- **Tickets:** the newest 200 from `/api/tickets`, refreshed every 10 s and whenever a call ends, filterable by department, ward and urgency. Requests carry the token as `Authorization: Bearer`, for when the backend checks it.

Code: `src/supervisor/` (`protocol.ts` turns the server's messages into the page's state and is tested without a browser; `console-client.ts` is the socket; `Supervisor.tsx` is the page). The protocol itself is in `../backend/README.md`, "Supervisor console".
