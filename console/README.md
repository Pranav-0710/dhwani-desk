# Console

Browser front end for Dhwani Desk: the phone page (talk to the agent from the browser mic) and, later, the supervisor console.

## Run

Start the backend first (`cd ../backend && uv run uvicorn app:app --reload`), then:

```bash
npm install
npm run dev
```

Open the URL Vite prints (http://localhost:5173), press **Call**, allow the microphone and speak Kannada. The dev server forwards `/ws/*` to the backend on port 8000.

## Checks (same as CI)

```bash
npm run lint
npm run typecheck
npm test
npm run build
```

## How the phone page works

- `src/audio/mic.ts` — captures the mic (echo cancellation on), low-pass filters and downsamples to the backend's rate (8 kHz), mixes in any line noise and sends 16-bit PCM in ~40 ms batches.
- `src/audio/noise.ts` — the **Line noise** toggle: pink background noise at about −34 dBFS ("Quiet street") or −24 dBFS ("Busy street"). It is added after the browser's own noise suppression, so speech recognition hears all of it, and can be changed during a call.
- `src/audio/player.ts` — plays the agent's streamed PCM gap-free.
- `src/call/call-client.ts` — one call over `/ws/call`. Half duplex for now: the mic sends silence while the agent speaks and until its audio has finished playing (plus 300 ms), so the echo bot never answers itself. Barge-in comes in #10.
