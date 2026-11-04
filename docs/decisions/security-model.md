# Security model: what a public tunnel exposes, and what is defended

Dhwani Desk is a competition demo with synthetic data and a fictional city, shared through a Cloudflare quick tunnel. This note says what an attacker can reach, what stops them, and what still would not. It does not claim the system is safe against a determined attacker: the controls below are sized to protect a paid API key and a demo for a few weeks. How to run it is in `docs/runbook.md`.

## What is worth protecting

| Asset | Why it matters here |
| --- | --- |
| Gnani credits, via `GNANI_API_KEY` | Real money, and the demo stops when they run out. The asset most likely to be attacked |
| The demo staying up | A line held or a backend killed during judging costs more than any data here |
| Speaking to callers (`CONSOLE_TOKEN`) | The token puts words in the agent's voice |
| Tickets and transcripts | Synthetic today. Built as if real (`docs/PRD.md`, "Privacy and compliance"), because a stray caller may say something real |
| `HF_TOKEN` and the checkpoint | Reads a private model. Whoever can change the checkpoint changes every decision |

## Trust boundaries

```
caller's browser ──┐                      ┌── /ws/call ──> Prisma, Timbre (our key)
                   ├─ Cloudflare ─ nginx ─┤
officer's browser ─┘   (ends TLS)  :8080  └── /ws/console, /api/* ──> SQLite, live calls
```

1. **Internet to nginx.** Nothing is trusted except `CF-Connecting-IP`, which Cloudflare sets itself. Port 8080 is bound to localhost, so the tunnel is the only way in from elsewhere.
2. **nginx to backend.** The backend's port is not published. It trusts `X-Real-IP` because only nginx can reach it. Run without nginx, that header could be forged: development only.
3. **Anonymous to officer.** `CONSOLE_TOKEN` is the only thing between the two, on the same public origin, unless `CONSOLE_LOCAL_ONLY` is on.
4. **Backend to Gnani and Hugging Face.** Audio and agent lines go out; transcripts and PCM come back. Vendor text is treated as untrusted for display: it is logged, not passed on.
5. **Caller's words to the rest of the system.** Transcripts are attacker-controlled text. They reach Laya, SQLite and the console, and never a language model, a shell or a query string.

## Who we assume

- **Anonymous caller.** Has the link and a browser. Can talk for a long time, say anything, and file tickets.
- **Forged-Origin script.** Has the link and a WebSocket library, so `Origin` means nothing to it. Can send any frames, and may have several IP addresses.
- **Someone with a leaked token.** Saw it on a shared screen or in a recording. Is an officer as far as the system can tell.
- **Malicious transcript or decision.** Text shaped to attack the console, or speech shaped to steer Laya.

## Threats, by real risk for this demo

Status: **fixed here**, **mitigated** (already in place before this note), or **accepted**.

| # | Threat | Risk | Status | What remains |
| --- | --- | --- | --- | --- |
| 1 | A script holds both lines all day and spends credits | High | **Fixed here.** `MAX_CALL_MINUTES_PER_DAY` caps line time in a rolling 24 hours; before, nothing capped the total (about 48 line-hours a day) | Up to the ceiling can still be spent, and reaching it refuses real callers too. Counted in memory, so a backend crash resets it |
| 2 | One byte and a `flush`, repeated, queues frames without limit and runs the backend out of memory, ending every call | High | **Fixed here.** `flush` now counts against the 3 s backlog | |
| 3 | A leaked token speaks to callers and reads tickets | High impact, low likelihood | **Accepted, narrowed.** One shared secret, no per-officer identity. Sign-ins are logged with their address, queued lines are capped, numbers are masked, and `CONSOLE_LOCAL_ONLY` makes a leaked token useless through the tunnel | With `CONSOLE_LOCAL_ONLY` off, a leaked token works until it is changed. Lines spoken are not attributed to anyone |
| 4 | `/ws/stt` as a free transcription service on our key | Medium | **Fixed here.** Off unless `STT_RELAY_ENABLED`; nothing in the console uses it | `/ws/call` also returns transcripts, so a script can still use it that way, inside every limit in #1 |
| 5 | One IPv6 /64 poses as many callers, passing the one-call-per-caller limit | Medium | **Fixed here.** A /64 counts as one caller in the backend | nginx's rate limits still key on the full address. Many real IP addresses defeat all per-caller limits; only the totals hold |
| 6 | Repeated connects open a Prisma session each, or use up the ceiling | Medium | **Fixed here.** nginx limits new call sockets per client, and `MAX_CALLS_PER_CLIENT_PER_HOUR` limits calls | Many addresses can still churn sessions. Whether Gnani charges for an unused session is not known |
| 7 | Strangers hold unauthenticated `/ws/console` sockets, or any web page tries tokens from a visitor's browser | Medium | **Fixed here.** Origin checked before accepting; sockets yet to say hello have their own caps, apart from signed-in consoles, so they cannot lock an officer out; signed-in consoles are capped | Forged Origin passes, within the caps. Holding all hello places needs many addresses and only blocks new sign-ins |
| 8 | Personal numbers shown raw on the console, or on a filmed screen | Medium | **Fixed here.** Consoles get transcripts and mid-call turns with runs of four or more digits masked, together over recent segments | A number split into pieces under four digits shows its first piece before the rest is recognised. The caller's own browser gets their own words as heard |
| 9 | Prisma's own error text, which may name the account or quota, reaches anonymous browsers and consoles | Low to medium | **Fixed here.** A recoverable `error` event sends fixed text, `closed` drops the reason; both are logged | |
| 10 | A byte every 29 s holds a line for free | Low to medium | **Fixed here.** A session must send `MIN_AUDIO_SHARE` of real time | A script streaming real-time silence still holds a line, up to the 10-minute cap, charged to #1 |
| 11 | Each Prisma reconnect restarts the 10-minute limit | Low | **Fixed here.** One limit per call | |
| 12 | 16 MiB frames and deeply nested JSON before any token check | Low | **Fixed here.** `WS_MAX_FRAME_BYTES` in the image; the hello's size is checked before parsing, and a too-deep hello no longer raises out of the route | |
| 13 | A ticket's `location_text` kept forever | Low (synthetic data) | **Fixed here.** Blanked after `TICKET_LOCATION_RETENTION_DAYS`. The ticket stays | Deleted text can stay in SQLite's free pages until they are reused |
| 14 | A leaked token queues officer lines without limit, each spending Timbre credits | Low | **Fixed here.** `MAX_OFFICER_LINES_QUEUED` | |
| 15 | A rejected `CONSOLE_TOKEN` printed in the startup log | Low | **Fixed here.** Settings errors hide their input | |
| 16 | A stray `backend/.env` copied into the image: `.dockerignore` only covered the repo root | Low | **Fixed here.** Matches at any depth | |
| 17 | The token gate names the variable and the file that hold the token | Informational | **Fixed here** | |
| 18 | The phone page and console share a public origin | Medium | **Accepted, with a switch.** See the runbook, "Operating the console". `/supervisor` itself is a static page with no data | A quick tunnel's address belongs to no account of ours, so Cloudflare Access cannot sit in front of it |
| 19 | Caller's speech steers Laya: a false emergency pages officers, or wording evades the safety step | Medium | **Accepted.** Inherent in a voice classifier; thresholds are measured in `eval/reports/`. No language model and no tools, so wrong answers change routing only | The phone page's socket also carries each decision's probabilities, which helps someone tuning their wording. `scripts/call_smoke.py` and the console's copy use the same message |
| 20 | A caller's long calls slow another caller's decisions (Laya runs one at a time) | Low | **Accepted.** Bounded by two lines | Up to a few seconds' delay, including on the safety step |
| 21 | A script files junk tickets | Low | **Accepted.** Needs real speech per ticket, inside #1 | Noise in the table during a demo |
| 22 | The checkpoint changes under us: `kn-ft-v1` is a tag, resolved to a commit at each start, so a write token could move it | Low | **Accepted, owner action.** Only config, tokenizer and ONNX paths are fetched, from a private repo. The commit in use is stored with every ticket. How the Laya library loads tokenizer files was not reviewed | Pin the full commit in `LAYA_CHECKPOINT_REVISION` for the final build |
| 23 | Cloudflare ends TLS, so it can see the token and audio | Low | **Accepted.** Part of using a quick tunnel | Change `CONSOLE_TOKEN` after a public demo |
| 24 | An officer's typed line is copied to other consoles unmasked | Low | **Accepted.** Agent lines come from templates or from an officer, never from the caller's words, and are not stored. Masking them would break the console's matching of a sent line to who said it | |

## Checked and found sound

- **Rendering.** All caller, agent and ticket text is rendered by React as text. There is no `dangerouslySetInnerHTML` or `innerHTML`, and the CSP allows only this origin's scripts.
- **SQL.** Every query is parameterised; `limit` and the ticket number are range-checked.
- **SSRF and paths.** No URL or path comes from a request. Gnani URLs and the checkpoint source come from settings only.
- **Officer commands.** Only after a valid hello; one known command at a time; `text` is type-checked and limited to 300 characters; state checks live in `Dialogue`.
- **Routes behind nginx.** Only `/ws/` and `/api/` reach the backend, so `/docs`, `/openapi.json` and `/health` do not face the internet.
- **Secrets in logs.** Keys are `SecretStr` or `repr=False`. No log line writes a transcript or a token itself; the text of an exception from a dependency was not checked.
- **Masking cost.** The masking patterns took under 0.05 s on 100,000 characters of hostile text.
- **CI.** Workflows run on `pull_request` with read-only permissions and pass PR text through the environment, not into scripts.

## Limits to know about

The credit ceiling and every other count live in one backend process's memory. They reset on restart, which is also how an owner clears them. They do not protect against a second process.

Reaching the ceiling refuses real callers as well as an attacker. That is deliberate: for a demo on a paid key, a refused call is cheaper than an empty account. Keep the tunnel up only while it is in use.

## Out of scope

- Volumetric DDoS, which is left to Cloudflare.
- A compromised host, container or dependency. Secrets are plain environment variables in the backend container, and SQLite is unencrypted on its volume.
- A compromised Gnani, Cloudflare or Hugging Face account.
- Real personal data. The demo must stay synthetic: real citizen data would need real identity for officers, an audit trail, encryption at rest and a legal basis this demo does not have.
- Phone-network threats. There is no telephony; the phone page is a browser.
