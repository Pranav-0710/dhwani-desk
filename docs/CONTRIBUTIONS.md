# Contributions

The challenge asks each team member's contribution to be documented, and teammates are interviewed separately. Update this file in the same PR as the work it describes.

Lakshmi Narasimman was away until 2026-10-16. In that time Jey Pranav also did the B-owned issues #5–#8 (seed set, Laya benchmark, data generator, Hugging Face repos), so they are logged under Jey Pranav's name below.

From 2026-10-28 Lakshmi Narasimman is away again, and Jey Pranav is doing the remaining B-owned issues (#13, #14, #18, #19, #20, #23, #24); they are logged under Jey Pranav's name too.

| Area | Jey Pranav | Lakshmi Narasimman |
| --- | --- | --- |
| Gnani integration | Prisma realtime STT client and proxy; Timbre TTS client and voice choice; browser call path | |
| Dialogue and escalation | Dialogue state machine, Kannada reply templates, escalation order; officer take-over and hand-back; barge-in | |
| Laya (benchmark, fine-tuning, serving) | Week-1 Kannada benchmark and go/no-go decision; ONNX serving in the call, safety-first decisions, latency report; fine-tuning run and checkpoint publishing | |
| Data and evaluation | Decision schema, fictional wards, seed set; synthetic data generator with consistency and meaning checks; eval harness and seed-set comparison | |
| Console and phone UI | Phone page: browser mic capture and agent playback; line noise toggle; supervisor console | |
| Delivery (Docker, demo, posts, write-up) | Repo bootstrap, CI, PRD; Hugging Face organization and repos; call sockets locked down for the tunnel; Docker Compose and tunnel runbook | |

## Log

| Date | Who | PR | What |
| --- | --- | --- | --- |
| 2026-10-09 | Jey Pranav | — | Repo bootstrap: workflow, CI, PRD |
| 2026-10-13 | Jey Pranav | #27 | Backend scaffold: FastAPI, uv, ruff, pytest (#1) |
| 2026-10-13 | Jey Pranav | #28 | Prisma realtime STT client and `/ws/stt` proxy (#2) |
| 2026-10-13 | Jey Pranav | #31 | Timbre TTS client; Kannada voice shortlist, Saanvi chosen (#3) |
| 2026-10-13 | Jey Pranav | #32 | Echo call: browser mic → Prisma → Timbre → speaker, with the phone page (#4) |
| 2026-10-13 | Jey Pranav | #33 | Decision schema, fictional wards of Hosapura, 50-sentence Kannada seed set (#5) |
| 2026-10-13 | Jey Pranav | #34 | Laya Kannada benchmark and week-1 gate decision: go, fine-tuned only (#6) |
| 2026-10-13 | Jey Pranav | #35 | Synthetic Kannada data generator and Kaggle notebook (#7) |
| 2026-10-13 | Jey Pranav | #36 | Hugging Face organization, private repos and token setup (#8) |
| 2026-10-16 | Jey Pranav | #37 | Dialogue state machine with Kannada templates, escalation and hang-up (#9) |
| 2026-10-16 | Jey Pranav | #39 | Two-person workflow: CODEOWNERS, required review, contributions backfill (#38) |
| 2026-10-19 | Jey Pranav | #40 | Tickets: SQLite store, masking across turns, transcript retention, console API (#17) |
| 2026-10-23 | Jey Pranav | #42 | Laya serving: ONNX decider, safety questions first, "one moment" line, latency report (#15) |
| 2026-10-23 | Jey Pranav | #43 | Officer take-over: console socket, typed replies spoken by Timbre, hand-back, thresholds API (#16) |
| 2026-10-24 | Jey Pranav | #45 | Call sockets locked down: allowed pages only, two calls at a time, silent sockets hung up (#29) |
| 2026-10-24 | Jey Pranav | #44 | Line noise toggle on the phone page: pink noise at two levels, kept on while the caller is muted (#11) |
| 2026-10-28 | Jey Pranav | #46 | Laya fine-tuning notebook: Kaggle run, ONNX export, load check, tagged checkpoint (#14) |
| 2026-10-29 | Jey Pranav | #47 | Supervisor console: live calls, decision bars, take-over, tickets table (#18) |
| 2026-10-29 | Jey Pranav | #48 | Docker Compose, nginx in front, tunnel runbook; console API needs the token (#21) |
| 2026-10-29 | Jey Pranav | #49 | Eval harness: PRD metrics table, phone-line audio, test-set builder and recorder; seed-set comparison (#19) |
| 2026-10-31 | Jey Pranav | #51 | Fine-tuned Laya on by default; Hub checkpoints load as real files ONNX Runtime accepts (#50) |
| 2026-10-31 | Jey Pranav | #53 | Held-out test set: 150 sentences, Timbre-voiced audio, Prisma transcripts, first fine-tuned report (#13) |
| 2026-10-31 | Jey Pranav | #52 | Barge-in: the caller can talk over the agent; playback stops and the line is cut (#10) |
| 2026-11-01 | Jey Pranav | #54 | Teaser and demo posts: [X](https://x.com/jey_pranav53258/status/2116867959910191274), [LinkedIn](https://lnkd.in/p/gNMNPkh4) (#20) |
| 2026-11-01 | Jey Pranav | #55 | Demo video posted on [LinkedIn](https://lnkd.in/p/gNMNPkh4) and [X](https://x.com/jey_pranav53258/status/2116888213851963672); shot list and synthetic caller audio; call 1 moved to a complaint the model does not flag as a hazard (#23) |
| 2026-11-01 | Jey Pranav | #57 | Docker with Laya: CPU-only PyTorch (no CUDA download), container data paths kept whatever .env says (#56) |
| 2026-11-01 | Jey Pranav | #58 | Latency measured with the fine-tuned model; the agent says "okay" as soon as a turn ends, first audio about 1.5 s (#22) |
| 2026-11-01 | Jey Pranav | #60 | A call opens a new Prisma session when one drops mid-call, up to 3 times, without greeting again (#59) |
| 2026-11-01 | Jey Pranav | #62 | Reconnect follow-up: one microphone per call, retry only recoverable Prisma failures, not after hang-up (#61) |
| 2026-11-01 | Jey Pranav | #64 | Demo hardening: SQLite side files ignored, port 8080 on localhost only, console token at least 20 characters, rate limits on the token routes (#63) |
| 2026-11-01 | Jey Pranav | #66 | A failed turn hands the call to an officer instead of going silent; barge-in no longer cuts the ticket number or safety advice (#65) |
| 2026-11-01 | Jey Pranav | #69 | Review fixes: one call per caller and a 10-minute cap, ticket filing off the event loop, corrected wards kept, generic error text, digit-word numbers masked (#68) |
| 2026-11-01 | Jey Pranav | #71 | Low-priority review fixes: strict CSP, barge-in keeps the first syllables and survives a race, eval survives a Prisma error, CI actions and base images pinned (#70) |
| 2026-11-02 | Jey Pranav | #73 | Clearer phone page and console: safety calls first, one-screen phone with the newest line in view, Kannada-first status, keyboard take-over, contrast, readable at 100% and 50% zoom (#72) |
