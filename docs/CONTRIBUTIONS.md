# Contributions

The challenge asks each team member's contribution to be documented, and teammates are interviewed separately. Update this file in the same PR as the work it describes.

Lakshmi Narasimman was away until 2026-10-16. In that time Jey Pranav also did the B-owned issues #5–#8 (seed set, Laya benchmark, data generator, Hugging Face repos), so they are logged under Jey Pranav's name below.

From 2026-10-28 Lakshmi Narasimman is away again, and Jey Pranav is doing the remaining B-owned issues (#13, #14, #18, #19, #20, #23, #24); they are logged under Jey Pranav's name too.

| Area | Jey Pranav | Lakshmi Narasimman |
| --- | --- | --- |
| Gnani integration | Prisma realtime STT client and proxy; Timbre TTS client and voice choice; browser call path | |
| Dialogue and escalation | Dialogue state machine, Kannada reply templates, escalation order; officer take-over and hand-back | |
| Laya (benchmark, fine-tuning, serving) | Week-1 Kannada benchmark and go/no-go decision; ONNX serving in the call, safety-first decisions, latency report; fine-tuning run and checkpoint publishing | |
| Data and evaluation | Decision schema, fictional wards, seed set; synthetic data generator with consistency and meaning checks | |
| Console and phone UI | Phone page: browser mic capture and agent playback | |
| Delivery (Docker, demo, posts, write-up) | Repo bootstrap, CI, PRD; Hugging Face organization and repos; call sockets locked down for the tunnel | |

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
| 2026-10-28 | Jey Pranav | #46 | Laya fine-tuning notebook: Kaggle run, ONNX export, load check, tagged checkpoint (#14) |
