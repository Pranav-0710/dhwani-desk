# Contributions

The challenge asks each team member's contribution to be documented, and teammates are interviewed separately. Update this file in the same PR as the work it describes.

Lakshmi Narasimman was away until 2026-10-16. In that time Jey Pranav also did the B-owned issues #5–#8 (seed set, Laya benchmark, data generator, Hugging Face repos), so they are logged under Jey Pranav's name below.

From 2026-10-28 Lakshmi Narasimman is away again, and Jey Pranav is doing the remaining B-owned issues (#13, #14, #18, #19, #20, #23, #24); they are logged under Jey Pranav's name too.

| Area | Jey Pranav | Lakshmi Narasimman |
| --- | --- | --- |
| Gnani integration | Prisma realtime STT client and proxy; Timbre TTS client and voice choice; browser call path | |
| Dialogue and escalation | Dialogue state machine, Kannada reply templates, escalation order; officer take-over and hand-back; barge-in | |
| Laya (benchmark, fine-tuning, serving) | Week-1 Kannada benchmark and go/no-go decision; ONNX serving in the call, safety-first decisions, latency report; fine-tuning run and checkpoint publishing | |
| Data and evaluation | Decision schema, fictional wards, seed set; synthetic data generator with consistency and meaning checks; eval harness and seed-set comparison | Generator changes for the second fine-tune (#67): fewer hazard scenarios, more officer requests |
| Console and phone UI | Phone page: browser mic capture and agent playback; line noise toggle; supervisor console | |
| Delivery (Docker, demo, posts, write-up) | Repo bootstrap, CI, PRD; Hugging Face organization and repos; call sockets locked down for the tunnel; Docker Compose and tunnel runbook; demo video, posts and write-up | |
| Review and testing | | Reviewed and approved #39; tested the app |

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
| 2026-11-02 | Jey Pranav | #75 | Read-back lines no longer repeat the "okay" said just before them (#74) |
| 2026-11-02 | Jey Pranav | #77 | Threat model and fixes: credit ceiling, /ws/stt off, console sockets capped and origin-checked, masked numbers on consoles, ticket locations blanked after 30 days, flush backlog fixed (#76) |
| 2026-11-04 | Lakshmi Narasimman | #78 | Second fine-tune (#67): non-hazard drainage and road scenarios (hazards now 4 of 12 and 3 of 10), officer requests raised to 20% of rows and repeat complaints to 25%, varied officer-request wording in the generation prompt; notebooks 02 and 03 set up for the `v2.0` data tag and the `kn-ft-v2` checkpoint |
| 2026-11-05 | Jey Pranav | #81 | Rule for replacing the live checkpoint, committed before `kn-ft-v2` was scored: any fall in safety recall or a mere tie keeps `kn-ft-v1` (#79) |
| 2026-11-05 | Jey Pranav | #82 | Flaky console test fixed: a cancel arriving during teardown no longer escapes the test client, with one helper for every cancel-then-wait (#80) |
| 2026-11-05 | Jey Pranav | #84 | Second fine-tune finished (#67): hand check and filter for the `v2.0` data, `kn-ft-v2` trained and scored; it missed more hazards, so the switch rule kept `kn-ft-v1` (#83) |
| 2026-11-05 | Jey Pranav | #85 | Write-up with results, models used and the rejected second fine-tune; README results and run steps; team list and contributions brought up to date (#24) |
| 2026-11-06 | Jey Pranav | #87 | Safety phrase net around the decider, committed before scoring, then scored once and kept out of live calls by the switch rule; newest-turn screening measured and rejected (#86) |
| 2026-11-06 | Jey Pranav | #89 | Harsher phone lines in the eval (codec, lost packets, noise as loud as the voice) and Prisma v3.0 as an option, with the switch rule fixed first (#88) |
| 2026-11-06 | Jey Pranav | #91 | Fixes from a full code review: the safety hand-off survives a failed turn write, ward numbers in Kannada digits or Latin script, masking of numbers said with double or in English words, Prisma handshake errors retried, barge-in audio keeps the line noise, a console token is kept unless the server refused it, and a fake seed phone number (#90) |
| 2026-11-07 | Jey Pranav | #93 | Review leftovers: a resumed Prisma session drops the audio that piled up during the gap instead of overflowing, a hang-up mid-line is no longer logged as a TTS failure, and after a hand-back the ticket keeps every turn told to the officer as the location (#92) |
| 2026-11-07 | Jey Pranav | #96 | README rewritten for judges and new readers: key numbers, call and architecture diagrams, how decisions are made, results charts, the switch rule and the models it rejected, safety, privacy and audit design, run steps and limits (#95) |
| 2026-11-07 | Jey Pranav | #98 | int8 retested on the fine-tuned checkpoint and rejected: about 6× faster, but it drops 23 of the live model's safety hand-offs on 350 non-test sentences; latency note, README and write-up updated (#97) |
| 2026-11-09 | Jey Pranav | #101 | Third fine-tune set up: `--add-rows` adds one flag's rows from another dataset version, and the notebook trains `kn-ft-v3` on v1's data plus v2's 356 officer requests, with the recipe and switch rule fixed before training (#99) |
| 2026-11-09 | Jey Pranav | #102 | Real mobile call in the eval: `phone_call.py` builds a marked playlist of the spoken test clips and cuts the far end's call recording back into clips; `transcribe.py` gains a phone condition and `run.py` a `--conditions` option (#100) |
| 2026-11-09 | Jey Pranav | #104 | `kn-ft-v3` scored once on the held-out test set and kept out: more officer requests and fewer false hazards, but 27 of 34 hazards caught against 30; reports, README, write-up and training notes updated (#103) |
| 2026-11-09 | Jey Pranav | #106 | int8 on part of the fine-tuned encoder tested and rejected: MLP-only, layers 0–10 and layers 0–15 each drop safety hand-offs or officer requests on 350 non-test sentences; latency note updated (#105) |
| 2026-11-09 | Jey Pranav | #108 | README brought up to date: 123 training and 17 evaluation tests, partial int8 in the latency limitation, and the built phone-call kit in the phone-line limitation (#107) |
| 2026-11-09 | Jey Pranav | #110 | One-page solution pitch for the submission form: problem, solution, what sets it apart, held-out results and the switch rule; linked at the top of the README (#109) |
| 2026-11-10 | Jey Pranav | #113 | A safety or emergency hand-off files a ticket marked emergency at once, kept even if the caller hangs up, and the full decision fills it in; one ticket per call; every decision the agent acts on is stored with its probabilities and revision; latency note and `call_smoke` timer corrected (#112) |
