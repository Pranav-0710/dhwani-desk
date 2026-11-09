# Decision: how Laya fits in the live call

Status: **serve Laya as fp32 ONNX on 8 CPU threads, safety questions first, with "one moment" spoken while the rest runs.** The PRD's old target of 300 ms for every question in one batch cannot be met on the laptop's CPU by a 322M-parameter model. The targets are now ≤ 0.6 s p95 for the safety questions and ≤ 2.5 s p95 for the whole decision on a caller's first turn; it measures 0.54 s and 1.70 s. Laya rereads the whole call every turn, so later turns are slower. Measured 2026-10-20 to 2026-10-23, base multilingual checkpoint, Laya 0.4.6, laptop CPU (Intel Core Ultra 7 155H, 16 GB, on mains power). **Measure again on the fine-tuned checkpoint (#14)** with the script below; it is the same size, so the timings should hold. int8 was retested on it on 2026-11-07 and is still not usable ("What was tried").

## What a caller turn costs

`uv run python -m scripts.laya_latency` in `backend/` times both steps on the 50 seed sentences, 3 passes each. `--turns 3` joins three sentences into one transcript, the length of a call a few turns in, since Laya rereads everything the caller has said on every turn. Run it on mains power: on battery the laptop throttles its CPU and the timings mean nothing (a run on battery was 3 to 6 times slower and erratic).

Measured 2026-10-23 on mains power:

| Transcript | Safety step, 4 questions | The rest, 4 questions | Whole decision |
| --- | --- | --- | --- |
| One sentence | median 0.45 s, p95 0.54 s | median 0.99 s, p95 1.14 s | median 1.44 s, p95 1.70 s |
| Two sentences | median 0.66 s, p95 0.81 s | median 1.23 s, p95 1.48 s | median 1.89 s, p95 2.30 s |
| Three sentences | median 0.98–1.04 s, p95 1.53–2.20 s | median 1.67–1.80 s, p95 2.42–3.63 s | median 2.68–2.86 s, p95 3.81–5.70 s |

A seed sentence is a whole complaint. A real call's later turns are usually short (a landmark, a ward name), so most turns sit between the first two rows; three full sentences is a long call. That row is two runs. Their medians agree, but the slowest turns did not: long transcripts are sensitive to whatever else the laptop is running (a browser was busy both times), so read its p95 as a range.

The first split put urgency in the second step (3 safety questions, then 5): 0.37 s / 1.74 s median on one sentence. Urgency moved into the safety step so that an emergency outranks a request for an officer in either step (below). That costs the safety step ~0.1 s, and the whole decision is a little faster, since the two steps are now even.

The ward and personal numbers cost nothing: code answers them (`decisions/extractors.py`, per the [week-1 gate](laya-kannada-gate.md)), which also takes two questions out of every Laya run.

Loading takes ~15 s (about a minute when the 1.3 GB weights are not yet in the disk cache). The backend loads Laya once, before it accepts the first call.

## What the caller hears

1. The caller stops talking. After a 0.6 s pause the turn is complete (`dialogue/turns.py`).
2. Laya answers the four questions that can end the call at once: danger, how urgent, a request for an officer, not a civic matter.
3. If danger or an emergency comes back, or else a request for an officer, the hand-off line plays straight away from the audio cache. The rest of the decision then runs anyway, so the officer who takes over sees the department and ward on the console. The turn is stored before the line plays, since a caller told to call 112 may hang up at once.
4. Otherwise, if the rest takes longer than 0.3 s, the agent says "ಸರಿ, ಒಂದು ಕ್ಷಣ" ("okay, one moment"), also from the cache. On a first turn that comes ~1.35 s after the caller stopped (0.6 + 0.45 + 0.3), inside the PRD's "first agent audio ≤ 1.5 s", and the reply ~2.0 s after. On a call three sentences long, "one moment" comes ~1.9 s and the reply ~3.4 s after the caller stopped: that misses the target, and is the case to recheck on the fine-tuned checkpoint. int8 would have helped here, but it changes the answers (below).

Splitting the decision never changes what the caller hears, only when: `Dialogue.screened` applies the same rules, in the same order, as `Dialogue.respond`. That is why urgency is in the safety step: `respond` treats an emergency as a safety risk ahead of a request for an officer, so "my father has chest pain, connect me to an officer" must hear "call 112", not just "connecting you". "Not a civic matter" alone does not end the call at step 2: a distressed caller with a police matter is handed off, not redirected, and only the full decision knows how the caller is.

## Limits

- **One call at a time.** The decider runs one Laya run at a time (each already uses all 8 threads, and the tokenizer is not thread-safe). A second caller's safety step waits for the first caller's run, so the timings above hold for one call at a time, which is the demo.
- **Long calls are cut to their latest words.** For a transcript too long for it, Laya keeps the start and silently drops the end, which is what the next reply most depends on. Kannada runs about 1.5 characters per token: at 1,500 characters (984 tokens) Laya dropped 23 tokens for the safety questions and 149 for the rest. The decider sends at most the latest 700 tokens, counted with Laya's own tokenizer (`MAX_TRANSCRIPT_TOKENS`), and on the same long call Laya then drops nothing.
- **The weights load twice at startup.** `ONNXAgent` builds its own session before the decider replaces it with the 8-thread one; that costs about a second, once.

## What was tried

Timed on one long fixture (all eight questions in one run), before the split:

| Runtime | All 8 questions | Same answers as PyTorch |
| --- | --- | --- |
| PyTorch, as installed (8 threads) | 5.7 s | — |
| ONNX fp32, ONNX Runtime's default threads | 9.3 s | yes (max logit difference ~1e-5) |
| **ONNX fp32, 8 threads** | **4.5 s** | yes |
| ONNX int8 (dynamic quantization) | 2.0–2.7 s | **no: 58 of 80 answers match** |
| ONNX int8 on `kn-ft-v1`, plain or per-channel (#97) | about 6× faster | **no: drops 23 of the fp32 model's safety hand-offs in 350 transcripts** |
| ONNX int8 on part of `kn-ft-v1`'s encoder (#105) | safety step 0.54 → 0.38 s (median) | **no: every variant drops safety hand-offs or officer requests** |

- **Threads.** The default thread count fights over the laptop's mix of performance and efficiency cores. `LAYA_THREADS=8` is set on the decider's own ONNX Runtime session.
- **int8 is not used, on the base checkpoint or the fine-tuned one.** On the base checkpoint it halves the time, but the model is unsure about Kannada: most of its yes/no answers sit near 0.5, and int8's rounding flips them (largest change 0.83). Per-channel and weights-only variants did not help. The fine-tuned checkpoint was expected to answer confidently enough to survive the rounding, so it was retested on 2026-11-07 (#97). The comparison used 350 transcripts: the 50 seed sentences and 300 training rows, never the held-out test set. Each decision the call acts on was compared with the fp32 model:

  | | Plain int8 | Per-channel int8 |
  | --- | --- | --- |
  | Safety hand-offs fp32 makes that int8 drops | 23 | 23 |
  | Every acted-on decision the same | 153 of 350 | 162 of 350 |
  | Department routed the same | 261 of 350 | 269 of 350 |

  In the same run, plain int8 was about 6× faster: the safety step took 0.20 s against 1.29 s (median), and the whole decision 0.62 s against 4.03 s. Both were timed on battery, so only the ratio holds. Losing 23 safety hand-offs rules it out, whatever the speed, so the live call stays on fp32. A faster safety step now needs a smaller model, or quantization-aware fine-tuning, not post-training int8.

  int8 on only part of the encoder was tested next (#105), on the same 350 transcripts. The rule was fixed before the run: a variant must drop **no** safety hand-off and **no** officer request that fp32 makes. All three failed:

  | int8 on | Safety hand-offs dropped | Officer requests dropped | Every acted-on decision the same |
  | --- | --- | --- | --- |
  | The MLP matrices of all 22 layers | 21 | 7 | 176 of 350 |
  | Layers 0–10, all matrices | 2 | 4 | 275 of 350 |
  | Layers 0–15, all matrices | 21 | 6 | 162 of 350 |

  The MLP-only variant took the safety step from 0.54 s to 0.38 s (median). The other two were timed while the laptop was busy, so only their decisions count. Even half the encoder in int8 changes a fifth of the decisions, so no further split was tried: picking the one that happens to pass on these 350 transcripts would be fitting to them. Quantizing needs two workarounds in onnxruntime's quantizer: skip its shape re-inference (`onnx_quantizer.save_and_reload_model_with_shape_infer`), and pass `extra_options={"DefaultTensorType": onnx.TensorProto.FLOAT}`.
- **Shorter inputs do not help.** Padding is not the cost, and capping Laya's `max_len` at 192 cut the question text and broke answers.
- **Screening only the newest turn does not help either.** Measured 2026-11-05 on `kn-ft-v1` with 50 three-turn calls (a seed sentence, then a landmark, then a ward number). The safety step takes 0.43 s on the whole call and 0.27 s on the newest turn, so it saves about 0.15 s. Of the 31 calls still running after turn 1, the answer changed for urgency in 21 at turn 2 and 15 at turn 3, and for out of scope in 8 and 3. Urgency goes on the ticket, so this is not used. Length is not the main cost on calls like these: the whole decision takes 1.17 s at turn 1 and 1.36 s at turn 3. Slow replies on a call come mostly from a busy laptop.

## Exporting a checkpoint

Laya's pip package has no exporter. The upstream one, `laya-dotnet/tools/export_onnx.py` at commit `0d0c73e317724325b9a12dc9d7d376232b3e3b25` of [NandhaKishorM/laya](https://github.com/NandhaKishorM/laya), exported the base checkpoint. The steps for the fine-tuned one (`--verify` compares the export with PyTorch):

```bash
huggingface-cli download Dhwanidesk/dhwani-desk-laya --revision <tag> --local-dir ckpt
git clone https://github.com/NandhaKishorM/laya laya-src
git -C laya-src checkout 0d0c73e317724325b9a12dc9d7d376232b3e3b25
python laya-src/laya-dotnet/tools/export_onnx.py ckpt --out export --verify
huggingface-cli upload Dhwanidesk/dhwani-desk-laya export/ckpt/model.onnx onnx/model.onnx
huggingface-cli upload Dhwanidesk/dhwani-desk-laya export/ckpt/model.onnx.data onnx/model.onnx.data
```

Then tag that commit, set `LAYA_CHECKPOINT_REVISION` to the tag and `DECIDER=laya`, and run `scripts.laya_latency` again. The backend expects Laya's files at the repo root (`rl_agent_config.json`, `tokenizer/`) and the graph in `onnx/`. Every ticket stores the repo and commit that decided it (`laya:<repo>@<commit>`).

## End of caller speech to first agent audio (#22)

Measured on 2026-11-01 with `scripts.call_smoke` against the fine-tuned checkpoint (`kn-ft-v1`, `DECIDER=laya`, `LAYA_THREADS=8`) on the laptop, CPU about 30% busy with other apps. Four calls, each a garbage complaint and then a place ("ಕಸದ ಗಾಡಿ ಒಂದು ವಾರದಿಂದ ಬಂದಿಲ್ಲ", "ಗಾಂಧಿನಗರ ಸರ್ಕಾರಿ ಆಸ್ಪತ್ರೆ ಹತ್ತಿರ"). The caller is a Timbre voice; the time is taken from the end of the caller's words.

| Turn | What was timed | After the caller stopped |
| --- | --- | --- |
| Second turn (the ward, then the ticket read-back) | the ticket read-back starting | 2.74, 2.82, 3.00, 2.99 s |
| First turn ("one moment", then the question) | "one moment" starting | 2.2 s in one call; 12.2 to 12.7 s in the other three |

On the second turn the script's timer at the time caught only the answer, not the "one moment" said before it (fixed in #112: `call_smoke` now prints both, the first agent audio and the answer's start).

Where the time goes on the second turn: Prisma returns the transcript about 0.75 s after the words end, the turn collector waits `TURN_PAUSE_S` = 0.6 s, Laya's safety step and decision take about 1.1 s, and Timbre starts the ticket line in about 0.3 s.

On the first turn, three calls out of four waited about 12 s for Prisma's transcript of the caller's first line; the fourth got it in about 3 s. In those calls the caller started speaking within a second of the session opening. Later turns never waited like this.

Without an acknowledgement, the answer to a normal turn started about 2.8 s after the caller stopped, against a 1.5 s target. "One moment" is said once a decision has run 0.3 s, so the caller heard that a little earlier: about 2.5 s, from the parts above (0.75 + 0.6 + the safety step's 0.5 + 0.3 + Timbre's 0.3), not measured.

### With an acknowledgement (`acknowledge_turns`)

The agent now says a cached "ಸರಿ." ("okay") as soon as a turn ends, before Laya decides; the slow-decision "ಒಂದು ಕ್ಷಣ." ("one moment") is then not said, so the caller hears one short filler, not two. Four calls, same lines, same machine (busier this time):

| | After the caller stopped |
| --- | --- |
| First agent audio ("ಸರಿ."), on the caller's first turn | 1.51, 1.60, 1.63, 2.03 s |
| The answer itself (question or ticket read-back) | 3.7 to 6.0 s |

The script's old timer caught "ಸರಿ." on the first turn only; on later turns it timed the answer. "ಸರಿ." is said the moment a turn ends, before anything is decided, so a later turn's should come as fast. One call on 2026-11-10 with the fixed timer (same lines, `kn-ft-v1`) bears that out: "ಸರಿ." at 1.70 s and 1.15 s after the caller stopped, the answer at 2.37 s and 2.29 s, on the first and second turn.

So first audio is at the 1.5 s target, give or take 0.1 s, and the caller is not left in silence. The answer itself still waits for Laya, which took 2 to 4 s per decision in this run (about 1.1 s in the earlier one; the laptop was busier). What would bring the answer closer:

- **A faster safety step.** It is most of Laya's time. Post-training int8 is about 6× faster but loses safety hand-offs on both checkpoints ("What was tried"), so this now means quantization-aware fine-tuning or a smaller model.
- **A shorter `TURN_PAUSE_S`.** Saves up to 0.6 s per turn, at the risk of cutting a caller who pauses mid-sentence into two turns. Tune only against real callers.
- **The slow first transcript.** Check with Gnani whether a new Prisma session needs a moment before speech; the backend could open the session before the greeting finishes.
