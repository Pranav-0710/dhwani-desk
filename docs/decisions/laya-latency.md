# Decision: how Laya fits in the live call

Status: **serve Laya as fp32 ONNX on 8 CPU threads, safety questions first, with "one moment" spoken while the rest runs.** The PRD's old target of 300 ms for every question in one batch cannot be met on the laptop's CPU by a 322M-parameter model. The targets are now ≤ 0.6 s p95 for the safety questions and ≤ 2.5 s p95 for the whole decision on a caller's first turn; it measures 0.54 s and 1.70 s. Laya rereads the whole call every turn, so later turns are slower. Measured 2026-10-20 to 2026-10-23, base multilingual checkpoint, Laya 0.4.6, laptop CPU (Intel Core Ultra 7 155H, 16 GB, on mains power). **Measure again on the fine-tuned checkpoint (#14)** with the script below; it is the same size, so the timings should hold, but int8 may become usable.

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
4. Otherwise, if the rest takes longer than 0.3 s, the agent says "ಸರಿ, ಒಂದು ಕ್ಷಣ" ("okay, one moment"), also from the cache. On a first turn that comes ~1.35 s after the caller stopped (0.6 + 0.45 + 0.3), inside the PRD's "first agent audio ≤ 1.5 s", and the reply ~2.0 s after. On a call three sentences long, "one moment" comes ~1.9 s and the reply ~3.4 s after the caller stopped: that misses the target, and is the case to recheck on the fine-tuned checkpoint and with int8.

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

- **Threads.** The default thread count fights over the laptop's mix of performance and efficiency cores. `LAYA_THREADS=8` is set on the decider's own ONNX Runtime session.
- **int8 is not used.** It halves the time, but the base checkpoint is unsure about Kannada: most of its yes/no answers sit near 0.5, and int8's rounding flips them (largest change 0.83). Per-channel and weights-only variants did not help. A fine-tuned checkpoint should answer far more confidently, so test int8 again on it, against the test set (#13). Switch only if the accuracy holds. Quantizing needs two workarounds in onnxruntime's quantizer: skip its shape re-inference (`onnx_quantizer.save_and_reload_model_with_shape_infer`), and pass `extra_options={"DefaultTensorType": onnx.TensorProto.FLOAT}`.
- **Shorter inputs do not help.** Padding is not the cost, and capping Laya's `max_len` at 192 cut the question text and broke answers.

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
