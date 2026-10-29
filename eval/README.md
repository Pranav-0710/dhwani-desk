# Evaluation

Scores the deciders against the success metrics in `docs/PRD.md` (#19) and builds the held-out
test set (#13). Everything runs from `backend/`, with its environment.

| Script | What it does |
| --- | --- |
| `run.py` | Runs each decider through `screen` then `decide`, as the call does, and writes `reports/<name>.md` and `.json` |
| `build_test_set.py` | Applies the reader's corrections from the review sheet, checks the labels, writes `out/test.jsonl` and `data/test/manifest.json`; `--push` uploads the set |
| `record.py` | Records the test sentences one at a time into `out/audio/<id>.wav` (16 kHz) |
| `transcribe.py` | Streams each recording to Prisma in real time, clean at 16 kHz and at 8 kHz with phone-line noise; writes `out/transcripts.jsonl` |
| `line_audio.py` | WAV reading and writing, 16 → 8 kHz downsampling, pink noise at the phone page's levels |

`out/` holds the private material (test sentences, recordings, transcripts) and is git-ignored.
The test set itself lives only in the private dataset repo; git keeps the manifest (ids, a hash of
each sentence, label counts), so a report can be checked against the set it was run on.

## Seed set: compare deciders

```bash
cd backend
uv sync --group laya
uv run python ../eval/run.py --set seed \
    --laya zero-shot=path/to/laya-multilingual-onnx \
    --laya fine-tuned=Dhwanidesk/dhwani-desk-laya@kn-ft-v1
```

`--laya NAME=SOURCE` takes a Hub `repo@revision` or a local checkpoint directory (one with
`onnx/model.onnx`), and can be repeated. The keyword stand-in runs too unless `--no-keyword`. The
base model (`convaiinnovations/laya-multilingual@e4e9ddf`) ships no ONNX graph, so for zero-shot
export it first with the upstream exporter, as `training/finetune_laya.py` (#14) does.

The seed set was shown to the data generator as examples, so it is not held out: it ranks
deciders, it does not measure accuracy. `reports/seed-kn-ft-v1.md` is the first comparison, run on
local copies of the base model's ONNX export and of `kn-ft-v1`; a local checkpoint is named in the
report by a hash of its ONNX weights.

A department counts only when its confidence is high enough for the call to route it, and the
`safety_hazard` row sits next to the call's actual safety hand-off (a hazard or an emergency).

## Held-out test set

1. A Kannada reader corrects `training/out/review/test-set-draft-v1.xlsx`: the sentence in
   `Your Kannada (if changed)`, labels in `Fix labels` as `hazard=yes; urgency=urgent`, or `DROP`.
2. Build and upload it:
   ```bash
   uv run --with openpyxl --with huggingface_hub python ../eval/build_test_set.py \
       --sheet ../training/out/review/test-set-draft-v1.xlsx \
       --draft ../training/out/review/test-set-draft-v1.json --push
   ```
3. Record it (a quiet room, a laptop mic, the reader's usual pace). A sentence corrected after it
   was recorded comes up again:
   ```bash
   uv run --with sounddevice python ../eval/record.py --test-file ../eval/out/test.jsonl
   ```
4. Transcribe it. This spends Prisma credits, about 150 clips × 2 × their length; a stopped run
   carries on where it left off, and a failed clip is tried again on the next run. `--only 16k`
   or `--only 8k` runs one condition. Recordings made elsewhere must be 16-bit mono WAVs, each
   with an `<id>.json` beside it holding the sentence's `sha256_16`, as `record.py` writes:
   ```bash
   uv run python ../eval/transcribe.py
   ```
5. Score it on written text, the 16 kHz transcripts and the 8 kHz noisy ones:
   ```bash
   uv run python ../eval/run.py --set test --test-file ../eval/out/test.jsonl \
       --transcripts ../eval/out/transcripts.jsonl \
       --laya zero-shot=path/to/laya-multilingual-onnx \
       --laya fine-tuned=Dhwanidesk/dhwani-desk-laya@kn-ft-v1
   ```
   The 16 kHz and 8 kHz scores use only the sentences both have, so the drop compares like
   with like.

## Tests

```bash
uv run pytest ../eval/tests
```
