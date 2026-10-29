# Training

Laya benchmarking, synthetic data generation and fine-tuning. Fine-tuning and data generation run on Kaggle or Google Colab GPUs; the benchmark is inference-only and also runs on a laptop CPU.

Notebooks are committed with outputs stripped (`pipx run nbstripout training/*.ipynb`); CI rejects notebooks with outputs.

## Setup (laptop)

```bash
cd training
uv sync
uv run pytest            # generator checks; no GPU needed
```

## Week-1 Laya Kannada gate (#6)

```bash
uv run python laya_benchmark.py                    # all five variants, ~40 min on CPU
uv run python laya_benchmark.py --variants A D --limit 10
```

On Kaggle / Colab: open `01_laya_benchmark.ipynb` and run all cells (uses the GPU if there is one).

Writes `results/benchmark.json` (metrics, latency and every prediction) and prints a markdown table. The decision and its reasoning: `docs/decisions/laya-kannada-gate.md`.

## Synthetic training data (#7)

```bash
uv run python generate_data.py --dry-run --n 5    # print the prompts; no model, laptop is fine
```

The real run is on Kaggle / Colab: open `02_generate_data.ipynb`, add an `HF_TOKEN` secret, run the pilot, pick the model, then run the full set. It uploads to the private dataset repo `Dhwanidesk/dhwani-desk-data` (see `STRUCTURE.md`).

How it works: every row's labels are sampled first from the templates in `data/generation/scenarios.json` (department, scenario, urgency and hazard; ward and how it is mentioned; mood; flags). The language model only writes the caller's sentence for those labels. A sentence is kept only if it passes checks that keep it consistent with its labels:

- Kannada script only.
- The required ward or place phrase is there, and no other ward's name, landmark or number is (matched across spacing and word endings).
- The ID number is there, or there is no long number at all.

So the labels are right by construction. What the checks cannot catch is a sentence that is grammatical but wrong in meaning. That is what the 10% hand check in #12 is for.

Output rows use the seed format (`id`, `text`, `gloss`, `labels`) plus `meta` (scenario, location form, model, attempt).

## Fine-tuning Laya (#14)

```bash
uv run python finetune_laya.py --train-file train.jsonl --dry-run   # check the data; laptop is fine
```

The real run is on Kaggle: open `03_finetune_laya.ipynb`, set the accelerator to GPU T4 x2, add an `HF_TOKEN` secret (write) and use **Save Version → Save & Run All**, so the run survives a closed browser tab. Any failing step stops the notebook. Step 1 is a 64-row smoke run through training, export and the ONNX check; step 2 trains on the whole set for 3 epochs (about 1,060 optimizer updates at a batch of 64) and publishes.

What the script does:

- **Data:** `train.jsonl` from `Dhwanidesk/dhwani-desk-data` at the tag `--dataset-tag` (default `v1.0`; `--tag-dataset` makes it at the current head if missing).
- **Questions:** each sentence is asked the eight questions the backend asks Laya, built from `data/decision_schema.json` exactly as `backend/decisions/schema.py` builds them. `ward` and `contains_personal_id` are decided in backend code, so they are not trained. Urgency is trained as its level index.
- **Training:** `laya.train.finetune` from `convaiinnovations/laya-multilingual` at Laya 0.4.6's reviewed commit. Choice options are shuffled every epoch, so department is not learnt by position. Laya keeps 400 items aside to fit its calibration temperatures.
- **Export:** the upstream exporter at the commit that exported the base (`docs/decisions/laya-latency.md`), with `--verify` against PyTorch.
- **Load check:** the exported graph is loaded with `ONNXAgent`, as the backend loads it, and scored on the 50-sentence seed set (`load_check.json`). A full run is published only if it gets at least half the departments right and does not give every sentence the same one; zero-shot gets 0.08 on this check.
- **Publish:** `--push-tag kn-ft-v1` uploads to `Dhwanidesk/dhwani-desk-laya` in the layout the backend loads (`rl_agent_config.json`, `tokenizer/` and `encoder/` at the root, the graph in `onnx/`, plus `model.safetensors` and the run's reports) and tags the commit. A tag that exists is never overwritten, and a `--limit` run is never published.

Before training, `--push-tag` checks that the token reaches the model repo and that the tag is free, so a bad token fails in seconds, not after the run. Calibration holds out about 1,600 items (Laya's default is 400).

Known gap: the backend asks Laya about everything the caller has said so far, and training uses single sentences. A two-turn call (problem, then the place) is outside what the model has seen; the eval harness measures how much that costs.

Then set `DECIDER=laya` and `LAYA_CHECKPOINT_REVISION=<tag>` in `.env`. Zero-shot against fine-tuned on the held-out test set is measured by the eval harness (`eval/`, #19), not here: Laya's own before/after uses its calibration slice, which comes from the training data.
