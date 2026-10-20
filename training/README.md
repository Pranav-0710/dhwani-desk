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

The real run is on Kaggle / Colab: open `02_generate_data.ipynb`, add an `HF_TOKEN` secret and press Run all.

A full run takes several hours. Every 10 batches, `--resume` / `--checkpoint-every` save the progress to `checkpoint/` in the dataset repo. `--max-hours 11` stops cleanly before Kaggle's 12-hour limit. A stopped run continues when you press Run all again with the same settings. A checkpoint from different settings is refused. It uploads to the private dataset repo `Dhwanidesk/dhwani-desk-data` (see `STRUCTURE.md`).

How it works: every row's labels are sampled first from the templates in `data/generation/scenarios.json` (department, scenario, urgency and hazard; ward and how it is mentioned; mood; flags). The language model only writes the caller's sentence for those labels. A sentence is kept only if it passes checks that keep it consistent with its labels:

- Kannada script only.
- The required ward or place phrase is there, and no other ward's name, landmark or number is (matched across spacing and word endings).
- The ID number is there, or there is no long number at all.

- Meaning check: the model is shown its own sentence without the request and must pick the scenario it was written for, out of every scenario in that department, a few from others, and "none of these". This catches sentences that are well-formed but say something else; in the first Qwen3-14B pilot about 15% did. Turn it off with `--no-meaning-check`.

So the labels are right by construction, as far as the model can read its own Kannada. What the checks still cannot catch is a sentence that the model misreads the same way it miswrote it. That is what the 10% hand check in #12 is for.

Output rows use the seed format (`id`, `text`, `gloss`, `labels`) plus `meta` (scenario, location form, model, attempt).

The `gloss` is the model's own English for what it meant to say, not a translation of what it wrote, so it cannot be used to check a sentence. Check the Kannada against `meta.scenario` instead.
