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

The real run is on Kaggle / Colab: open `02_generate_data.ipynb`, add an `HF_TOKEN` secret and press Run all: it runs a 40-sentence pilot and prints samples to check; the full set starts only after `RUN_FULL = True`. It uploads to the private dataset repo `Dhwanidesk/dhwani-desk-data` (see `STRUCTURE.md`).

How it works: every row's labels are sampled first from the templates in `data/generation/scenarios.json` (department, scenario, urgency and hazard; ward and how it is mentioned; mood; flags). The language model only writes the caller's sentence for those labels. A sentence is kept only if it passes checks that keep it consistent with its labels:

- Kannada script only.
- The required ward or place phrase is there, and no other ward's name, landmark or number is (matched across spacing and word endings).
- The ID number is there, or there is no long number at all.

So the labels are right by construction. What the checks cannot catch is a sentence that is grammatical but wrong in meaning. That is what the 10% hand check in #12 is for.

Output rows use the seed format (`id`, `text`, `gloss`, `labels`) plus `meta` (scenario, location form, model, attempt).

The `gloss` is the model's own English for what it meant to say, not a translation of what it wrote, so it cannot be used to check a sentence. Check the Kannada against `meta.scenario` instead.
