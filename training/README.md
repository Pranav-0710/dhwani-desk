# Training

Laya benchmarking, synthetic data generation and fine-tuning. Fine-tuning and data generation run on Kaggle or Google Colab GPUs; the benchmark is inference-only and also runs on a laptop CPU.

Notebooks are committed with outputs stripped (`pipx run nbstripout training/*.ipynb`); CI rejects notebooks with outputs.

## Setup (laptop)

```bash
cd training
uv sync
```

## Week-1 Laya Kannada gate (#6)

```bash
uv run python laya_benchmark.py                    # all four variants, ~30 min on CPU
uv run python laya_benchmark.py --variants A D --limit 10
```

On Kaggle / Colab: open `01_laya_benchmark.ipynb` and run all cells (uses the GPU if there is one).

Writes `results/benchmark.json` (metrics, latency and every prediction) and prints a markdown table. The decision and its reasoning: `docs/decisions/laya-kannada-gate.md`.
