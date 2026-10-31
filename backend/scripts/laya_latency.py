"""Time Laya's two decision steps on the 50 seed sentences, the way the call runs them.

Loads the decider the backend would with DECIDER=laya, or a local checkpoint directory given
with --checkpoint, then times `screen` (the safety questions) and `decide` (the rest) on every
seed sentence. The numbers go in docs/decisions/laya-latency.md.

Usage, from backend/ after `uv sync --group laya`:
    uv run python -m scripts.laya_latency --checkpoint path/to/exported/checkpoint
    uv run python -m scripts.laya_latency    # LAYA_CHECKPOINT_REPO at LAYA_CHECKPOINT_REVISION
"""

import argparse
import json
import statistics
import time

from config import REPO_ROOT, get_settings
from decisions.laya import LayaDecider
from decisions.schema import OTHERS, SAFETY


def summary(seconds: list[float]) -> str:
    p95 = statistics.quantiles(seconds, n=20)[18]
    return f"median {statistics.median(seconds):.2f} s, p95 {p95:.2f} s"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", help="a local checkpoint directory")
    parser.add_argument("--threads", type=int, help="ONNX Runtime threads (default LAYA_THREADS)")
    parser.add_argument("--rounds", type=int, default=3, help="passes over the seed set")
    parser.add_argument(
        "--turns",
        type=int,
        default=1,
        help="join this many seed sentences into one transcript, like a call several turns in",
    )
    args = parser.parse_args()

    update: dict[str, object] = {"decider": "laya"}
    if args.checkpoint:
        update["laya_checkpoint_repo"] = args.checkpoint
    if args.threads:
        update["laya_threads"] = args.threads
    settings = get_settings().model_copy(update=update)
    seed = (REPO_ROOT / "data" / "seed" / "seed.jsonl").read_text("utf-8").splitlines()
    sentences = [json.loads(line)["text"] for line in seed if line.strip()]
    texts = [" ".join((sentences * args.turns)[i : i + args.turns]) for i in range(len(sentences))]

    started = time.perf_counter()
    decider = LayaDecider.load(settings)
    print(f"{decider.revision}: loaded in {time.perf_counter() - started:.1f} s")
    for text in texts[:2]:  # warm-up
        decider.decide(text, decider.screen(text))

    screens, rests = [], []
    for _ in range(args.rounds):
        for text in texts:
            started = time.perf_counter()
            screening = decider.screen(text)
            screened = time.perf_counter()
            decider.decide(text, screening)
            screens.append(screened - started)
            rests.append(time.perf_counter() - screened)

    totals = [a + b for a, b in zip(screens, rests, strict=True)]
    print(
        f"{len(texts)} transcripts of {args.turns} sentence(s) x {args.rounds} rounds, "
        f"{settings.laya_threads} threads"
    )
    print(f"safety step, {len(SAFETY)} questions: {summary(screens)}")
    print(f"the rest, {len(OTHERS)} questions:   {summary(rests)}")
    print(f"whole decision:            {summary(totals)}")


if __name__ == "__main__":
    main()
