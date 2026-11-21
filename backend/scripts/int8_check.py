"""Whether a faster Laya keeps every safety hand-off and officer request the live model makes.

The rule (docs/decisions/laya-latency.md, "What was tried"), fixed before any candidate is run: on
350 non-test transcripts (the 50 seed sentences and the first 300 training rows), the candidate
may drop **no** safety hand-off and **no** officer request that the live fp32 model makes. Every
other change in what the call would do is counted and reported, not ruled on.

The candidate is either the live checkpoint quantized to int8 here (all of it, its MLP matrices
only, or a range of encoder layers), or any local checkpoint directory, such as a quantization-
aware fine-tune or a smaller model exported the way training/finetune_laya.py publishes them.
What "the call would do" is the state machine's own first reply (dialogue/machine.py) to each
decision, so a change in a threshold or a rule changes this check too.

Usage, from backend/ after `uv sync --group laya`, with LAYA_CHECKPOINT_REPO and _REVISION set:
    uv run python -m scripts.int8_check --int8 all
    uv run python -m scripts.int8_check --int8 mlp --per-channel
    uv run python -m scripts.int8_check --int8 layers:0-10
    uv run python -m scripts.int8_check --candidate path/to/checkpoint

The training rows come from the private dataset repo at --dataset-revision (HF_TOKEN), or from
--train-file. Prints counts and timings only, never a sentence. Exit code 0: the candidate passes.
"""

import argparse
import importlib
import json
import shutil
import statistics
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from config import REPO_ROOT, get_settings
from decisions.thresholds import DEPARTMENT_CONFIDENCE
from dialogue.decision import Decision
from dialogue.machine import CountingDesk, Dialogue
from dialogue.templates import load_templates

DATASET_REPO = "Dhwanidesk/dhwani-desk-data"
SEED_FILE = REPO_ROOT / "data" / "seed" / "seed.jsonl"
TRAIN_ROWS = 300
# The encoder's weight matrices, in graph order: 22 layers of attention in (Wqkv) and out, then
# MLP in and out. The head's matrices come after them and are never quantized by a scope.
LAYERS, PER_LAYER = 22, 4
KINDS = ("attn_qkv", "attn_out", "mlp_in", "mlp_out")
TEMPLATES = load_templates()


@dataclass(frozen=True)
class Run:
    """One model's answers on every transcript, with the time each step took."""

    decisions: list[Decision]
    screen_s: list[float]
    whole_s: list[float]


class _Revision:
    revision = "int8-check"


def act(decision: Decision) -> tuple[tuple[str, ...], str | None]:
    """What the call does with this decision on a caller's first turn: (lines, reason)."""
    call = Dialogue(TEMPLATES, _Revision(), CountingDesk())  # type: ignore[arg-type]
    call.hear("...")
    reply = call.respond(decision)
    return reply.keys, reply.reason


def routed(decision: Decision) -> str | None:
    """The department the call would remember, or None while it is unsure."""
    confident = decision.department_confidence >= DEPARTMENT_CONFIDENCE
    return decision.department if confident else None


def compare(base: list[Decision], candidate: list[Decision]) -> dict[str, Any]:
    """Counts of what changes when the candidate replaces the live model, and the verdict."""
    if len(base) != len(candidate):
        raise ValueError("both models must answer the same transcripts")
    base_acts = [act(d) for d in base]
    cand_acts = [act(d) for d in candidate]
    pairs = list(zip(base_acts, cand_acts, strict=True))

    def dropped(reason: str) -> int:
        return sum(b[1] == reason and c[1] != reason for b, c in pairs)

    def added(reason: str) -> int:
        return sum(b[1] != reason and c[1] == reason for b, c in pairs)

    safety_dropped, officer_dropped = dropped("safety"), dropped("wants_human")
    return {
        "transcripts": len(base),
        "safety_handoffs_live": sum(b[1] == "safety" for b, _ in pairs),
        "safety_handoffs_dropped": safety_dropped,
        "safety_handoffs_added": added("safety"),
        "officer_requests_live": sum(b[1] == "wants_human" for b, _ in pairs),
        "officer_requests_dropped": officer_dropped,
        "officer_requests_added": added("wants_human"),
        "same_reply": sum(b == c for b, c in pairs),
        "same_routed_department": sum(
            routed(b) == routed(c) for b, c in zip(base, candidate, strict=True)
        ),
        "passes": safety_dropped == 0 and officer_dropped == 0,
    }


def pick_nodes(weight_matmuls: list[str], scope: str) -> list[str] | None:
    """The weight MatMul nodes to quantize, from all of them in graph order: None for all."""
    if scope == "all":
        return None
    encoder = LAYERS * PER_LAYER
    if len(weight_matmuls) < encoder:
        raise SystemExit(f"Expected {encoder}+ weight matrices, found {len(weight_matmuls)}")
    if scope == "mlp":
        layers = weight_matmuls[:encoder]
        return [n for i, n in enumerate(layers) if KINDS[i % PER_LAYER].startswith("mlp")]
    if scope.startswith("layers:"):
        first, _, last = scope.removeprefix("layers:").partition("-")
        low, high = int(first), int(last or first)
        if not 0 <= low <= high < LAYERS:
            raise SystemExit(f"Layers run from 0 to {LAYERS - 1}: {scope!r}")
        return weight_matmuls[low * PER_LAYER : (high + 1) * PER_LAYER]
    raise SystemExit(f"--int8 takes all, mlp or layers:A-B, not {scope!r}")


def quantize(checkpoint: Path, scope: str, per_channel: bool, work: Path) -> Path:
    """A copy of `checkpoint` with its ONNX graph quantized to int8; returns its directory."""
    import onnx
    from onnxruntime.quantization import QuantType, quantize_dynamic

    out = work / f"int8-{scope.replace(':', '-')}{'-pc' if per_channel else ''}"
    if out.exists():
        shutil.rmtree(out)
    (out / "onnx").mkdir(parents=True)
    shutil.copy2(checkpoint / "rl_agent_config.json", out)
    shutil.copytree(checkpoint / "tokenizer", out / "tokenizer")

    model = onnx.load(str(checkpoint / "onnx" / "model.onnx"))
    weights = {init.name for init in model.graph.initializer}
    names = [n.name for n in model.graph.node if n.op_type == "MatMul" and n.input[1] in weights]
    _skip_shape_inference()
    quantize_dynamic(
        model,
        str(out / "onnx" / "model.onnx"),
        op_types_to_quantize=["MatMul"],
        nodes_to_quantize=pick_nodes(names, scope),
        per_channel=per_channel,
        weight_type=QuantType.QInt8,
        extra_options={"DefaultTensorType": onnx.TensorProto.FLOAT},
    )
    return out


def _skip_shape_inference() -> None:
    """onnxruntime's quantizer re-infers shapes first, which fails on this graph; the graph
    already carries what quantization needs."""
    import onnx

    entry = importlib.import_module("onnxruntime.quantization.quantize")
    quantizer = importlib.import_module("onnxruntime.quantization.onnx_quantizer")
    entry.save_and_reload_model_with_shape_infer = lambda model: model
    entry.load_model_with_shape_infer = lambda path: onnx.load(str(path))
    quantizer.save_and_reload_model_with_shape_infer = lambda model: model


def transcripts(train_file: Path | None, revision: str, token: str | None) -> list[str]:
    """The 50 seed sentences, then the first TRAIN_ROWS training rows: never the test set."""
    seed = [json.loads(line)["text"] for line in _lines(SEED_FILE)]
    if train_file is None:
        from huggingface_hub import hf_hub_download

        train_file = Path(
            hf_hub_download(
                DATASET_REPO, "train.jsonl", repo_type="dataset", revision=revision, token=token
            )
        )
    train = [json.loads(line)["text"] for line in _lines(train_file)][:TRAIN_ROWS]
    return seed + train


def _lines(path: Path) -> list[str]:
    return [line for line in path.read_text("utf-8").splitlines() if line.strip()]


def run(decider: Any, texts: list[str]) -> Run:
    for text in texts[:2]:  # warm-up
        decider.decide(text, decider.screen(text))
    decisions, screens, wholes = [], [], []
    for text in texts:
        started = time.perf_counter()
        screening = decider.screen(text)
        screened = time.perf_counter()
        decisions.append(decider.decide(text, screening))
        screens.append(screened - started)
        wholes.append(time.perf_counter() - started)
    return Run(decisions, screens, wholes)


def checkpoint_dir(settings: Any, token: str | None) -> Path:
    """The live checkpoint's local directory (downloaded once, then reused)."""
    from decisions.laya import _download

    source = settings.laya_checkpoint_repo
    if Path(source).is_dir():
        return Path(source)
    return _download(source, settings.laya_checkpoint_revision, token)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    which = parser.add_mutually_exclusive_group(required=True)
    which.add_argument(
        "--int8", metavar="SCOPE", help="quantize the live model: all, mlp or layers:A-B (0-21)"
    )
    which.add_argument("--candidate", type=Path, help="a local checkpoint directory to check")
    parser.add_argument("--per-channel", action="store_true", help="per-channel int8 weights")
    parser.add_argument("--train-file", type=Path, help="a local train.jsonl instead of the Hub")
    parser.add_argument("--dataset-revision", default="v1.0", help="dataset repo tag or commit")
    parser.add_argument("--work", type=Path, help="where to write the quantized copy")
    parser.add_argument("--report", type=Path, help="also write the counts as JSON here")
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    from decisions.laya import LayaDecider

    settings = get_settings().model_copy(update={"decider": "laya"})
    token = settings.hf_token.get_secret_value() or None
    texts = transcripts(args.train_file, args.dataset_revision, token)
    live = LayaDecider.load(settings)
    print(f"live: {live.revision}; {len(texts)} transcripts (seed + training, no test rows)")

    if args.candidate:
        candidate_dir, label = args.candidate, f"candidate {args.candidate.name}"
    else:
        work = args.work or Path(tempfile.mkdtemp(prefix="laya-check-"))
        started = time.perf_counter()
        candidate_dir = quantize(checkpoint_dir(settings, token), args.int8, args.per_channel, work)
        label = f"int8 {args.int8}{' per-channel' if args.per_channel else ''}"
        print(f"{label}: quantized in {time.perf_counter() - started:.0f} s")
    candidate = LayaDecider.load(
        settings.model_copy(update={"laya_checkpoint_repo": str(candidate_dir)})
    )

    base_run, cand_run = run(live, texts), run(candidate, texts)
    counts = compare(base_run.decisions, cand_run.decisions)
    counts |= {
        "candidate": label,
        "live": live.revision,
        "safety_step_median_s": [
            round(statistics.median(r.screen_s), 3) for r in (base_run, cand_run)
        ],
        "whole_decision_median_s": [
            round(statistics.median(r.whole_s), 3) for r in (base_run, cand_run)
        ],
    }
    for key, value in counts.items():
        print(f"{key}: {value}")
    print("PASS" if counts["passes"] else "FAIL: it drops what the live model hands off")
    if args.report:
        args.report.write_text(json.dumps(counts, indent=1), "utf-8")
    sys.exit(0 if counts["passes"] else 1)


if __name__ == "__main__":
    main()
