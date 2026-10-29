"""Score the deciders on the test set and print the PRD's success-metrics table (#19).

Every decider runs the call's own two steps (`screen`, then `decide`) on three versions of each
test sentence, when they exist: the text as written, Prisma's transcript of it read aloud at
16 kHz, and Prisma's transcript of the same audio on an 8 kHz noisy line (`transcribe.py`).
A yes/no answer counts as "yes" past the threshold the call acts on (decisions/thresholds.py).

Usage, from the repo root with the backend's environment (`uv sync --group laya` for Laya):
    cd backend && uv run python ../eval/run.py --set seed
    cd backend && uv run python ../eval/run.py --test-file ../eval/out/test.jsonl \\
        --transcripts ../eval/out/transcripts.jsonl \\
        --laya zero-shot=<local base export> --laya fine-tuned=Dhwanidesk/dhwani-desk-laya@kn-ft-v1

Writes `eval/reports/<name>.md` and `.json`; numbers are never typed by hand.
"""

import argparse
import hashlib
import json
import statistics
import sys
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from common import REPORTS, SCHEMA, SEED_FILE, fingerprint, label_errors, read_jsonl

CONDITIONS = ("text", "prisma_16k", "prisma_8k_noise")
CONDITION_LABEL = {
    "text": "written text",
    "prisma_16k": "spoken, Prisma 16 kHz",
    "prisma_8k_noise": "spoken, Prisma 8 kHz + noise",
}
FLAGS = [name for name, spec in SCHEMA.items() if spec["type"] == "noul"]
CHOICES = [name for name, spec in SCHEMA.items() if spec["type"] != "noul"]
ROUTED, UNROUTED = "routed_department", "unrouted"


# --- deciders ------------------------------------------------------------------------------


def keyword_decider() -> Any:
    from dialogue.standin import KeywordDecider

    return KeywordDecider()


def laya_decider(source: str) -> Any:
    """`source`: a local checkpoint directory, or `repo@tag-or-commit` on the Hub."""
    from config import get_settings
    from decisions.laya import LayaDecider

    repo, _, revision = source.partition("@")
    settings = get_settings().model_copy(
        update={
            "decider": "laya",
            "laya_checkpoint_repo": repo,
            "laya_checkpoint_revision": revision,
        }
    )
    decider = LayaDecider.load(settings)
    if Path(source).is_dir():
        # A local copy is named only by its directory, so the report pins it by its weights.
        decider.revision += f" sha256:{weights_hash(Path(source) / 'onnx')}"
    return decider


def weights_hash(onnx_dir: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(onnx_dir.glob("model.onnx*")):
        with path.open("rb") as f:
            while block := f.read(1 << 20):
                digest.update(block)
    return digest.hexdigest()[:16]


def labels_of(decision: Any) -> dict[str, Any]:
    """A Decision as the test set's labels: yes past the threshold the call acts on. Also the
    department the call would route to: none below the confidence it needs to act."""
    from decisions import thresholds

    cut = {
        "safety_hazard": thresholds.SAFETY_HAZARD,
        "wants_human": thresholds.WANTS_HUMAN,
        "out_of_scope": thresholds.OUT_OF_SCOPE,
        "has_location": thresholds.HAS_LOCATION,
        "repeat_complaint": thresholds.REPEAT_COMPLAINT,
        "contains_personal_id": 0.5,
    }
    labels = {name: getattr(decision, name) > cut[name] for name in FLAGS}
    labels |= {name: getattr(decision, name) for name in CHOICES}
    routed = decision.department_confidence >= thresholds.DEPARTMENT_CONFIDENCE
    labels[ROUTED] = decision.department if routed else UNROUTED
    return labels


def predict(decider: Any, text: str) -> tuple[dict[str, Any], float, float]:
    """Labels, and the seconds the safety step and the whole decision took."""
    started = time.perf_counter()
    screening = decider.screen(text)
    screened = time.perf_counter()
    decision = decider.decide(text, screening)
    return labels_of(decision), screened - started, time.perf_counter() - started


# --- metrics -------------------------------------------------------------------------------


def macro_f1(gold: list[str], guess: list[str], classes: list[str]) -> float:
    """Over the `classes` that occur in `gold` or `guess`: a class the set never asks about
    is not a 0, but guessing it when the set never asks still costs."""
    scores = []
    for c in classes:
        if c not in gold and c not in guess:
            continue
        tp = sum(g == c and p == c for g, p in zip(gold, guess, strict=True))
        fp = sum(g != c and p == c for g, p in zip(gold, guess, strict=True))
        fn = sum(g == c and p != c for g, p in zip(gold, guess, strict=True))
        scores.append(2 * tp / (2 * tp + fp + fn) if tp else 0.0)
    return sum(scores) / len(scores) if scores else 0.0


def flag_scores(gold: list[bool], guess: list[bool]) -> dict[str, Any]:
    tp = sum(g and p for g, p in zip(gold, guess, strict=True))
    raised, true = sum(guess), sum(gold)
    return {
        "recall": tp / true if true else None,
        "precision": tp / raised if raised else None,
        "raised": raised,
        "true": true,
    }


def score(rows: list[dict[str, Any]], guesses: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"n": len(rows)}
    for name in CHOICES:
        gold = [r["labels"][name] for r in rows]
        guess = [g[name] for g in guesses]
        entry = {"accuracy": sum(a == b for a, b in zip(gold, guess, strict=True)) / len(rows)}
        if name == "department":
            classes = list(SCHEMA[name]["criteria"])
            entry["macro_f1"] = macro_f1(gold, guess, classes)
            entry["routed_macro_f1"] = macro_f1(gold, [g[ROUTED] for g in guesses], classes)
        result[name] = entry
    for name in FLAGS:
        result[name] = flag_scores([r["labels"][name] for r in rows], [g[name] for g in guesses])
    result["safety_handoff"] = flag_scores(
        [safety_handoff(r["labels"]) for r in rows], [safety_handoff(g) for g in guesses]
    )
    return result


def safety_handoff(labels: dict[str, Any]) -> bool:
    """The call hands off with the safety line on a hazard or on an emergency."""
    return labels["safety_hazard"] or labels["urgency"] == "emergency"


def p95(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(0.95 * (len(ordered) - 1)))]


# --- running -------------------------------------------------------------------------------


def inputs(rows: list[dict[str, Any]], transcripts: dict[str, dict[str, str]]) -> dict[str, list]:
    """For each condition, the (row, text) pairs that have that version of the sentence. The
    spoken conditions keep only the sentences both have, so the drop compares like with like."""
    by_condition: dict[str, list] = {"text": [(r, r["text"]) for r in rows]}
    spoken = [c for c in CONDITIONS[1:] if any(c in t for t in transcripts.values())]
    both = [r for r in rows if all(c in transcripts.get(r["id"], {}) for c in spoken)]
    for condition in spoken:
        if both:
            by_condition[condition] = [(r, transcripts[r["id"]][condition]) for r in both]
    return by_condition


def evaluate(
    deciders: dict[str, Callable[[], Any]], rows: list[dict[str, Any]], transcripts: dict
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for name, make in deciders.items():
        decider = make()
        results[name] = {"revision": getattr(decider, "revision", "keyword-standin")}
        for condition, pairs in inputs(rows, transcripts).items():
            guesses, safety_s, whole_s = [], [], []
            for _, text in pairs:
                labels, s, w = predict(decider, text)
                guesses.append(labels)
                safety_s.append(s)
                whole_s.append(w)
            entry = score([r for r, _ in pairs], guesses)
            entry["latency"] = {
                "safety_step_p95_s": p95(safety_s),
                "whole_decision_p95_s": p95(whole_s),
                "whole_decision_median_s": statistics.median(whole_s),
            }
            entry["mistakes"] = [
                {"id": r["id"], "department": (r["labels"]["department"], g["department"])}
                for (r, _), g in zip(pairs, guesses, strict=True)
                if r["labels"]["department"] != g["department"]
            ]
            results[name][condition] = entry
            print(
                f"{name} / {condition}: department macro-F1 {entry['department']['routed_macro_f1']:.2f}",
                flush=True,
            )
    return results


# --- report --------------------------------------------------------------------------------


def fmt(value: float | None, digits: int = 2) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def recall_cell(s: dict[str, Any]) -> str:
    return f"{fmt(s['recall'])} ({s['raised']} raised, {s['true']} true)"


def headline(results: dict[str, Any], spoken: str) -> list[str]:
    """The PRD's table, one column per decider."""
    from decisions.thresholds import DEPARTMENT_CONFIDENCE

    names = list(results)
    cond = spoken if all(spoken in results[n] for n in names) else "text"
    lines = [
        f"Measured on: {CONDITION_LABEL[cond]}, {results[names[0]][cond]['n']} sentences. "
        f"A department counts only when the call would route it (confidence ≥ "
        f"{DEPARTMENT_CONFIDENCE:.2f}); a flag counts past the threshold the call acts on.",
        "",
        "| Metric | Target | " + " | ".join(f"`{n}`" for n in names) + " |",
        "| --- | --- |" + " --- |" * len(names),
    ]

    def row(label: str, target: str, cell: Callable[[dict[str, Any]], str]) -> None:
        lines.append(
            f"| {label} | {target} | " + " | ".join(cell(results[n]) for n in names) + " |"
        )

    def latency(r: dict[str, Any]) -> str:
        if not str(r["revision"]).startswith("laya:"):
            return "—"
        lat = r[cond]["latency"]
        return f"{fmt(lat['safety_step_p95_s'])} s / {fmt(lat['whole_decision_p95_s'])} s"

    def drop(r: dict[str, Any]) -> str:
        clean, noisy = r.get("prisma_16k"), r.get("prisma_8k_noise")
        if not (clean and noisy):
            return "needs `transcribe.py`"
        a, b = clean["department"]["routed_macro_f1"], noisy["department"]["routed_macro_f1"]
        return f"{fmt(a - b)} ({fmt(a)} → {fmt(b)}, {clean['n']} sentences)"

    row("`department` macro-F1", "≥ 0.80", lambda r: fmt(r[cond]["department"]["routed_macro_f1"]))
    row("`safety_hazard` recall", "≥ 0.90", lambda r: recall_cell(r[cond]["safety_hazard"]))
    row("`wants_human` recall", "≥ 0.90", lambda r: recall_cell(r[cond]["wants_human"]))
    row(
        "Safety hand-off recall (hazard or emergency)",
        "",
        lambda r: recall_cell(r[cond]["safety_handoff"]),
    )
    row(
        "Laya latency p95, safety step / whole decision (this machine)",
        "≤ 0.6 s / ≤ 2.5 s",
        latency,
    )
    row("Department macro-F1 drop, 8 kHz + noise vs 16 kHz", "reported", drop)
    lines += [
        "",
        "End of caller speech → first agent audio (≤ 1.5 s) is measured by "
        "`backend/scripts/call_smoke.py` (#22), not here.",
        "",
        "Revisions: " + "; ".join(f"`{n}` = `{results[n]['revision']}`" for n in names) + ".",
    ]
    return lines


def detail(results: dict[str, Any]) -> list[str]:
    lines = []
    for name, by_condition in results.items():
        lines += ["", f"### {name} (`{by_condition['revision']}`)", ""]
        conds = [c for c in CONDITIONS if c in by_condition]
        lines.append("| | " + " | ".join(CONDITION_LABEL[c] for c in conds) + " |")
        lines.append("| --- |" + " --- |" * len(conds))
        for key, label in (("routed_macro_f1", "routed"), ("macro_f1", "any confidence")):
            lines.append(
                f"| department macro-F1, {label} | "
                + " | ".join(fmt(by_condition[c]["department"][key]) for c in conds)
                + " |"
            )
        for q in CHOICES:
            lines.append(
                f"| {q} accuracy | "
                + " | ".join(fmt(by_condition[c][q]["accuracy"]) for c in conds)
                + " |"
            )
        for q in [*FLAGS, "safety_handoff"]:
            cells = []
            for c in conds:
                s = by_condition[c][q]
                cells.append(
                    f"recall {fmt(s['recall'])}, precision {fmt(s['precision'])} ({s['raised']}/{s['true']})"
                )
            lines.append(f"| {q} | " + " | ".join(cells) + " |")
        lines.append(
            "| whole decision, median / p95 | "
            + " | ".join(
                f"{fmt(by_condition[c]['latency']['whole_decision_median_s'])} / {fmt(by_condition[c]['latency']['whole_decision_p95_s'])} s"
                for c in conds
            )
            + " |"
        )
    return lines


def report(results: dict[str, Any], set_name: str, note: str) -> str:
    when = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        f"# Evaluation: {set_name}",
        "",
        f"Generated by `eval/run.py` on {when}. {note}",
        "",
        "## Success metrics (docs/PRD.md)",
        "",
    ]
    lines += headline(results, "prisma_16k")
    lines += [
        "",
        "Flags show *raised / truly positive* in the detail tables, so false alarms are visible next to misses.",
        "",
        "## Detail",
    ]
    lines += detail(results)
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--set", choices=["test", "seed"], default="test")
    parser.add_argument("--test-file", type=Path, help="the test set JSONL (needed for --set test)")
    parser.add_argument("--transcripts", type=Path, help="transcribe.py's output")
    parser.add_argument("--no-keyword", action="store_true", help="skip the keyword stand-in")
    parser.add_argument(
        "--laya",
        action="append",
        default=[],
        metavar="NAME=SOURCE",
        help="a Laya checkpoint: NAME=dir or NAME=repo@revision; repeatable",
    )
    parser.add_argument("--name", help="report file name (default: the set and today's date)")
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    if args.set == "seed":
        rows, note = (
            read_jsonl(SEED_FILE),
            (
                "The seed set (data/seed) is hand-written but was shown to the data generator as "
                "examples, so it is not held out: use it to compare deciders, not to claim accuracy."
            ),
        )
    else:
        if not args.test_file:
            parser.error("--set test needs --test-file (built by build_test_set.py)")
        rows, note = (
            read_jsonl(args.test_file),
            "The held-out test set (#13): never seen by the generator or the training run.",
        )
    bad = {r["id"]: e for r in rows if (e := label_errors(r["labels"]))}
    if bad:
        raise SystemExit(f"Labels that do not fit the schema: {bad}")
    transcripts = {t["id"]: t for t in read_jsonl(args.transcripts)} if args.transcripts else {}
    texts = {r["id"]: r["text"] for r in rows}
    stale = sorted(
        i
        for i, t in transcripts.items()
        if i in texts and t.get("sha256_16") != fingerprint(texts[i])
    )
    if stale:
        raise SystemExit(
            f"Recorded from another version of the sentence: {stale}. Record them again."
        )

    deciders: dict[str, Callable[[], Any]] = {}
    if not args.no_keyword:
        deciders["keyword"] = keyword_decider
    for spec in args.laya:
        name, _, source = spec.partition("=")
        if not source:
            parser.error(f"--laya {spec!r}: use NAME=SOURCE")
        deciders[name] = lambda source=source: laya_decider(source)

    results = evaluate(deciders, rows, transcripts)
    name = args.name or f"{args.set}-{datetime.now(UTC):%Y-%m-%d}"
    REPORTS.mkdir(parents=True, exist_ok=True)
    text = report(results, "seed set" if args.set == "seed" else "held-out test set", note)
    (REPORTS / f"{name}.md").write_text(text, "utf-8")
    (REPORTS / f"{name}.json").write_text(
        json.dumps(results, indent=1, ensure_ascii=False), "utf-8"
    )
    print("\n" + "\n".join(text.splitlines()[:16]))
    print(f"\nWrote {REPORTS / name}.md")


if __name__ == "__main__":
    main()
