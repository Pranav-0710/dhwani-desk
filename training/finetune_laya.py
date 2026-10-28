"""Fine-tune Laya's multilingual checkpoint on the synthetic Kannada set and publish it (#14).

The week-1 gate (docs/decisions/laya-kannada-gate.md) found zero-shot Laya unusable on Kannada
callers, so the build runs a fine-tuned checkpoint. This script:

1. downloads `train.jsonl` from the dataset repo at a pinned tag (made if missing);
2. turns every row into a Laya training row: the caller's Kannada sentence as the state, and
   the eight questions the backend asks Laya, worded exactly as it asks them;
3. fine-tunes the multilingual checkpoint at Laya's reviewed commit with `laya.train.finetune`;
4. exports it to ONNX with the upstream exporter at a pinned commit, checked against PyTorch;
5. uploads it to the model repo in the layout the backend loads, and tags the commit.

Runs on Kaggle (03_finetune_laya.ipynb); `--dry-run` checks the data on a laptop without the
model weights. `ward` and `contains_personal_id` are not trained: the backend decides them in
code (decisions/extractors.py).
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
SCHEMA: dict[str, Any] = json.loads((REPO / "data" / "decision_schema.json").read_text("utf-8"))[
    "questions"
]

# The questions the backend asks Laya (backend/decisions/schema.py); the rest are decided in code.
DECIDED_IN_CODE = ("ward", "contains_personal_id")
LAYA_ANSWERS = tuple(name for name in SCHEMA if name not in DECIDED_IN_CODE)

BASE_REPO = "convaiinnovations/laya-multilingual"
# Laya 0.4.6's reviewed commit of the base (laya.revisions.PINNED_REVISIONS).
BASE_REVISION = "e4e9ddf21a7b1903b7acffd8814ad4307bf63a67"
DATASET_REPO = "Dhwanidesk/dhwani-desk-data"
MODEL_REPO = "Dhwanidesk/dhwani-desk-laya"
# The exporter that produced the base export timed in docs/decisions/laya-latency.md.
EXPORTER_REPO = "https://github.com/NandhaKishorM/laya"
EXPORTER_COMMIT = "0d0c73e317724325b9a12dc9d7d376232b3e3b25"
EXPORTER = "laya-dotnet/tools/export_onnx.py"
# Published beside the graph: what the backend loads, and what the run was.
PUBLISHED = ("tokenizer", "encoder", "questions.json", "train_report.json", "train_run.json")
# A full run is published only if the exported graph, loaded as the backend loads it, gets at
# least this share of the seed set's departments right and does not answer them all the same.
# Zero-shot scored 0.25 macro-F1 (docs/decisions/laya-kannada-gate.md).
MIN_DEPARTMENT_ACCURACY = 0.5
MIN_DEPARTMENTS_PREDICTED = 3
SEED_FILE = REPO / "data" / "seed" / "seed.jsonl"


def laya_questions(names: tuple[str, ...] = LAYA_ANSWERS) -> dict[str, dict[str, Any]]:
    """The questions as the backend serves them: type, instructions and criteria, no more."""
    questions = {}
    for name in names:
        spec = SCHEMA[name]
        question = {"type": spec["type"], "instructions": spec["instructions"]}
        if "criteria" in spec:
            question["criteria"] = spec["criteria"]
        questions[name] = question
    return questions


QUESTIONS = laya_questions()


def expected(name: str, label: Any) -> Any:
    """A label as Laya's trainer takes it: a score question wants the level's index."""
    spec = SCHEMA[name]
    return spec["criteria"].index(label) if spec["type"] == "score" else label


def training_row(row: dict[str, Any]) -> dict[str, Any]:
    """One dataset row as Laya's `{state, questions, expected}` training row."""
    labels = row["labels"]
    return {
        "id": row["id"],
        "state": row["text"],
        "questions": QUESTIONS,
        "expected": {name: expected(name, labels[name]) for name in LAYA_ANSWERS},
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text("utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")


def check_labels(rows: list[dict[str, Any]]) -> None:
    """Stop before training on a label the questions cannot express (a schema drift)."""
    for row in rows:
        for name in LAYA_ANSWERS:
            value, spec = row["labels"][name], SCHEMA[name]
            ok = isinstance(value, bool) if spec["type"] == "noul" else value in spec["criteria"]
            if not ok:
                raise SystemExit(f"{row['id']}: {name}={value!r} is not an answer to {name}")


# --- Hugging Face Hub ---------------------------------------------------------------------


def hub():
    from huggingface_hub import HfApi

    return HfApi(token=os.environ.get("HF_TOKEN") or None)


def token_owner(api: Any) -> str:
    """Whose HF_TOKEN this is, or a stop with the reason it does not work."""
    if not os.environ.get("HF_TOKEN"):
        raise SystemExit("HF_TOKEN is empty: add it as a notebook secret and attach it")
    try:
        return api.whoami()["name"]
    except Exception as exc:
        raise SystemExit(f"HF_TOKEN is not a working Hugging Face token: {exc}") from exc


def pinned_dataset(revision: str, create: bool) -> str:
    """The dataset commit behind the tag `revision`; with `create`, tag the current head."""
    from huggingface_hub.errors import RepositoryNotFoundError

    api = hub()
    try:
        refs = api.list_repo_refs(DATASET_REPO, repo_type="dataset")
    except RepositoryNotFoundError as exc:
        # The Hub answers 404, not 403, for a private repo the token may not see.
        raise SystemExit(
            f"HF_TOKEN (user {token_owner(api)}) cannot see the private {DATASET_REPO}. Use a "
            "token of a Dhwanidesk member: a classic write token, or a fine-grained one with "
            "read and write access to the Dhwanidesk organization's repos."
        ) from exc
    tags = {tag.name: tag.target_commit for tag in refs.tags}
    if revision in tags:
        return tags[revision]
    if not create:
        raise SystemExit(f"{DATASET_REPO} has no tag {revision!r}; pass --tag-dataset to make it")
    head = api.dataset_info(DATASET_REPO).sha
    api.create_tag(
        DATASET_REPO,
        tag=revision,
        revision=head,
        repo_type="dataset",
        tag_message="Training set of the first fine-tune (#12, #14)",
    )
    print(f"Tagged {DATASET_REPO}@{head[:10]} as {revision}", flush=True)
    return head


def download_train(commit: str, out: Path) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            DATASET_REPO, "train.jsonl", repo_type="dataset", revision=commit, local_dir=out
        )
    )


def base_checkpoint(weights: bool = True) -> str:
    from huggingface_hub import snapshot_download

    files = ["rl_agent_config.json", "tokenizer/*", "encoder/*"]
    return snapshot_download(
        BASE_REPO,
        revision=BASE_REVISION,
        allow_patterns=[*files, "model.safetensors"] if weights else files,
    )


def stage_upload(checkpoint: Path, export: Path, stage: Path) -> None:
    """Lay the checkpoint out as the backend loads it: config and tokenizer at the root, the
    graph in onnx/, plus the PyTorch weights for later runs and the run's reports."""
    shutil.rmtree(stage, ignore_errors=True)
    (stage / "onnx").mkdir(parents=True)
    for name in PUBLISHED:
        source = checkpoint / name
        if source.is_dir():
            shutil.copytree(source, stage / name)
        elif source.exists():
            shutil.copy2(source, stage / name)
    # The exporter's copy of the config is the one the graph was checked against.
    shutil.copy2(export / "rl_agent_config.json", stage / "rl_agent_config.json")
    link_or_copy(checkpoint / "model.safetensors", stage / "model.safetensors")
    for name in ("model.onnx", "model.onnx.data"):
        link_or_copy(export / name, stage / "onnx" / name)


def link_or_copy(source: Path, target: Path) -> None:
    """A hard link where the filesystem allows one: the weights are over a gigabyte each."""
    target.unlink(missing_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def ready_to_publish(tag: str) -> None:
    """Fail before hours of training, not after: the token reaches the repo and the tag is free."""
    api = hub()
    try:
        user = api.whoami()["name"]
        api.repo_info(MODEL_REPO)
        tags = {t.name for t in api.list_repo_refs(MODEL_REPO).tags}
    except Exception as exc:
        raise SystemExit(f"Cannot reach {MODEL_REPO} with HF_TOKEN: {exc}") from exc
    if tag in tags:
        raise SystemExit(f"{MODEL_REPO} already has {tag!r}; choose a new --push-tag")
    print(f"Will publish as {user} to {MODEL_REPO}, tag {tag}", flush=True)


def publish(stage: Path, tag: str, note: str) -> str:
    """Upload the staged checkpoint and tag the commit; returns the commit."""
    api = hub()
    if tag in {t.name for t in api.list_repo_refs(MODEL_REPO).tags}:
        raise SystemExit(f"{MODEL_REPO} already has {tag!r}; choose a new --push-tag")
    try:
        commit = api.upload_folder(
            folder_path=str(stage),
            repo_id=MODEL_REPO,
            commit_message=f"Fine-tuned checkpoint {tag}",
        )
    except Exception as exc:
        raise SystemExit(
            f"Upload failed ({exc}). The checkpoint is staged in {stage}; upload it with\n"
            f"  huggingface-cli upload {MODEL_REPO} {stage} .\n"
            f"  huggingface-cli tag {MODEL_REPO} {tag}"
        ) from exc
    print(f"Uploaded {MODEL_REPO}@{commit.oid}", flush=True)
    api.create_tag(MODEL_REPO, tag=tag, revision=commit.oid, tag_message=note)
    return commit.oid


def decode(answers: dict[str, Any]) -> dict[str, Any]:
    """Laya's answers as the backend reads them (backend/decisions/laya.py)."""
    labels = {}
    for name in LAYA_ANSWERS:
        answer, spec = answers[name], SCHEMA[name]
        if spec["type"] == "noul":
            labels[name] = answer["noul"] >= 0.5
        elif spec["type"] == "choice":
            labels[name] = answer["choice"]
        else:
            levels = answer["probabilities"]
            labels[name] = spec["criteria"][int(max(levels, key=levels.get))]
    return labels


def score(rows: list[dict[str, Any]], predicted: list[dict[str, Any]]) -> dict[str, Any]:
    report: dict[str, Any] = {"rows": len(rows), "note": "seed set: a load check, not the eval"}
    for name in LAYA_ANSWERS:
        pairs = [
            (row["labels"][name], guess[name]) for row, guess in zip(rows, predicted, strict=True)
        ]
        entry: dict[str, Any] = {"accuracy": round(sum(g == p for g, p in pairs) / len(pairs), 3)}
        if SCHEMA[name]["type"] == "noul":
            entry["raised / true"] = f"{sum(p for _, p in pairs)} / {sum(g for g, _ in pairs)}"
        report[name] = entry
    report["departments_predicted"] = len({guess["department"] for guess in predicted})
    return report


def check_published(stage: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Load the staged checkpoint with ONNXAgent, as the backend does, and score it on `rows`."""
    from laya.onnx_agent import ONNXAgent

    agent = ONNXAgent(str(stage), onnx_path=str(stage / "onnx" / "model.onnx"))
    predicted = [decode(agent.system_one(row["text"], QUESTIONS)["answers"]) for row in rows]
    return score(rows, predicted)


def good_enough(report: dict[str, Any]) -> bool:
    return (
        report["department"]["accuracy"] >= MIN_DEPARTMENT_ACCURACY
        and report["departments_predicted"] >= MIN_DEPARTMENTS_PREDICTED
    )


# --- steps --------------------------------------------------------------------------------


def export_onnx(checkpoint: Path, work: Path) -> Path:
    """Export with the pinned upstream exporter, verified against PyTorch; returns its folder."""
    # One clone for every run in the session, without the repo's large LFS files.
    source = work.parent / "laya-src"
    git = {**os.environ, "GIT_LFS_SKIP_SMUDGE": "1"}
    if not source.exists():
        clone = ["git", "clone", "-q", "--filter=blob:none", EXPORTER_REPO, str(source)]
        subprocess.run(clone, check=True, env=git)
    checkout = ["git", "-C", str(source), "checkout", "-q", EXPORTER_COMMIT]
    subprocess.run(checkout, check=True, env=git)
    out = work / "export"
    out.mkdir(parents=True, exist_ok=True)
    # The exporter compares its graph with PyTorch weights it finds beside its output, and
    # failing that with whatever Laya checkpoint is in the Hugging Face cache: the base, not ours.
    link_or_copy(checkpoint / "model.safetensors", out / "model.safetensors")
    export = [sys.executable, str(source / EXPORTER), str(checkpoint), "--out", str(out)]
    subprocess.run([*export, "--verify", "--force"], check=True)
    return out / checkpoint.name


def train(data: Path, base: str, out: Path, args: argparse.Namespace) -> dict[str, Any]:
    from laya.train import TrainConfig, dry_run, finetune

    config = TrainConfig(
        epochs=args.epochs,
        micro_batch=args.micro_batch,
        grad_accum=args.grad_accum,
        # Options in a new order every epoch, so department is not learnt by position.
        shuffle_options=("choice",),
        # ~1,600 held-out items to fit the temperatures on; Laya's default 400 leaves about
        # five emergencies.
        calib_max=1600,
        calib_frac=0.07,
        seed=args.seed,
    )
    summary = dry_run(str(data), base, config)
    if summary["skipped"]:
        raise SystemExit(f"Laya cannot train on some questions: {summary['skipped']}")
    if args.dry_run:
        return summary
    return finetune(str(data), base, str(out), config)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset-tag", default="v1.0", help="dataset repo tag to train on")
    parser.add_argument("--tag-dataset", action="store_true", help="make --dataset-tag if missing")
    parser.add_argument("--train-file", type=Path, help="a local train.jsonl instead of the Hub")
    parser.add_argument("--limit", type=int, help="train on the first N rows only (a smoke run)")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--micro-batch", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("out/finetune"))
    parser.add_argument("--dry-run", action="store_true", help="check the data; no training")
    parser.add_argument("--no-export", action="store_true")
    parser.add_argument("--push-tag", help="upload to the model repo and tag it, e.g. kn-ft-v1")
    args = parser.parse_args()
    if args.push_tag and (args.limit or args.dry_run or args.no_export):
        parser.error("--push-tag needs a full run: no --limit, --dry-run or --no-export")
    if args.push_tag:
        ready_to_publish(args.push_tag)

    started = time.monotonic()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.train_file:
        source, dataset = args.train_file, f"local:{args.train_file.name}"
    else:
        commit = pinned_dataset(args.dataset_tag, args.tag_dataset)
        source, dataset = download_train(commit, args.out / "data"), f"{DATASET_REPO}@{commit}"
    rows = read_jsonl(source)[: args.limit]
    check_labels(rows)
    data = args.out / "laya_train.jsonl"
    write_jsonl(data, [training_row(row) for row in rows])
    print(f"{len(rows)} rows from {dataset}, {len(LAYA_ANSWERS)} questions each", flush=True)

    checkpoint = args.out / "ckpt"
    summary = train(data, base_checkpoint(weights=not args.dry_run), checkpoint, args)
    run = {
        "dataset": dataset,
        "rows": len(rows),
        "base": f"{BASE_REPO}@{BASE_REVISION}",
        "questions": list(LAYA_ANSWERS),
        "epochs": args.epochs,
        "effective_batch": args.micro_batch * args.grad_accum,
        "minutes": round((time.monotonic() - started) / 60, 1),
        "summary": {k: v for k, v in summary.items() if k != "train_report"},
    }
    (args.out / "run.json").write_text(json.dumps(run, indent=2, default=str), "utf-8")
    if args.dry_run or args.no_export:
        print(json.dumps(run["summary"], indent=2, default=str))
        return
    shutil.copy2(args.out / "run.json", checkpoint / "train_run.json")

    export = export_onnx(checkpoint, args.out)
    print(f"Exported and verified: {export}", flush=True)
    stage = args.out / "publish"
    stage_upload(checkpoint, export, stage)
    report = check_published(stage, read_jsonl(SEED_FILE))
    (stage / "load_check.json").write_text(json.dumps(report, indent=2), "utf-8")
    print(json.dumps(report, indent=2), flush=True)
    if not args.push_tag:
        return
    if not good_enough(report):
        raise SystemExit(
            f"Not published: department accuracy {report['department']['accuracy']} over "
            f"{report['departments_predicted']} departments is below the bar "
            f"({MIN_DEPARTMENT_ACCURACY}, {MIN_DEPARTMENTS_PREDICTED}). Staged in {stage}."
        )
    note = f"Fine-tuned on {dataset} from {BASE_REPO}@{BASE_REVISION}"
    commit = publish(stage, args.push_tag, note)
    print(f"Published {MODEL_REPO}@{commit} as {args.push_tag}", flush=True)
    print(f"Backend: DECIDER=laya, LAYA_CHECKPOINT_REVISION={args.push_tag}", flush=True)


if __name__ == "__main__":
    main()
