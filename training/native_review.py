"""A native Kannada speaker's review of the training sentences: a sheet to fill in, and the merge.

The labels and sentences so far were checked against written rules, not by a native speaker. This
makes that check cheap to do: `sheet` writes a CSV a reader can open in Excel or Google Sheets and
mark row by row; `apply` merges the marked sheet back into a new training file. Training rows only:
the held-out test set (`data/test`, `eval/out`) is never read or written here.

Usage, with `--rows` a `train.jsonl` (a dataset-repo download, or `--train-file` of finetune_laya):
    python training/native_review.py sheet --rows train.jsonl --n 300 --out review.csv
    python training/native_review.py apply --rows train.jsonl --sheet filled.csv --out reviewed/

The reader fills `verdict` with one of:
- `ok`          the sentence is natural Kannada and says what its labels say
- `fix`         the Kannada is wrong or unnatural: put the corrected sentence in `corrected_text`
- `label_wrong` the sentence is fine but the labels (department, urgency, hazard...) do not fit it
- `drop`        unusable: nonsense, or not a civic complaint at all
Blank rows are untouched. A `fix` keeps every number as it was (a changed number or an empty or
non-Kannada fix is refused), so ward and ID labels still hold.
"""

import argparse
import collections
import csv
import json
import random
import re
import sys
from pathlib import Path
from typing import Any

Row = dict[str, Any]
VERDICTS = ("ok", "fix", "label_wrong", "drop")
COLUMNS = (
    "id",
    "text",
    "department",
    "urgency",
    "safety_hazard",
    "wants_human",
    "has_location",
    "contains_personal_id",
    "verdict",
    "corrected_text",
    "note",
)
KANNADA = re.compile("[ಀ-೿]")
# Latin and Kannada digits: a fix may not add, drop or change any.
DIGITS = re.compile(r"[0-9೦-೯]+")


def read_jsonl(path: Path) -> list[Row]:
    return [json.loads(line) for line in path.read_text("utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[Row]) -> None:
    text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    path.write_text(text, "utf-8", newline="\n")


def sample(rows: list[Row], n: int, seed: int) -> list[Row]:
    """`n` rows spread evenly over the departments, in a fixed order for a seed; a department
    with too few rows gives all it has."""
    by_department: dict[str, list[Row]] = collections.defaultdict(list)
    for row in rows:
        by_department[row["labels"]["department"]].append(row)
    rng = random.Random(seed)
    for group in by_department.values():
        rng.shuffle(group)
    chosen: list[Row] = []
    groups = [by_department[name] for name in sorted(by_department)]
    while len(chosen) < n and any(groups):
        for group in groups:
            if group and len(chosen) < n:
                chosen.append(group.pop())
    return sorted(chosen, key=lambda row: row["id"])


def yes_no(value: Any) -> str:
    return "yes" if value else "no"


def sheet_rows(rows: list[Row]) -> list[dict[str, str]]:
    return [
        {
            "id": row["id"],
            "text": row["text"],
            "department": str(row["labels"]["department"]),
            "urgency": str(row["labels"]["urgency"]),
            "safety_hazard": yes_no(row["labels"]["safety_hazard"]),
            "wants_human": yes_no(row["labels"]["wants_human"]),
            "has_location": yes_no(row["labels"]["has_location"]),
            "contains_personal_id": yes_no(row["labels"]["contains_personal_id"]),
            "verdict": "",
            "corrected_text": "",
            "note": "",
        }
        for row in rows
    ]


def write_sheet(path: Path, rows: list[dict[str, str]]) -> None:
    # A byte-order mark, so Excel opens the Kannada as UTF-8.
    with path.open("w", encoding="utf-8-sig", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def read_sheet(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as source:
        return list(csv.DictReader(source))


def check_fix(row: Row, fixed: str) -> str | None:
    """Why `fixed` cannot replace the row's sentence, or None when it can."""
    if not fixed.strip():
        return "a fix needs corrected_text"
    if not KANNADA.search(fixed):
        return "corrected_text has no Kannada"
    if DIGITS.findall(fixed) != DIGITS.findall(row["text"]):
        return "corrected_text changes a number (ward numbers and IDs must stay as they were)"
    return None


def apply(rows: list[Row], marked: list[dict[str, str]]) -> tuple[list[Row], list[Row], dict]:
    """The reviewed rows, the rows taken out (each with a `dropped` reason), and a summary.
    Stops on a sheet that does not belong to these rows or that is filled in wrongly."""
    by_id = {row["id"]: row for row in rows}
    seen: set[str] = set()
    verdicts: dict[str, dict[str, str]] = {}
    problems: list[str] = []
    for line, mark in enumerate(marked, start=2):  # line 1 is the header
        row_id, verdict = mark["id"], mark["verdict"].strip().lower()
        if row_id not in by_id:
            problems.append(f"line {line}: {row_id!r} is not in the rows")
            continue
        if row_id in seen:
            problems.append(f"line {line}: {row_id!r} appears twice")
        seen.add(row_id)
        if mark["text"] != by_id[row_id]["text"]:
            problems.append(f"line {line}: {row_id} was edited in `text`; use corrected_text")
        if not verdict:
            continue
        if verdict not in VERDICTS:
            problems.append(f"line {line}: {row_id} verdict {verdict!r} is not one of {VERDICTS}")
        elif verdict == "fix" and (why := check_fix(by_id[row_id], mark["corrected_text"])):
            problems.append(f"line {line}: {row_id}: {why}")
        verdicts[row_id] = {**mark, "verdict": verdict}
    if problems:
        raise SystemExit("The sheet cannot be applied:\n" + "\n".join(problems))

    kept: list[Row] = []
    dropped: list[Row] = []
    for row in rows:
        mark = verdicts.get(row["id"])
        if mark is None:
            kept.append(row)
        elif mark["verdict"] == "ok":
            kept.append({**row, "reviewed": "native"})
        elif mark["verdict"] == "fix":
            fixed = mark["corrected_text"].strip()
            kept.append({**row, "text": fixed, "reviewed": "native-fixed"})
        else:
            dropped.append({**row, "dropped": mark["verdict"], "note": mark["note"]})
    counts = collections.Counter(mark["verdict"] for mark in verdicts.values())
    summary = {
        "rows": len(rows),
        "reviewed": len(verdicts),
        "verdicts": dict(counts),
        "kept": len(kept),
        "dropped": len(dropped),
        "share_natural": round(counts["ok"] / len(verdicts), 3) if verdicts else None,
        "reviewed_by": "a native Kannada speaker (the sheet's reader)",
    }
    return kept, dropped, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    make = commands.add_parser("sheet", help="write the sheet for a reader to fill in")
    make.add_argument("--rows", type=Path, required=True, help="a train.jsonl")
    make.add_argument("--n", type=int, default=300, help="how many rows to ask about")
    make.add_argument("--seed", type=int, default=0)
    make.add_argument("--out", type=Path, required=True, help="the CSV to write")
    merge = commands.add_parser("apply", help="merge a filled-in sheet into a new training set")
    merge.add_argument("--rows", type=Path, required=True, help="the train.jsonl it was made from")
    merge.add_argument("--sheet", type=Path, required=True, help="the filled-in CSV")
    merge.add_argument("--out", type=Path, required=True, help="a folder for the reviewed files")
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    rows = read_jsonl(args.rows)
    if args.command == "sheet":
        chosen = sample(rows, args.n, args.seed)
        write_sheet(args.out, sheet_rows(chosen))
        print(f"{len(chosen)} rows of {len(rows)} written to {args.out}")
        return
    kept, dropped, summary = apply(rows, read_sheet(args.sheet))
    args.out.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out / "train.jsonl", kept)
    write_jsonl(args.out / "reviewed_out.jsonl", dropped)
    (args.out / "review.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1), "utf-8", newline="\n"
    )
    print(f"kept {len(kept)} of {len(rows)}; {summary['verdicts']}")


if __name__ == "__main__":
    main()
