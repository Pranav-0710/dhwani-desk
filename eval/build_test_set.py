"""Build the held-out test set from the corrected review sheet (#13).

Reads the sheet a Kannada reader corrected (`Your Kannada`, `Fix labels`) and the draft it was
made from, applies the corrections, checks every label against data/decision_schema.json, and
writes:
- `eval/out/test.jsonl`: the test set (private: it is uploaded, never committed);
- `data/test/manifest.json`: ids, a hash of each sentence and the label counts, for git.
With `--push`, uploads the set to the private dataset repo as `test/text.jsonl`.

`Fix labels` takes `name=value` pairs separated by `;`, with the sheet's column names or the
schema's (`hazard=yes; urgency=urgent; ward=w04`), or `DROP` to leave the row out.

Usage, from backend/:
    uv run --with openpyxl python ../eval/build_test_set.py \\
        --sheet ../training/out/review/test-set-draft-v1.xlsx \\
        --draft ../training/out/review/test-set-draft-v1.json [--push]
"""

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

from common import (
    DATASET_REPO,
    MANIFEST,
    OUT,
    SCHEMA,
    TEST_FILE,
    fingerprint,
    label_errors,
    write_jsonl,
)

# The sheet's column names for the labels, and the schema's.
SHEET_NAMES = {
    "department": "department",
    "ward": "ward",
    "urgency": "urgency",
    "hazard": "safety_hazard",
    "location": "has_location",
    "repeat": "repeat_complaint",
    "officer": "wants_human",
    "out of scope": "out_of_scope",
    "id number": "contains_personal_id",
    "mood": "caller_state",
}
YES, NO = {"yes", "y", "true", "1"}, {"no", "n", "false", "0"}


def parse_fixes(text: str) -> dict[str, Any] | None:
    """`Fix labels` as {question: value}; None for DROP."""
    text = (text or "").strip()
    if text.upper() == "DROP":
        return None
    fixes: dict[str, Any] = {}
    for part in filter(None, (p.strip() for p in text.split(";"))):
        name, sep, value = part.partition("=")
        if not sep:
            raise ValueError(f"{part!r}: write name=value")
        key = SHEET_NAMES.get(name.strip().lower(), name.strip())
        if key not in SCHEMA:
            raise ValueError(f"{name.strip()!r} is not a label")
        value = value.strip()
        if SCHEMA[key]["type"] == "noul":
            if value.lower() not in YES | NO:
                raise ValueError(f"{key}: write yes or no, not {value!r}")
            fixes[key] = value.lower() in YES
        else:
            fixes[key] = value
    return fixes


def corrected(
    draft: list[dict[str, Any]], sheet: list[dict[str, str]]
) -> tuple[list[dict], list[str]]:
    """The draft with the sheet's corrections; also what was dropped or changed, for the log."""
    by_id = {row["id"]: row for row in draft}
    ids = [line["id"] for line in sheet]
    if twice := sorted({i for i in ids if ids.count(i) > 1}):
        raise SystemExit(f"In the sheet twice: {twice}")
    if unknown := sorted(set(ids) - set(by_id)):
        raise SystemExit(f"In the sheet but not the draft: {unknown}")
    if missing := sorted(set(by_id) - set(ids)):
        raise SystemExit(f"In the draft but not the sheet (write DROP to leave one out): {missing}")
    rows, log = [], []
    for line in sheet:
        row = json.loads(json.dumps(by_id[line["id"]]))
        fixes = parse_fixes(line.get("fix", ""))
        if fixes is None:
            log.append(f"{row['id']}: dropped")
            continue
        if (text := (line.get("kannada") or "").strip()) and text != row["text"]:
            row["text"] = text
            log.append(f"{row['id']}: sentence rewritten")
        if fixes:
            row["labels"].update(fixes)
            log.append(f"{row['id']}: {fixes}")
        if errors := label_errors(row["labels"]):
            raise SystemExit(f"{row['id']}: {errors}")
        rows.append(row)
    return rows, log


def read_sheet(path: Path) -> list[dict[str, str]]:
    from openpyxl import load_workbook

    sheet = load_workbook(path, read_only=True)["Test set"]
    header = [str(c or "").strip() for c in next(sheet.iter_rows(max_row=1, values_only=True))]
    column = {
        name: header.index(name) for name in ("id", "Your Kannada (if changed)", "Fix labels")
    }
    lines = []
    for values in sheet.iter_rows(min_row=2, values_only=True):
        if not values[column["id"]]:
            continue
        lines.append(
            {
                "id": str(values[column["id"]]),
                "kannada": str(values[column["Your Kannada (if changed)"]] or ""),
                "fix": str(values[column["Fix labels"]] or ""),
            }
        )
    return lines


def manifest(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = {
        name: dict(sorted(Counter(str(r["labels"][name]) for r in rows).items())) for name in SCHEMA
    }
    return {
        "set": "held-out test set (#13)",
        "where": f"{DATASET_REPO}/{TEST_FILE} (private)",
        "rows": len(rows),
        "sentences": [{"id": r["id"], "sha256_16": fingerprint(r["text"])} for r in rows],
        "label_counts": counts,
    }


def push(path: Path) -> str:
    from huggingface_hub import HfApi

    api = HfApi(token=os.environ.get("HF_TOKEN") or None)
    commit = api.upload_file(
        path_or_fileobj=str(path),
        path_in_repo=TEST_FILE,
        repo_id=DATASET_REPO,
        repo_type="dataset",
        commit_message="Held-out test set (#13)",
    )
    return commit.oid


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sheet", type=Path, required=True)
    parser.add_argument("--draft", type=Path, required=True)
    parser.add_argument("--push", action="store_true", help=f"upload to {DATASET_REPO}")
    args = parser.parse_args()

    rows, log = corrected(json.loads(args.draft.read_text("utf-8")), read_sheet(args.sheet))
    out = OUT / "test.jsonl"
    write_jsonl(out, rows)
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest(rows), indent=1, ensure_ascii=False) + "\n", "utf-8")
    print("\n".join(log) or "no corrections")
    print(f"{len(rows)} rows -> {out} and {MANIFEST}")
    if args.push:
        print(f"Uploaded {DATASET_REPO}@{push(out)}")


if __name__ == "__main__":
    main()
