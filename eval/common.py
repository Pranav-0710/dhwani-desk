"""What every eval script shares: where things live, and the test set's row format.

A test row is the seed format (`data/seed/seed.jsonl`): `{id, text, gloss, labels}`, with
`labels` keyed by the questions in `data/decision_schema.json`. The text test set stays private
(the dataset repo's `test/text.jsonl`); only `data/test/manifest.json` is in git.
"""

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
BACKEND = REPO / "backend"
# The eval imports the backend's deciders, so the numbers come from the code that runs calls.
sys.path.insert(0, str(BACKEND))

SCHEMA: dict[str, Any] = json.loads((REPO / "data" / "decision_schema.json").read_text("utf-8"))[
    "questions"
]
WARD_IDS = {w["id"] for w in json.loads((REPO / "data" / "wards.json").read_text("utf-8"))["wards"]}
OUT = REPO / "eval" / "out"  # audio and transcripts: private, git-ignored
REPORTS = REPO / "eval" / "reports"
SEED_FILE = REPO / "data" / "seed" / "seed.jsonl"
MANIFEST = REPO / "data" / "test" / "manifest.json"
DATASET_REPO = "Dhwanidesk/dhwani-desk-data"
TEST_FILE = "test/text.jsonl"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text("utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")


def label_errors(labels: dict[str, Any]) -> list[str]:
    """Why these labels do not fit the schema; empty when they do."""
    errors = [f"missing {name}" for name in SCHEMA if name not in labels]
    for name, value in labels.items():
        spec = SCHEMA.get(name)
        if spec is None:
            errors.append(f"unknown question {name}")
        elif spec["type"] == "noul":
            if not isinstance(value, bool):
                errors.append(f"{name}={value!r} is not yes/no")
        elif name == "ward":
            if value != "unknown" and value not in WARD_IDS:
                errors.append(f"ward={value!r} is not a ward id")
        elif value not in spec["criteria"]:
            errors.append(f"{name}={value!r} is not one of {list(spec['criteria'])}")
    return errors


def fingerprint(text: str) -> str:
    """A short hash of a sentence, so the manifest can pin the set without holding its text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
