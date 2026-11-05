"""Build the `v2.0` training files from the generator's rows (#67).

A 10% sample was read by hand against each sentence's scenario and labels, by a non-native reader:
`results/v2-hand-check.json` holds the row ids and verdicts. Checked rows stay only if they passed
all three questions. The others are dropped if they have more than one Kannada word that is rare
across the whole set, since wrong words and non-words tend to be rare. No sentence or label is
edited, and every dropped row is kept in `filtered_out.jsonl` with the reason.

The filter was chosen on that one small sample. It keeps 90 of the 98 good checked rows and 116 of
the 185 bad ones, so it improves the set but does not clean it.

Usage, with `--rows` holding `train.jsonl` and `report.json` from dataset commit 956696ba:
    python training/clean_v2.py --rows <dir> --out <dir>
"""

import argparse
import collections
import json
import re
from pathlib import Path
from typing import Any

HAND_CHECK = Path(__file__).resolve().parent / "results" / "v2-hand-check.json"
KANNADA_WORD = re.compile("[ಀ-೿‌‍]+")
RARE_ROWS, MAX_RARE = 2, 1
FLAGS = (
    "safety_hazard",
    "wants_human",
    "repeat_complaint",
    "out_of_scope",
    "contains_personal_id",
    "has_location",
)
Row = dict[str, Any]


def words(text: str) -> set[str]:
    return set(KANNADA_WORD.findall(text))


def clean(rows: list[Row], check: dict[str, list[str]]) -> tuple[list[Row], list[Row]]:
    """The rows kept, and the rows dropped with a `dropped` reason, both in their original order."""
    in_rows = collections.Counter(word for row in rows for word in words(row["text"]))
    sampled, good = set(check["sampled"]), set(check["all_three"])

    def reason(row: Row) -> str | None:
        if row["id"] in sampled:
            return None if row["id"] in good else "hand_check_failed"
        rare = sum(in_rows[word] <= RARE_ROWS for word in words(row["text"]))
        return "rare_words" if rare > MAX_RARE else None

    kept: list[Row] = []
    dropped: list[Row] = []
    for row in rows:
        if why := reason(row):
            dropped.append({**row, "dropped": why})
        else:
            kept.append(row)
    return kept, dropped


def label_shares(rows: list[Row]) -> dict[str, Any]:
    labels = [row["labels"] for row in rows]
    departments: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])
    for label in labels:
        departments[label["department"]][0] += 1
        departments[label["department"]][1] += bool(label["safety_hazard"])
    ordered = sorted(departments.items())
    return {
        "n": len(rows),
        **{flag: round(sum(bool(x[flag]) for x in labels) / len(rows), 3) for flag in FLAGS},
        "dept_rows": {name: count[0] for name, count in ordered},
        "dept_hazard": {name: round(count[1] / count[0], 2) for name, count in ordered},
    }


def report(
    generator_report: dict[str, Any],
    check: dict[str, list[str]],
    kept: list[Row],
    dropped: list[Row],
) -> dict[str, Any]:
    checked = len(check["sampled"])
    return {
        **generator_report,
        "hand_check": {
            "rows": checked,
            "natural": len(check["natural"]),
            "says_scenario": checked - len(check["not_scenario"]),
            "labels_right": checked - len(check["wrong_label"]),
            "all_three": len(check["all_three"]),
            "checked_by": "labelling rules and scenarios, not a native speaker",
        },
        "filter": {
            "rule": (
                f"drop a row with more than {MAX_RARE} Kannada word found in at most "
                f"{RARE_ROWS} rows; sampled rows follow the hand check"
            ),
            "dropped": dict(collections.Counter(row["dropped"] for row in dropped)),
            "kept": len(kept),
            "label_shares_kept": label_shares(kept),
        },
    }


def write_jsonl(path: Path, rows: list[Row]) -> None:
    text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    path.write_text(text, "utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rows", type=Path, required=True, help="the generator's files")
    parser.add_argument("--out", type=Path, required=True, help="where to write the v2.0 files")
    args = parser.parse_args()

    check = json.loads(HAND_CHECK.read_text("utf-8"))
    lines = (args.rows / "train.jsonl").read_text("utf-8").splitlines()
    rows = [json.loads(line) for line in lines if line.strip()]
    kept, dropped = clean(rows, check)

    args.out.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out / "train.jsonl", kept)
    write_jsonl(args.out / "filtered_out.jsonl", dropped)
    generator_report = json.loads((args.rows / "report.json").read_text("utf-8"))
    summary = report(generator_report, check, kept, dropped)
    (args.out / "report.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1), "utf-8", newline="\n"
    )
    print(f"kept {len(kept)} of {len(rows)}; dropped {summary['filter']['dropped']}")


if __name__ == "__main__":
    main()
