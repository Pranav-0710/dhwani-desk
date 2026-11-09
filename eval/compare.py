"""Say whether a new Laya checkpoint may replace the live one, from one `eval/run.py` report.

The rule was fixed and committed before the new checkpoint was scored, so it cannot be bent to fit
the scores. The test set is small (34 hazards, 7 officer requests), so a tie is not a win and every
check runs on the spoken audio a real call would hear.

SWITCH needs every guardrail on both spoken conditions and at least one gain at 16 kHz.

Guardrails:
- `safety_hazard`, `safety_handoff` and `wants_human` recall do not fall at all: PRD, missing an
  emergency is worse than a false alarm.
- `safety_hazard` precision falls by at most 0.05: raising every call would otherwise pass.
- Routed department macro-F1 falls by at most 0.01.
- Every other question falls by at most 0.05, or one sentence if that is more.
- Whole-decision p95 latency is at most 20% slower, compared within the one report. With
  `--new-report` the two p95s come from separate runs, so machine load can decide it: the check
  still counts, and the report says so.

Gains (at least one):
- Routed department macro-F1 up by 0.02 or more.
- `safety_hazard` precision up by 0.05 or more: fewer false hazards.
- `wants_human` recall up by two sentences or more.

Usage, from the repo root:
    python eval/compare.py --report eval/reports/test-kn-ft-v2.json --old kn-ft-v1 --new kn-ft-v2

`--new-report` takes the new answers from a second report, to compare the same checkpoint on two
speech models' transcripts (#88).

Writes `eval/reports/switch-<new>.md`. Exit code 0 means SWITCH, 1 means KEEP.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, NamedTuple

REPORTS = Path(__file__).resolve().parent / "reports"
HEADLINE, NOISY = "prisma_16k", "prisma_8k_noise"
NO_FALL = ("safety_hazard", "safety_handoff", "wants_human")
OTHER_FLAGS = ("has_location", "repeat_complaint", "out_of_scope", "contains_personal_id")
OTHER_CHOICES = ("ward", "urgency", "caller_state")
OTHER_SLIP, PRECISION_SLIP, DEPARTMENT_SLIP = 0.05, 0.05, 0.01
SLOWER = 1.2
DEPARTMENT_GAIN, PRECISION_GAIN, OFFICER_GAIN_SENTENCES = 0.02, 0.05, 2
EPSILON = 1e-9


class Check(NamedTuple):
    name: str
    condition: str
    old: float
    new: float
    needed: float
    passed: bool


def _at_least(name: str, condition: str, old: float, new: float, needed: float) -> Check:
    return Check(name, condition, old, new, needed, new >= needed - EPSILON)


def guardrails(
    old: dict[str, Any], new: dict[str, Any], condition: str, same_run: bool = True
) -> list[Check]:
    a, b = old[condition], new[condition]
    checks = [
        _at_least(f"{q} recall", condition, a[q]["recall"], b[q]["recall"], a[q]["recall"])
        for q in NO_FALL
    ]
    precision = a["safety_hazard"]["precision"]
    checks.append(
        _at_least(
            "safety_hazard precision",
            condition,
            precision,
            b["safety_hazard"]["precision"],
            precision - PRECISION_SLIP,
        )
    )
    department = a["department"]["routed_macro_f1"]
    checks.append(
        _at_least(
            "department macro-F1, routed",
            condition,
            department,
            b["department"]["routed_macro_f1"],
            department - DEPARTMENT_SLIP,
        )
    )
    for q in OTHER_CHOICES:
        was = a[q]["accuracy"]
        checks.append(
            _at_least(f"{q} accuracy", condition, was, b[q]["accuracy"], was - OTHER_SLIP)
        )
    for q in OTHER_FLAGS:
        was = a[q]["recall"]
        slip = max(OTHER_SLIP, 1 / a[q]["true"])
        checks.append(_at_least(f"{q} recall", condition, was, b[q]["recall"], was - slip))
    latency = a["latency"]["whole_decision_p95_s"], b["latency"]["whole_decision_p95_s"]
    limit = latency[0] * SLOWER
    checks.append(
        Check(
            "whole decision p95, s" + ("" if same_run else " (two separate runs)"),
            condition,
            latency[0],
            latency[1],
            limit,
            latency[1] <= limit + EPSILON,
        )
    )
    return checks


def gains(old: dict[str, Any], new: dict[str, Any]) -> list[Check]:
    a, b = old[HEADLINE], new[HEADLINE]
    department = a["department"]["routed_macro_f1"]
    precision = a["safety_hazard"]["precision"]
    officer = a["wants_human"]["recall"]
    return [
        _at_least(
            "department macro-F1, routed",
            HEADLINE,
            department,
            b["department"]["routed_macro_f1"],
            department + DEPARTMENT_GAIN,
        ),
        _at_least(
            "safety_hazard precision",
            HEADLINE,
            precision,
            b["safety_hazard"]["precision"],
            precision + PRECISION_GAIN,
        ),
        _at_least(
            "wants_human recall",
            HEADLINE,
            officer,
            b["wants_human"]["recall"],
            officer + OFFICER_GAIN_SENTENCES / a["wants_human"]["true"],
        ),
    ]


def decide(
    old: dict[str, Any], new: dict[str, Any], same_run: bool = True
) -> tuple[bool, list[Check], list[Check]]:
    """Whether to switch, then every guardrail and every gain behind that answer. `same_run`:
    both come from one report, so their latencies were measured side by side."""
    missing = [c for c in (HEADLINE, NOISY) for r in (old, new) if c not in r]
    if missing:
        raise SystemExit(f"The report has no spoken scores for: {sorted(set(missing))}")
    guards = guardrails(old, new, HEADLINE, same_run) + guardrails(old, new, NOISY, same_run)
    won = gains(old, new)
    return all(c.passed for c in guards) and any(c.passed for c in won), guards, won


def _table(checks: list[Check]) -> list[str]:
    lines = [
        "| Check | Audio | Old | New | Needed | Passed |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    lines += [
        f"| {c.name} | {c.condition} | {c.old:.3f} | {c.new:.3f} | {c.needed:.3f} | "
        f"{'yes' if c.passed else '**no**'} |"
        for c in checks
    ]
    return lines


def render(
    old_name: str, new_name: str, switch: bool, guards: list[Check], won: list[Check]
) -> str:
    verdict = f"SWITCH to `{new_name}`" if switch else f"KEEP `{old_name}`"
    return "\n".join(
        [
            f"# {new_name} against {old_name}: {verdict}",
            "",
            "Generated by `eval/compare.py`; the rule is in its docstring.",
            "",
            "## Guardrails: every one must pass",
            "",
            *_table(guards),
            "",
            "## Gains: at least one must pass",
            "",
            *_table(won),
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--report", type=Path, required=True, help="eval/run.py's JSON report")
    parser.add_argument("--old", required=True, help="the live checkpoint's name in the report")
    parser.add_argument("--new", required=True, help="the new checkpoint's name in the report")
    parser.add_argument("--new-report", type=Path, help="take --new from this report instead")
    parser.add_argument("--name", help="report name (default: switch-<new>)")
    parser.add_argument(
        "--labels", nargs=2, metavar=("OLD", "NEW"), help="names shown in the report title"
    )
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    results = json.loads(args.report.read_text("utf-8"))
    new_results = json.loads(args.new_report.read_text("utf-8")) if args.new_report else results
    switch, guards, won = decide(
        results[args.old], new_results[args.new], same_run=args.new_report is None
    )
    old_label, new_label = args.labels or (args.old, args.new)
    text = render(old_label, new_label, switch, guards, won)
    name = args.name or f"switch-{args.new}"
    (REPORTS / f"{name}.md").write_text(text, "utf-8")
    print(text)
    sys.exit(0 if switch else 1)


if __name__ == "__main__":
    main()
