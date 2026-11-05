import copy
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

COMPARE_PATH = Path(__file__).resolve().parents[2] / "eval" / "compare.py"
spec = importlib.util.spec_from_file_location("eval_compare", COMPARE_PATH)
assert spec and spec.loader
compare = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = compare
spec.loader.exec_module(compare)


def flag(recall: float, precision: float, true: int) -> dict[str, float]:
    return {"recall": recall, "precision": precision, "true": true}


def condition() -> dict[str, Any]:
    return {
        "department": {"routed_macro_f1": 0.70},
        "ward": {"accuracy": 0.90},
        "urgency": {"accuracy": 0.50},
        "caller_state": {"accuracy": 0.70},
        "safety_hazard": flag(30 / 34, 0.58, 34),
        "safety_handoff": flag(30 / 34, 0.58, 34),
        "wants_human": flag(3 / 7, 0.60, 7),
        "has_location": flag(0.95, 0.90, 87),
        "repeat_complaint": flag(6 / 13, 1.0, 13),
        "out_of_scope": flag(6 / 18, 0.75, 18),
        "contains_personal_id": flag(3 / 4, 1.0, 4),
        "latency": {"whole_decision_p95_s": 5.0},
    }


def report() -> dict[str, Any]:
    return {compare.HEADLINE: condition(), compare.NOISY: condition()}


def improved() -> dict[str, Any]:
    """Better on one gain and unchanged everywhere else."""
    new = report()
    new[compare.HEADLINE]["department"]["routed_macro_f1"] = 0.72
    return new


def test_a_real_gain_with_every_guardrail_held_switches() -> None:
    assert compare.decide(report(), improved())[0] is True


def test_matching_the_old_checkpoint_keeps_it() -> None:
    assert compare.decide(report(), report())[0] is False


@pytest.mark.parametrize("question", ["safety_hazard", "safety_handoff", "wants_human"])
@pytest.mark.parametrize("audio", ["prisma_16k", "prisma_8k_noise"])
def test_one_more_missed_emergency_keeps_the_old_checkpoint(question: str, audio: str) -> None:
    new = improved()
    new[audio][question]["recall"] -= 1 / new[audio][question]["true"]
    switch, guards, _ = compare.decide(report(), new)
    assert switch is False
    assert [(c.name, c.condition) for c in guards if not c.passed] == [
        (f"{question} recall", audio)
    ]


def test_raising_every_call_as_a_hazard_does_not_pass_as_better() -> None:
    new = improved()
    for audio in (compare.HEADLINE, compare.NOISY):
        new[audio]["safety_hazard"].update(recall=1.0, precision=0.23)
    assert compare.decide(report(), new)[0] is False


def test_a_gain_elsewhere_does_not_pay_for_a_worse_department_score() -> None:
    new = report()
    new[compare.HEADLINE]["wants_human"]["recall"] = 5 / 7
    assert compare.decide(report(), new)[0] is True
    new[compare.NOISY]["department"]["routed_macro_f1"] = 0.68
    assert compare.decide(report(), new)[0] is False


def test_one_sentence_may_move_a_small_question_but_two_may_not() -> None:
    new = improved()
    new[compare.HEADLINE]["contains_personal_id"]["recall"] = 2 / 4
    assert compare.decide(report(), new)[0] is True
    new[compare.HEADLINE]["contains_personal_id"]["recall"] = 1 / 4
    assert compare.decide(report(), new)[0] is False


def test_two_more_officer_requests_count_as_a_gain_but_one_does_not() -> None:
    new = report()
    new[compare.HEADLINE]["wants_human"]["recall"] = 4 / 7
    assert compare.decide(report(), new)[0] is False
    new[compare.HEADLINE]["wants_human"]["recall"] = 5 / 7
    assert compare.decide(report(), new)[0] is True


def test_a_much_slower_checkpoint_keeps_the_old_one() -> None:
    new = improved()
    new[compare.HEADLINE]["latency"]["whole_decision_p95_s"] = 6.5
    assert compare.decide(report(), new)[0] is False


def test_a_report_without_spoken_scores_cannot_decide() -> None:
    new = improved()
    del new[compare.NOISY]
    with pytest.raises(SystemExit, match=compare.NOISY):
        compare.decide(report(), new)


def test_the_command_writes_the_decision_and_exits_with_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "report.json"
    path.write_text(json.dumps({"old": report(), "new": copy.deepcopy(improved())}), "utf-8")
    monkeypatch.setattr(compare, "REPORTS", tmp_path)
    monkeypatch.setattr(
        sys, "argv", ["compare.py", "--report", str(path), "--old", "old", "--new", "new"]
    )
    with pytest.raises(SystemExit) as finished:
        compare.main()
    assert finished.value.code == 0
    assert (tmp_path / "switch-new.md").read_text("utf-8").startswith("# new against old: SWITCH")
