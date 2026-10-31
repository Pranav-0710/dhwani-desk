"""The questions Laya answers on every caller turn, split into the two steps of a decision.

Built from data/decision_schema.json, the file the training data was labelled with, so Laya is
served the same wording it is fine-tuned on. Two of its ten questions are answered by code
instead (decisions/extractors.py, per docs/decisions/laya-kannada-gate.md).

A shared contract: the state machine, the console and the eval harness depend on these names.
"""

import json
from typing import Any

from config import REPO_ROOT
from dialogue.decision import CALLER_STATES, URGENCY_LEVELS

SCHEMA: dict[str, Any] = json.loads(
    (REPO_ROOT / "data" / "decision_schema.json").read_text("utf-8")
)["questions"]

DECIDED_IN_CODE = ("ward", "contains_personal_id")
# Asked first and on their own, because each can end the call (Dialogue.screened).
SAFETY_QUESTIONS = ("safety_hazard", "urgency", "wants_human", "out_of_scope")
OTHER_QUESTIONS = ("department", "has_location", "repeat_complaint", "caller_state")

if missing := SCHEMA.keys() - {*DECIDED_IN_CODE, *SAFETY_QUESTIONS, *OTHER_QUESTIONS}:
    raise ValueError(f"decision_schema.json has questions nobody answers: {sorted(missing)}")
if SCHEMA["urgency"]["criteria"] != list(URGENCY_LEVELS):
    raise ValueError("urgency levels differ between decision_schema.json and the dialogue")
if list(SCHEMA["caller_state"]["criteria"]) != list(CALLER_STATES):
    raise ValueError("caller states differ between decision_schema.json and the dialogue")


def laya_questions(names: tuple[str, ...]) -> dict[str, dict[str, Any]]:
    """The named questions in the form Laya takes: type, instructions, criteria."""
    questions = {}
    for name in names:
        spec = SCHEMA[name]
        question = {"type": spec["type"], "instructions": spec["instructions"]}
        if "criteria" in spec:
            question["criteria"] = spec["criteria"]
        questions[name] = question
    return questions


SAFETY = laya_questions(SAFETY_QUESTIONS)
OTHERS = laya_questions(OTHER_QUESTIONS)
