"""Laya serving: the answers made by code, the questions Laya is asked, and how LayaDecider
turns Laya's answers into a decision (on a stand-in for Laya, so no model is needed)."""

import json
from typing import Any

import pytest
from fakes import isolated_settings

from config import REPO_ROOT
from decisions.extractors import find_ward, says_personal_number
from decisions.laya import MAX_TRANSCRIPT_TOKENS, LayaDecider, recent
from decisions.schema import (
    DECIDED_IN_CODE,
    OTHER_QUESTIONS,
    OTHERS,
    SAFETY,
    SAFETY_QUESTIONS,
    SCHEMA,
)
from dialogue.decision import Screening

SEED = [
    json.loads(line)
    for line in (REPO_ROOT / "data" / "seed" / "seed.jsonl").read_text("utf-8").splitlines()
    if line.strip()
]


@pytest.mark.parametrize("row", SEED, ids=lambda row: row["id"])
def test_code_answers_the_ward_and_personal_number_as_the_seed_set_labels_them(row: dict) -> None:
    assert find_ward(row["text"]) == row["labels"]["ward"]
    assert says_personal_number(row["text"]) == row["labels"]["contains_personal_id"]


@pytest.mark.parametrize(
    ("text", "ward"),
    [
        ("ಹೂವಿನಕೇರಿ ಅಲ್ಲ, ಗಾಂಧಿನಗರ", "w04"),  # a correction: the last ward said wins
        ("ಗಾಂಧಿ ನಗರ ಅಲ್ಲಿ ಕಸ", "w04"),  # spaced differently
        ("ನಾಲ್ಕನೇ ವಾರ್ಡ್ ಅಲ್ಲಿ ಕಸ", "w04"),
        ("4ನೇ ವಾರ್ಡ್", "w04"),
        ("ವಾರ್ಡ್ ನಂಬರ್ 10", "w10"),
        ("ವಾರ್ಡ್ 11 ಅಲ್ಲಿ ಕಸ", "unknown"),  # no such ward
        ("ನಮ್ಮ ಮನೆ ಹತ್ತಿರ ಕಸ", "unknown"),
    ],
)
def test_the_ward_is_found_however_it_is_said(text: str, ward: str) -> None:
    assert find_ward(text) == ward


def test_laya_is_asked_every_other_question_in_the_training_wording() -> None:
    asked = {**SAFETY, **OTHERS}
    assert list(SAFETY) == list(SAFETY_QUESTIONS) and list(OTHERS) == list(OTHER_QUESTIONS)
    assert asked.keys() == SCHEMA.keys() - set(DECIDED_IN_CODE)
    for name, question in asked.items():
        assert question["instructions"] == SCHEMA[name]["instructions"]
        assert question["type"] == SCHEMA[name]["type"]


# What Laya might answer, by question, in the shape `ONNXAgent.system_one` returns.
ANSWERS: dict[str, dict[str, Any]] = {
    "safety_hazard": {"type": "noul", "noul": 0.12},
    "wants_human": {"type": "noul", "noul": 0.05},
    "out_of_scope": {"type": "noul", "noul": 0.2},
    "department": {
        "type": "choice",
        "choice": "water",
        "probabilities": {"water": 0.83, "roads": 0.1, "other": 0.07},
    },
    # Most likely "routine", though the expected level (1.53) would round to "urgent".
    "urgency": {"type": "score", "probabilities": {"0": 0.42, "1": 0.03, "2": 0.15, "3": 0.4}},
    "has_location": {"type": "noul", "noul": 0.9},
    "repeat_complaint": {"type": "noul", "noul": 0.3},
    "caller_state": {"type": "choice", "choice": "frustrated"},
}


class FakeLaya:
    def __init__(self) -> None:
        self.asked: list[tuple[str, list[str]]] = []

    def system_one(self, state: str, questions: dict[str, dict[str, Any]]) -> dict[str, Any]:
        self.asked.append((state, list(questions)))
        return {
            "answers": {name: ANSWERS[name] for name in questions},
            "usage": {"truncated_questions": []},
        }


def test_the_safety_step_asks_laya_only_the_safety_questions() -> None:
    laya = FakeLaya()
    screening = LayaDecider(laya, "laya:test@abc").screen("ನೀರು ಬಂದಿಲ್ಲ")
    # Urgency is the most likely level, not the expected one.
    assert screening == Screening(
        safety_hazard=0.12, urgency="routine", wants_human=0.05, out_of_scope=0.2
    )
    assert laya.asked == [("ನೀರು ಬಂದಿಲ್ಲ", list(SAFETY_QUESTIONS))]


def test_the_full_decision_combines_laya_code_and_the_safety_step() -> None:
    laya = FakeLaya()
    decider = LayaDecider(laya, "laya:test@abc")
    transcript = "ಹೂವಿನಕೇರಿ ಅಲ್ಲಿ ನೀರು ಬಂದಿಲ್ಲ ನನ್ನ ನಂಬರ್ 9876543210"
    screening = Screening(safety_hazard=0.12, urgency="soon", wants_human=0.05, out_of_scope=0.2)

    decision = decider.decide(transcript, screening)

    assert laya.asked == [(transcript, list(OTHER_QUESTIONS))]
    assert (decision.department, decision.department_confidence) == ("water", 0.83)
    assert (decision.has_location, decision.repeat_complaint) == (0.9, 0.3)
    assert decision.caller_state == "frustrated"
    # From code, not Laya.
    assert (decision.ward, decision.contains_personal_id) == ("w01", 1.0)
    # From the safety step, not asked again.
    assert (
        decision.safety_hazard,
        decision.urgency,
        decision.wants_human,
        decision.out_of_scope,
    ) == (0.12, "soon", 0.05, 0.2)
    assert decider.revision == "laya:test@abc"


def test_a_long_call_shows_laya_its_latest_words_but_keeps_an_early_ward() -> None:
    laya = FakeLaya()
    transcript = "ಗಾಂಧಿನಗರ ಅಲ್ಲಿ ರೋಡ್ ಹಾಳಾಗಿದೆ " + "ತುಂಬಾ ತೊಂದರೆ ಆಗ್ತಿದೆ " * 200
    screening = Screening(safety_hazard=0.1, urgency="soon", wants_human=0.1, out_of_scope=0.1)

    decision = LayaDecider(laya, "laya:test@abc").decide(transcript, screening)

    ((state, _),) = laya.asked
    assert len(state) <= MAX_TRANSCRIPT_TOKENS and transcript.endswith(state)  # by characters here
    assert transcript[-len(state) - 1] == " "  # cut at a whole word
    assert decision.ward == "w04"


def test_a_short_call_is_shown_whole() -> None:
    assert recent("ರೋಡ್ ಹಾಳಾಗಿದೆ", len) == "ರೋಡ್ ಹಾಳಾಗಿದೆ"


def test_a_long_call_keeps_as_many_of_its_latest_words_as_fit() -> None:
    words = [f"w{i}" for i in range(1000)]

    def one_token_per_word(text: str) -> int:
        return len(text.split())

    kept = recent(" ".join(words), one_token_per_word).split()
    assert kept == words[-MAX_TRANSCRIPT_TOKENS:]


@pytest.mark.parametrize("revision", ["", "main"])
def test_a_hub_checkpoint_must_be_pinned_to_a_tag_or_commit(revision: str) -> None:
    settings = isolated_settings(decider="laya", laya_checkpoint_revision=revision)
    with pytest.raises(ValueError, match="LAYA_CHECKPOINT_REVISION"):
        LayaDecider.load(settings)
