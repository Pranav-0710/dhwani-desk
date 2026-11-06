import dataclasses

import pytest

from decisions.safety_phrases import RULESET, WithSafetyPhrases, asks_for_officer, says_hazard
from dialogue.decision import Decision, Screening

QUIET = Screening(safety_hazard=0.1, urgency="soon", wants_human=0.2, out_of_scope=0.3)


class Stub:
    revision = "stub-1"

    def __init__(self, answer: Screening = QUIET) -> None:
        self.answer = answer
        self.seen: Screening | None = None

    def screen(self, transcript: str) -> Screening:
        return self.answer

    def decide(self, transcript: str, screening: Screening) -> Decision:
        self.seen = screening
        return Decision(
            department="water",
            department_confidence=0.9,
            ward="unknown",
            urgency=screening.urgency,
            safety_hazard=screening.safety_hazard,
            has_location=0.1,
            repeat_complaint=0.1,
            wants_human=screening.wants_human,
            caller_state="calm",
            out_of_scope=screening.out_of_scope,
            contains_personal_id=0.0,
        )


@pytest.mark.parametrize(
    "text",
    [
        "ಕಂಬದಿಂದ ವೈರ್ ಬಿದ್ದಿದೆ",
        "ಸ್ಪಾರ್ಕ್ ಆಗ್ತಿದೆ",
        "ಮ್ಯಾನ್ ಹೋಲ್ ಮುಚ್ಚಳ ಇಲ್ಲ",
        "ಮ್ಯಾನ್ಹೋಲ್ ತೆರೆದಿದೆ",
        "ಚರಂಡಿ ನೀರು ಮನೆ ಒಳಗೆ ಬರ್ತಿದೆ",
        "ಕುಡಿಯೋ ನೀರಲ್ಲಿ ಚರಂಡಿ ನೀರು",
        "ನಾಯಿ ಮಗುಗೆ ಕಚ್ಚಿದೆ",
        "ಮನೆ ಹತ್ರ ಹಾವು ಬಂದಿದೆ",
        "ಕಸಕ್ಕೆ ಬೆಂಕಿ ಹಾಕಿದ್ದಾರೆ",
    ],
)
def test_a_danger_the_rules_name_counts(text: str) -> None:
    assert says_hazard(text)


@pytest.mark.parametrize(
    "text",
    [
        "ಕುಡಿಯೋ ನೀರು ಬಂದಿಲ್ಲ",  # no drinking water: a complaint, not contamination
        "ಪೈಪ್ ಲೈನ್ ಮುಚ್ಚಳ ಇಲ್ಲ",  # a lid, but not a manhole's
        "ರಸ್ತೆಯಲ್ಲಿ ಗುಂಡಿ ಇದೆ",
        "ಬೀದಿ ದೀಪ ಉರಿತಿಲ್ಲ",
    ],
)
def test_an_everyday_complaint_is_not_a_danger(text: str) -> None:
    assert not says_hazard(text)


@pytest.mark.parametrize(
    "text",
    [
        "ಆಫೀಸರ್ ಜೊತೆ ಮಾತಾಡ್ಬೇಕು",
        "ಅಧಿಕಾರಿ ಜೊತೆ ಮಾತಾಡಿಸಿ",
        "ನನಗೆ ಒಬ್ಬ ಆಫೀಸರ್ ಬೇಕು",
        "ಮೇಲಧಿಕಾರಿಯನ್ನ ಕೊಡಿ",
        "ಸೂಪರ್ವೈಸರ್ಗೆ ಫೋನ್ ಕೊಡಿ",
        "ಯಾರಾದ್ರೂ ಮನುಷ್ಯರು ಮಾತಾಡಿ",
        "ಮೆಷಿನ್ ಬೇಡ, ಆಫೀಸರ್ ಕೊಡಿ",
        "ರೋಬೋಟ್ ಜೊತೆ ಮಾತಾಡಕ್ಕಾಗಲ್ಲ",
        "ಆಫೀಸರ್, ಬೇಕು.",
    ],
)
def test_asking_for_a_person_counts(text: str) -> None:
    assert asks_for_officer(text)


@pytest.mark.parametrize(
    "text",
    [
        "ಆಫೀಸರ್ ಬಂದಿಲ್ಲ",  # the officer never came
        "ಅಧಿಕಾರಿ ಮಾತಾಡ್ತಿಲ್ಲ",  # the officer is not talking
        "ಆಫೀಸರ್ ಬಂದು ನೋಡ್ಬೇಕು",  # should come and look: a visit, not a call
        "ಸಂಬಳ ಬೇಕು",  # asking, but for no person
        "ಆಫೀಸರ್ ಹೇಳಿದ್ರು ನಾಳೆ ಬರ್ತೀವಿ ಅಂತ ನೀರು ಬೇಕು",  # the asking is too far away
    ],
)
def test_mentioning_an_officer_does_not_count(text: str) -> None:
    assert not asks_for_officer(text)


def test_a_hit_raises_only_its_own_answer() -> None:
    answer = WithSafetyPhrases(Stub()).screen("ಮನೆ ಹತ್ರ ಹಾವು ಬಂದಿದೆ")
    assert answer == dataclasses.replace(QUIET, safety_hazard=1.0)
    answer = WithSafetyPhrases(Stub()).screen("ಆಫೀಸರ್ ಜೊತೆ ಮಾತಾಡ್ಬೇಕು")
    assert answer == dataclasses.replace(QUIET, wants_human=1.0)


def test_no_hit_leaves_the_answer_alone() -> None:
    assert WithSafetyPhrases(Stub()).screen("ರಸ್ತೆಯಲ್ಲಿ ಗುಂಡಿ ಇದೆ") == QUIET


def test_it_never_lowers_what_the_decider_raised() -> None:
    sure = Screening(safety_hazard=0.97, urgency="emergency", wants_human=0.9, out_of_scope=0.1)
    assert WithSafetyPhrases(Stub(sure)).screen("ರಸ್ತೆಯಲ್ಲಿ ಗುಂಡಿ ಇದೆ") == sure


def test_the_raised_answers_reach_the_full_decision() -> None:
    inner = Stub()
    decider = WithSafetyPhrases(inner)
    decision = decider.decide("x", decider.screen("ಬೆಂಕಿ ಬಿದ್ದಿದೆ"))
    assert decision.safety_hazard == 1.0
    assert inner.seen is not None and inner.seen.safety_hazard == 1.0


def test_the_revision_names_the_list() -> None:
    assert WithSafetyPhrases(Stub()).revision == f"stub-1+{RULESET}"
