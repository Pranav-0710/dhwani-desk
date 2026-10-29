"""Every transition of the dialogue state machine, driven by scripted decisions."""

import dataclasses
import re

import pytest
from fakes import NOTHING, ROADS, ROADS_IN_W04, ScriptedDecider

from dialogue.decision import Decision, Screening
from dialogue.machine import (
    MAX_LOCATION_ASKS,
    MAX_UNROUTED_TURNS,
    CountingDesk,
    Dialogue,
    State,
)
from dialogue.templates import LINE_KEYS, load_templates, placeholders

TEMPLATES = load_templates()


def dialogue(*decisions: Decision | Exception) -> tuple[Dialogue, ScriptedDecider]:
    decider = ScriptedDecider(*decisions)
    return Dialogue(TEMPLATES, decider, CountingDesk(start=1004)), decider


def test_the_call_opens_with_the_greeting_and_listens() -> None:
    call, _ = dialogue()
    reply = call.greet()
    assert reply.keys == ("greeting",)
    assert reply.lines == (TEMPLATES.lines["greeting"],)
    assert reply.state is State.LISTENING


def test_a_complete_first_turn_reads_back_the_ticket_and_closes() -> None:
    call, _ = dialogue(ROADS_IN_W04)
    reply = call.turn("ಗಾಂಧಿನಗರ ಅಲ್ಲಿ ರೋಡ್ ತುಂಬಾ ಹಾಳಾಗಿದೆ")

    assert reply.keys == ("read_back_ward", "closing")
    assert reply.state is State.CLOSED and call.state is State.CLOSED
    assert reply.reason == "ticket"
    assert reply.ticket is not None
    assert (reply.ticket.number, reply.ticket.department, reply.ticket.ward) == (
        "1004",
        "roads",
        "w04",
    )
    # The ward is read by name, the department by its Kannada label, the number digit by digit.
    assert "ಗಾಂಧಿನಗರ" in reply.lines[0] and "ರಸ್ತೆ" in reply.lines[0]
    assert "1 0 0 4" in reply.lines[0]


def test_ticket_numbers_keep_counting_across_calls() -> None:
    desk = CountingDesk(start=1004)
    numbers = []
    for _ in range(2):
        call = Dialogue(TEMPLATES, ScriptedDecider(ROADS_IN_W04), desk)
        numbers.append(call.turn("ರೋಡ್").ticket.number)
    assert numbers == ["1004", "1005"]


def test_not_a_civic_complaint_is_redirected_and_closed_without_a_ticket() -> None:
    call, _ = dialogue(dataclasses.replace(NOTHING, out_of_scope=0.9))
    reply = call.turn("ಕರೆಂಟ್ ಬಿಲ್ ತುಂಬಾ ಜಾಸ್ತಿ ಬಂದಿದೆ")
    assert reply.keys == ("redirect",)
    assert (reply.state, reply.reason, reply.ticket) == (State.CLOSED, "out_of_scope", None)


@pytest.mark.parametrize(
    ("text", "decision"),
    [
        ("ಎದೆ ನೋವು ಆಂಬುಲೆನ್ಸ್ ಬೇಕು", dataclasses.replace(NOTHING, safety_hazard=0.95)),
        ("ಮ್ಯಾನ್ ಹೋಲ್ ಮುಚ್ಚಳ ಇಲ್ಲ ಆಂಬುಲೆನ್ಸ್ ಕರೆದಿದ್ದೀವಿ", dataclasses.replace(ROADS, urgency="emergency")),
    ],
    ids=["chest_pain", "open_manhole_and_ambulance"],
)
def test_a_risk_that_is_also_out_of_scope_gets_112_and_an_officer(
    text: str, decision: Decision
) -> None:
    # Safety comes first: redirecting and hanging up on someone in danger is the worst mistake.
    call, _ = dialogue(dataclasses.replace(decision, out_of_scope=0.9))
    reply = call.turn(text)
    assert reply.keys == ("safety_redirect",)
    assert (reply.state, reply.reason) == (State.HANDED_OFF, "safety")


@pytest.mark.parametrize(
    "decision",
    [
        dataclasses.replace(ROADS_IN_W04, safety_hazard=0.8),
        dataclasses.replace(ROADS_IN_W04, urgency="emergency"),
    ],
    ids=["hazard", "emergency"],
)
def test_a_safety_risk_hands_off_with_the_safety_line(decision: Decision) -> None:
    call, _ = dialogue(decision)
    reply = call.turn("ಕಂಬದಿಂದ ವೈರ್ ಬಿದ್ದು ಸ್ಪಾರ್ಕ್ ಆಗ್ತಿದೆ")
    assert reply.keys == ("safety",)
    assert (reply.state, reply.reason) == (State.HANDED_OFF, "safety")
    # The officer taking over sees what the agent already knows.
    assert (call.slots.department, call.slots.ward) == ("roads", "w04")


def test_a_caller_who_asks_for_a_person_is_handed_off() -> None:
    call, _ = dialogue(dataclasses.replace(ROADS, wants_human=0.8))
    reply = call.turn("ಆಫೀಸರ್ ಜೊತೆ ಮಾತಾಡಬೇಕು")
    assert reply.keys == ("handoff_human",)
    assert (reply.state, reply.reason) == (State.HANDED_OFF, "wants_human")


@pytest.mark.parametrize(
    ("screening", "key", "reason"),
    [
        (
            Screening(safety_hazard=0.8, urgency="soon", wants_human=0.1, out_of_scope=0.1),
            "safety",
            "safety",
        ),
        (
            Screening(safety_hazard=0.8, urgency="soon", wants_human=0.1, out_of_scope=0.9),
            "safety_redirect",
            "safety",
        ),
        (
            Screening(safety_hazard=0.1, urgency="soon", wants_human=0.8, out_of_scope=0.1),
            "handoff_human",
            "wants_human",
        ),
    ],
)
def test_the_safety_step_alone_can_hand_off(screening: Screening, key: str, reason: str) -> None:
    call, _ = dialogue()
    call.hear("...")
    reply = call.screened(screening)
    assert reply is not None and reply.keys == (key,)
    assert (reply.state, reply.reason) == (State.HANDED_OFF, reason)
    assert (reply.decision, reply.screening) == (None, screening)


def test_an_emergency_outranks_a_request_for_an_officer_in_either_step() -> None:
    # "My father has chest pain, connect me to an officer": the caller must still hear 112.
    chest_pain = dataclasses.replace(
        NOTHING, safety_hazard=0.5, urgency="emergency", wants_human=0.9, out_of_scope=0.9
    )
    call, decider = dialogue(chest_pain)
    reply = call.turn("ಅಪ್ಪನಿಗೆ ಎದೆ ನೋವು, ಅಧಿಕಾರಿಗೆ ಕನೆಕ್ಟ್ ಮಾಡಿ")
    assert (reply.keys, reply.reason) == (("safety_redirect",), "safety")
    assert reply.screening is not None  # handed off on the safety step
    assert Dialogue(TEMPLATES, decider, CountingDesk()).respond(chest_pain).keys == (
        "safety_redirect",
    )


def test_not_a_civic_matter_waits_for_the_full_decision() -> None:
    # A distressed caller with an out-of-scope problem is handed off, not redirected, and only
    # the full decision knows how the caller is.
    call, _ = dialogue()
    call.hear("...")
    assert (
        call.screened(
            Screening(safety_hazard=0.1, urgency="soon", wants_human=0.1, out_of_scope=0.9)
        )
        is None
    )
    assert call.state is State.LISTENING


def test_a_failure_after_a_safety_hand_off_keeps_the_hand_off() -> None:
    class FailsAfterScreening(ScriptedDecider):
        def decide(self, transcript: str, screening: Screening) -> Decision:
            raise RuntimeError("model crashed")

    decider = FailsAfterScreening(dataclasses.replace(ROADS, safety_hazard=0.9))
    call = Dialogue(TEMPLATES, decider, CountingDesk())
    reply = call.turn("ಕಂಬದಿಂದ ವೈರ್ ಬಿದ್ದು ಸ್ಪಾರ್ಕ್ ಆಗ್ತಿದೆ")
    assert (reply.keys, reply.reason) == (("safety",), "safety")


@pytest.mark.parametrize("caller_state", ["distressed", "abusive"])
def test_a_distressed_or_abusive_caller_is_handed_off_calmly(caller_state: str) -> None:
    call, _ = dialogue(dataclasses.replace(ROADS_IN_W04, caller_state=caller_state))
    reply = call.turn("...")
    assert reply.keys == ("handoff_calm",)
    assert (reply.state, reply.reason) == (State.HANDED_OFF, caller_state)


def test_thresholds_are_strict_so_a_borderline_answer_is_not_acted_on() -> None:
    borderline = dataclasses.replace(
        ROADS_IN_W04, out_of_scope=0.7, safety_hazard=0.6, wants_human=0.7
    )
    call, _ = dialogue(borderline)
    assert call.turn("ರೋಡ್").keys == ("read_back_ward", "closing")


def test_an_escalation_on_a_later_turn_still_hands_off() -> None:
    call, _ = dialogue(ROADS, dataclasses.replace(ROADS, wants_human=0.9))
    assert call.turn("ರೋಡ್ ಹಾಳಾಗಿದೆ").keys == ("ask_location",)
    reply = call.turn("ಆಫೀಸರ್ ಜೊತೆ ಮಾತಾಡಬೇಕು")
    assert reply.keys == ("handoff_human",)
    assert reply.state is State.HANDED_OFF


def test_a_failing_decider_hands_off_instead_of_going_quiet() -> None:
    call, decider = dialogue(RuntimeError("model crashed"), ROADS_IN_W04)
    reply = call.turn("ರೋಡ್ ಹಾಳಾಗಿದೆ")
    assert reply.keys == ("handoff_unclear",)
    assert (reply.state, reply.reason, reply.decision) == (State.HANDED_OFF, "decider_error", None)
    assert call.turn("ಹಲೋ").keys == ()
    assert len(decider.transcripts) == 1


def test_a_decision_that_arrives_after_the_call_ended_changes_nothing() -> None:
    call, _ = dialogue(dataclasses.replace(NOTHING, wants_human=0.9))
    call.turn("ಆಫೀಸರ್ ಬೇಕು")
    assert call.respond(ROADS_IN_W04).keys == ()
    assert call.state is State.HANDED_OFF


def test_an_unclear_problem_is_asked_about_then_handed_off() -> None:
    call, _ = dialogue(*[NOTHING] * MAX_UNROUTED_TURNS)
    first = call.turn("ಹಲೋ")
    assert first.keys == ("ask_problem",)
    assert first.state is State.LISTENING
    last = call.turn("ಅದೇ ಸಮಸ್ಯೆ")
    assert last.keys == ("handoff_unclear",)
    assert (last.state, last.reason) == (State.HANDED_OFF, "unroutable")


def test_a_missing_location_is_asked_for_then_handed_off() -> None:
    call, _ = dialogue(*[ROADS] * (MAX_LOCATION_ASKS + 1))
    for _ in range(MAX_LOCATION_ASKS):
        assert call.turn("ರೋಡ್ ಹಾಳಾಗಿದೆ").keys == ("ask_location",)
    last = call.turn("ಗೊತ್ತಿಲ್ಲ")
    assert last.keys == ("handoff_unclear",)
    assert (last.state, last.reason) == (State.HANDED_OFF, "no_location")


def test_a_place_without_a_ward_asks_for_the_ward_once_then_files_without_it() -> None:
    near_a_shop = dataclasses.replace(ROADS, has_location=0.9)
    call, _ = dialogue(near_a_shop, near_a_shop)
    assert call.turn("ಮೆಡಿಕಲ್ ಶಾಪ್ ಎದುರು ಗುಂಡಿ ಇದೆ").keys == ("ask_ward",)
    reply = call.turn("ಗೊತ್ತಿಲ್ಲ")
    assert reply.keys == ("read_back", "closing")
    assert reply.ticket is not None and reply.ticket.ward == "unknown"


def test_slots_filled_on_earlier_turns_are_kept() -> None:
    ward_only = dataclasses.replace(NOTHING, ward="w07", has_location=0.9)
    call, _ = dialogue(ROADS, ward_only)
    assert call.turn("ರೋಡ್ ಹಾಳಾಗಿದೆ").keys == ("ask_location",)
    # The second answer is unsure about the department, but the first one was confident.
    reply = call.turn("ತೋಟದಪಾಳ್ಯ")
    assert reply.keys == ("read_back_ward", "closing")
    assert (reply.ticket.department, reply.ticket.ward) == ("roads", "w07")


def test_the_decider_reads_the_whole_call_so_far() -> None:
    call, decider = dialogue(ROADS, ROADS_IN_W04)
    call.turn(" ರೋಡ್ ಹಾಳಾಗಿದೆ ")
    call.turn("ಗಾಂಧಿನಗರ")
    assert decider.transcripts == ["ರೋಡ್ ಹಾಳಾಗಿದೆ", "ರೋಡ್ ಹಾಳಾಗಿದೆ ಗಾಂಧಿನಗರ"]


def test_a_repeat_complaint_is_noted_before_the_read_back() -> None:
    call, _ = dialogue(dataclasses.replace(ROADS_IN_W04, repeat_complaint=0.9))
    reply = call.turn("ಮೂರನೇ ಸಲ ಕಾಲ್ ಮಾಡ್ತಿದೀನಿ")
    assert reply.keys == ("repeat_noted", "read_back_ward", "closing")
    assert reply.ticket.repeat_complaint


def test_a_frustrated_caller_gets_one_apology() -> None:
    frustrated = dataclasses.replace(ROADS, caller_state="frustrated")
    call, _ = dialogue(frustrated, dataclasses.replace(frustrated, ward="w04"))
    assert call.turn("ಇನ್ನೂ ಸರಿ ಮಾಡಿಲ್ಲ").keys == ("apology", "ask_location")
    assert call.turn("ಗಾಂಧಿನಗರ").keys == ("read_back_ward", "closing")


@pytest.mark.parametrize(
    "decision",
    [dataclasses.replace(NOTHING, wants_human=0.9), ROADS_IN_W04],
    ids=["handed_off", "closed"],
)
def test_after_the_call_ends_the_agent_stays_quiet(decision: Decision) -> None:
    call, decider = dialogue(decision)
    call.turn("ಮೊದಲ ಮಾತು")
    reply = call.turn("ಇನ್ನೊಂದು ಮಾತು")
    assert reply.keys == () and reply.lines == ()
    assert len(decider.transcripts) == 1  # nobody was asked to decide again


def test_a_blank_turn_is_ignored() -> None:
    call, decider = dialogue()
    reply = call.turn("   ")
    assert reply.keys == () and reply.state is State.LISTENING
    assert decider.transcripts == [] and call.turns == []


def test_the_dialogue_event_carries_state_slots_ticket_and_decision() -> None:
    call, _ = dialogue(ROADS_IN_W04)
    event = call.turn("ರೋಡ್").to_client(call.slots)
    assert event["type"] == "dialogue"
    assert event["keys"] == ["read_back_ward", "closing"]
    assert (event["state"], event["reason"]) == ("closed", "ticket")
    assert event["slots"] == {"department": "roads", "ward": "w04", "location_given": True}
    assert event["ticket"]["number"] == "1004"
    assert event["decision"]["department_confidence"] == 0.9


def test_templates_cover_every_line_and_department_in_kannada_only() -> None:
    assert TEMPLATES.lines.keys() >= LINE_KEYS
    for key, text in TEMPLATES.lines.items():
        spoken = re.sub(r"\{\w+\}", "", text)  # placeholder names are not spoken
        assert not re.search(r"[A-Za-z]", spoken), key
        assert placeholders(text) <= {"department", "ward", "ticket"}, key
    assert set(TEMPLATES.departments) == {
        "water",
        "sewage-drainage",
        "solid-waste",
        "roads",
        "streetlights",
        "stray-animals",
        "other",
    }
    assert len(TEMPLATES.wards) == 10


def test_only_lines_without_placeholders_are_cached() -> None:
    fixed = TEMPLATES.fixed_lines()
    assert TEMPLATES.lines["greeting"] in fixed
    assert TEMPLATES.lines["read_back_ward"] not in fixed
    assert len(fixed) == len(TEMPLATES.lines) - 2  # both read-back lines carry the ticket
