"""The faster-Laya check's rule and node choice, on scripted decisions (no model)."""

import dataclasses

import pytest
from fakes import NOTHING, ROADS, ROADS_IN_W04

from scripts import int8_check as check

DANGER = dataclasses.replace(ROADS_IN_W04, safety_hazard=0.9)
EMERGENCY = dataclasses.replace(ROADS_IN_W04, urgency="emergency")
OFFICER = dataclasses.replace(ROADS, wants_human=0.9)


def test_the_call_s_own_reply_decides_what_counts() -> None:
    assert check.act(DANGER)[1] == "safety"
    assert check.act(EMERGENCY)[1] == "safety"  # an emergency hands off like a hazard
    assert check.act(OFFICER)[1] == "wants_human"
    assert check.act(ROADS_IN_W04) == (("read_back_ward", "closing"), "ticket")
    assert check.act(ROADS)[0] == ("ask_location",)


def test_an_identical_candidate_passes() -> None:
    live = [DANGER, OFFICER, ROADS_IN_W04, NOTHING]
    counts = check.compare(live, list(live))

    assert counts["passes"] is True
    assert (counts["safety_handoffs_live"], counts["officer_requests_live"]) == (1, 1)
    assert counts["same_reply"] == counts["same_routed_department"] == 4


@pytest.mark.parametrize(
    ("live", "candidate", "what"),
    [
        (DANGER, dataclasses.replace(DANGER, safety_hazard=0.55), "safety_handoffs_dropped"),
        (EMERGENCY, dataclasses.replace(EMERGENCY, urgency="urgent"), "safety_handoffs_dropped"),
        (OFFICER, dataclasses.replace(OFFICER, wants_human=0.65), "officer_requests_dropped"),
    ],
)
def test_one_dropped_hand_off_fails_the_candidate(live, candidate, what: str) -> None:
    counts = check.compare([live, ROADS_IN_W04], [candidate, ROADS_IN_W04])

    assert counts[what] == 1
    assert counts["passes"] is False


def test_extra_hand_offs_and_other_changes_are_counted_not_failed() -> None:
    live = [ROADS_IN_W04, ROADS, ROADS_IN_W04]
    candidate = [DANGER, dataclasses.replace(ROADS, department_confidence=0.5), OFFICER]
    counts = check.compare(live, candidate)

    assert counts["passes"] is True  # a false alarm is not a missed emergency
    assert (counts["safety_handoffs_added"], counts["officer_requests_added"]) == (1, 1)
    assert counts["same_reply"] == 0
    assert counts["same_routed_department"] == 2  # the unsure one is no longer routed


def test_both_models_must_answer_the_same_transcripts() -> None:
    with pytest.raises(ValueError, match="same transcripts"):
        check.compare([DANGER], [])


def test_scopes_pick_encoder_matrices_in_graph_order() -> None:
    names = [f"m{i}" for i in range(check.LAYERS * check.PER_LAYER + 3)]  # 3 head matrices

    assert check.pick_nodes(names, "all") is None
    mlp = check.pick_nodes(names, "mlp")
    assert mlp is not None and len(mlp) == 44 and mlp[:4] == ["m2", "m3", "m6", "m7"]
    assert check.pick_nodes(names, "layers:0-1") == [f"m{i}" for i in range(8)]
    assert check.pick_nodes(names, "layers:21") == ["m84", "m85", "m86", "m87"]


@pytest.mark.parametrize("scope", ["layers:0-22", "layers:5-2", "half"])
def test_a_scope_outside_the_encoder_is_refused(scope: str) -> None:
    names = [f"m{i}" for i in range(check.LAYERS * check.PER_LAYER)]
    with pytest.raises(SystemExit):
        check.pick_nodes(names, scope)


def test_a_graph_too_small_to_be_this_encoder_is_refused() -> None:
    with pytest.raises(SystemExit, match="weight matrices"):
        check.pick_nodes(["m0", "m1"], "mlp")
