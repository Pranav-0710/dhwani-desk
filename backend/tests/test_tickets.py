"""Ticket storage: masking, numbering, retention, the console API and what the dialogue files."""

import dataclasses
from datetime import UTC, datetime, timedelta

import pytest
from fakes import NOTHING, ROADS, ROADS_IN_W04, ScriptedDecider, isolated_settings
from fastapi.testclient import TestClient

import app as app_module
from dialogue.machine import CountingDesk, Dialogue
from dialogue.templates import load_templates
from tickets.masking import mask_personal_ids
from tickets.store import TicketRequest, TicketStore


def request(**overrides: object) -> TicketRequest:
    fields: dict = {
        "call_id": "call-1",
        "department": "roads",
        "ward": "w04",
        "urgency": "soon",
        "repeat_complaint": False,
        "location_text": "ಗಾಂಧಿನಗರ ಸರ್ಕಾರಿ ಆಸ್ಪತ್ರೆ ಹತ್ತಿರ",
        "decision": ROADS_IN_W04,
        "decider_revision": "scripted",
    }
    return TicketRequest(**(fields | overrides))


@pytest.mark.parametrize(
    ("text", "personal_id", "expected"),
    [
        ("ನನ್ನ ನಂಬರ್ 9876543210", 0.1, "ನನ್ನ ನಂಬರ್ XXXXXXXXXX"),
        ("ಆಧಾರ್ 1234 5678 9012 ಲಿಂಕ್", 0.1, "ಆಧಾರ್ XXXX XXXX XXXX ಲಿಂಕ್"),
        ("98450-12345 ಗೆ ಕಾಲ್ ಮಾಡಿ", 0.1, "XXXXX-XXXXX ಗೆ ಕಾಲ್ ಮಾಡಿ"),
        # Short numbers stay unless the decider heard a personal ID...
        ("ಖಾತೆ 4521 ವಾರ್ಡ್ 4", 0.1, "ಖಾತೆ 4521 ವಾರ್ಡ್ 4"),
        # ...then four digits and up go, but a ward number or a count stays readable.
        ("ಖಾತೆ 4521 ವಾರ್ಡ್ 4 ಮೂರು ದಿನ 3", 0.9, "ಖಾತೆ XXXX ವಾರ್ಡ್ 4 ಮೂರು ದಿನ 3"),
        # Punctuation between digit groups, and Kannada digits.
        ("98450, 12345", 0.1, "XXXXX, XXXXX"),
        ("(984) 501-2345", 0.1, "(XXX) XXX-XXXX"),
        ("984.501.2345", 0.1, "XXX.XXX.XXXX"),
        ("98450 — 12345", 0.1, "XXXXX — XXXXX"),
        ("೯೮೪೫೦೧೨೩೪೫", 0.1, "XXXXXXXXXX"),
    ],
)
def test_personal_numbers_are_masked(text: str, personal_id: float, expected: str) -> None:
    assert mask_personal_ids(text, personal_id) == expected


def test_tickets_are_numbered_from_1001_and_survive_a_restart(tmp_path) -> None:
    path = tmp_path / "dhwani.db"
    store = TicketStore(path)
    assert [store.file(request()), store.file(request(ward="w07"))] == ["1001", "1002"]
    store.close()

    reopened = TicketStore(path)
    assert [t["number"] for t in reopened.tickets()] == ["1002", "1001"]  # newest first
    assert reopened.file(request()) == "1003"
    ticket = reopened.ticket(1001)
    assert ticket is not None
    assert (ticket["department"], ticket["ward"], ticket["status"]) == ("roads", "w04", "open")
    assert ticket["decision"]["department_confidence"] == 0.9
    assert ticket["decider_revision"] == "scripted"
    assert reopened.ticket(999) is None and reopened.ticket(1004) is None


def test_a_ticket_keeps_the_location_but_masks_a_number_in_it(ticket_store) -> None:
    with_id = dataclasses.replace(ROADS_IN_W04, contains_personal_id=0.9)
    number = ticket_store.file(
        request(location_text="ಗಾಂಧಿನಗರ ನನ್ನ ನಂಬರ್ 1234567890", decision=with_id)
    )
    assert ticket_store.ticket(int(number))["location_text"] == "ಗಾಂಧಿನಗರ ನನ್ನ ನಂಬರ್ XXXXXXXXXX"


def test_transcripts_are_masked_and_deleted_after_the_retention_period(ticket_store) -> None:
    ticket_store.record_turn("call-1", 1, "ರೋಡ್ ಹಾಳಾಗಿದೆ", 0.1)
    ticket_store.record_turn("call-1", 2, "ನನ್ನ ನಂಬರ್ 9876543210", 0.1)
    assert ticket_store.transcript("call-1") == ["ರೋಡ್ ಹಾಳಾಗಿದೆ", "ನನ್ನ ನಂಬರ್ XXXXXXXXXX"]

    assert ticket_store.purge_transcripts(7) == 0  # nothing is old yet
    eight_days_on = datetime.now(UTC) + timedelta(days=8)
    assert ticket_store.purge_transcripts(7, at=eight_days_on) == 2
    assert ticket_store.transcript("call-1") == []
    ticket_store.file(request())
    assert ticket_store.purge_transcripts(7, at=eight_days_on) == 0
    assert len(ticket_store.tickets()) == 1  # tickets are kept


CONSOLE = {"Authorization": "Bearer console-test-token"}


@pytest.fixture
def console_token(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = isolated_settings(console_token="console-test-token")
    monkeypatch.setattr(app_module, "get_settings", lambda: settings)


def test_the_console_api_lists_and_finds_tickets(ticket_store, console_token) -> None:
    ticket_store.file(request())
    ticket_store.file(request(department="water"))
    with TestClient(app_module.app, headers=CONSOLE) as client:
        listed = client.get("/api/tickets", params={"limit": 1}).json()
        assert [t["number"] for t in listed] == ["1002"]
        assert client.get("/api/tickets/1001").json()["department"] == "roads"
        assert client.get("/api/tickets/1999").status_code == 404
        for junk in ("abc", "99999999999999999999999", "5"):
            assert client.get(f"/api/tickets/{junk}").status_code == 422
        assert client.get("/api/tickets", params={"limit": 0}).status_code == 422


def test_the_retention_sweep_runs_when_the_app_starts(ticket_store, monkeypatch) -> None:
    ticket_store.record_turn("old-call", 1, "ರೋಡ್", 0.1)
    monkeypatch.setattr(
        ticket_store,
        "purge_transcripts",
        lambda days: TicketStore.purge_transcripts(
            ticket_store, days, at=datetime.now(UTC) + timedelta(days=days + 1)
        ),
    )
    with TestClient(app_module.app):
        pass
    assert ticket_store.transcript("old-call") == []


def test_the_dialogue_files_the_callers_location_words_with_the_ticket() -> None:
    near_a_shop = dataclasses.replace(ROADS, has_location=0.9)
    desk = CountingDesk()
    call = Dialogue(load_templates(), ScriptedDecider(ROADS, near_a_shop, ROADS_IN_W04), desk)

    call.turn("ರೋಡ್ ತುಂಬಾ ಹಾಳಾಗಿದೆ")
    call.turn("ಮೆಡಿಕಲ್ ಶಾಪ್ ಎದುರು")
    reply = call.turn("ಗಾಂಧಿನಗರ")

    assert reply.ticket is not None and reply.ticket.number == "1001"
    (filed,) = desk.filed
    # The turn that gave the place and the turn that named the ward, not the complaint itself.
    assert filed.location_text == "ಮೆಡಿಕಲ್ ಶಾಪ್ ಎದುರು ಗಾಂಧಿನಗರ"
    assert (filed.call_id, filed.decider_revision) == (call.call_id, "scripted")
    assert filed.decision == ROADS_IN_W04


def test_no_ticket_is_filed_for_a_hand_off_or_a_redirect() -> None:
    desk = CountingDesk()
    for decision in (
        dataclasses.replace(NOTHING, wants_human=0.9),
        dataclasses.replace(NOTHING, out_of_scope=0.9),
    ):
        Dialogue(load_templates(), ScriptedDecider(decision), desk).turn("...")
    assert desk.filed == []


def test_a_number_spread_over_two_turns_is_masked_in_both(ticket_store) -> None:
    # The first half looks harmless on its own; the second turn makes it a phone number.
    ticket_store.record_turn("call-1", 1, "ನನ್ನ ನಂಬರ್ 98450", 0.1)
    assert ticket_store.transcript("call-1") == ["ನನ್ನ ನಂಬರ್ 98450"]
    ticket_store.record_turn("call-1", 2, "12345 ಗೆ ಕಾಲ್ ಮಾಡಿ", 0.1)
    assert ticket_store.transcript("call-1") == ["ನನ್ನ ನಂಬರ್ XXXXX", "XXXXX ಗೆ ಕಾಲ್ ಮಾಡಿ"]


def test_a_later_personal_id_decision_masks_earlier_turns(ticket_store) -> None:
    ticket_store.record_turn("call-1", 1, "ಖಾತೆ ಸಂಖ್ಯೆ 4521", 0.1)
    ticket_store.record_turn("call-1", 2, "ಅದೇ ನನ್ನ ಅಕೌಂಟ್", 0.9)
    ticket_store.record_turn("call-1", 3, "ರೇಷನ್ ಕಾರ್ಡ್ 7788", 0.1)  # still counts as flagged
    assert ticket_store.transcript("call-1") == [
        "ಖಾತೆ ಸಂಖ್ಯೆ XXXX",
        "ಅದೇ ನನ್ನ ಅಕೌಂಟ್",
        "ರೇಷನ್ ಕಾರ್ಡ್ XXXX",
    ]
    ticket_store.record_turn("call-2", 1, "ಟಿಕೆಟ್ 4521", 0.1)  # other calls are unaffected
    assert ticket_store.transcript("call-2") == ["ಟಿಕೆಟ್ 4521"]


def test_a_failed_retention_pass_does_not_stop_later_ones(ticket_store, monkeypatch) -> None:
    import asyncio

    passes = []

    def flaky(days: int) -> int:
        passes.append(days)
        if len(passes) == 1:
            raise RuntimeError("database is locked")
        return 0

    monkeypatch.setattr(ticket_store, "purge_transcripts", flaky)
    monkeypatch.setattr(app_module, "RETENTION_SWEEP_S", 0.01)

    async def run_briefly() -> None:
        sweeper = asyncio.create_task(app_module._sweep_transcripts())
        await asyncio.sleep(0.1)
        sweeper.cancel()
        await asyncio.gather(sweeper, return_exceptions=True)

    asyncio.run(run_briefly())
    assert len(passes) >= 2


@pytest.mark.parametrize(
    "headers",
    [
        {},
        {"Authorization": "Bearer wrong"},
        {"Authorization": "console-test-token"},
        {"Authorization": "Basic console-test-token"},
    ],
)
def test_the_console_api_needs_the_console_token(ticket_store, console_token, headers) -> None:
    ticket_store.file(request())
    with TestClient(app_module.app, headers=headers) as client:
        for route in ("/api/tickets", "/api/tickets/1001", "/api/thresholds"):
            refused = client.get(route)
            assert refused.status_code == 401
            assert refused.headers["www-authenticate"] == "Bearer"
            assert "roads" not in refused.text


def test_the_console_api_is_off_without_a_token(ticket_store, monkeypatch) -> None:
    monkeypatch.setattr(app_module, "get_settings", lambda: isolated_settings())
    with TestClient(app_module.app, headers={"Authorization": "Bearer "}) as client:
        assert client.get("/api/tickets").status_code == 503
