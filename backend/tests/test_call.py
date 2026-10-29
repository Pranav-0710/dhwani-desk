import asyncio
import dataclasses
import threading
import time
from collections.abc import AsyncIterator

import pytest
from fakes import ROADS_IN_W04, FakeStream, ScriptedBrowser, ScriptedDecider, use_fakes

import app as app_module
from audio.speaker import Speaker
from dialogue.decision import Decision, Screening
from gnani.prisma import SttEvent
from gnani.timbre import TimbreError
from tickets.store import TicketStore


class FakeTts:
    """Stands in for TimbreClient: each line becomes two PCM chunks, or fails on request."""

    def __init__(self, fail_on: str | None = None) -> None:
        self.fail_on = fail_on
        self.spoken: list[str] = []
        self.closed = False

    async def __aenter__(self) -> "FakeTts":
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        self.closed = True

    async def stream(self, text: str, voice: str | None = None) -> AsyncIterator[bytes]:
        if text == self.fail_on:
            raise TimbreError(503, "SERVICE_UNAVAILABLE", "busy")
        self.spoken.append(text)
        yield b"\x01\x00" * 4
        yield b"\x02\x00" * 4


async def wait_for(condition, timeout: float = 5.0) -> None:
    async with asyncio.timeout(timeout):
        while not condition():
            await asyncio.sleep(0.01)


class RecordingOutbox:
    def __init__(self) -> None:
        self.sent: list[dict | bytes] = []

    async def json(self, payload: dict) -> None:
        self.sent.append(payload)

    async def audio(self, pcm: bytes) -> None:
        self.sent.append(pcm)


def test_speaker_frames_each_line_and_survives_a_tts_failure() -> None:
    outbox = RecordingOutbox()

    async def scenario() -> None:
        speaker = Speaker(FakeTts(fail_on="broken"), outbox, sample_rate=16000)
        speaker.say("broken")
        speaker.say("ಸರಿ")
        await wait_for(lambda: outbox.sent.count({"type": "agent_audio_end"}) == 2)
        await speaker.aclose()

    asyncio.run(scenario())

    assert outbox.sent == [
        {"type": "agent_text", "text": "broken"},
        {"type": "agent_audio_start", "sample_rate": 16000},
        {"type": "tts_error", "message": "Timbre SERVICE_UNAVAILABLE: busy (HTTP 503)"},
        {"type": "agent_audio_end"},
        {"type": "agent_text", "text": "ಸರಿ"},
        {"type": "agent_audio_start", "sample_rate": 16000},
        b"\x01\x00" * 4,
        b"\x02\x00" * 4,
        {"type": "agent_audio_end"},
    ]


def test_speaker_close_stops_mid_line_without_sending_more() -> None:
    outbox = RecordingOutbox()

    class SlowTts(FakeTts):
        async def stream(self, text: str, voice: str | None = None) -> AsyncIterator[bytes]:
            yield b"\x01\x00"
            await asyncio.sleep(3600)
            yield b"never"

    async def scenario() -> None:
        speaker = Speaker(SlowTts(), outbox, sample_rate=8000)
        speaker.say("ನಮಸ್ಕಾರ")
        await wait_for(lambda: b"\x01\x00" in outbox.sent)
        await speaker.aclose()

    asyncio.run(scenario())

    assert outbox.sent[-1] == b"\x01\x00"
    assert {"type": "agent_audio_end"} not in outbox.sent


def lines_of(sent: list[dict | bytes], kind: str, field: str) -> list:
    return [m[field] for m in sent if isinstance(m, dict) and m["type"] == kind]


def use_dialogue_fakes(monkeypatch: pytest.MonkeyPatch, tts: FakeTts, decider=None) -> None:
    use_fakes(monkeypatch, FakeStream(), tts)
    monkeypatch.setattr(app_module, "turn_pause_s", 0.02)
    monkeypatch.setattr(app_module, "TTS_CACHE", {})
    monkeypatch.setattr(app_module, "HANG_UP_GRACE_S", 0.01)
    if decider is not None:
        monkeypatch.setattr(app_module, "get_decider", lambda: decider)


def ended_lines(count: int):
    return lambda sent: sent.count({"type": "agent_audio_end"}) >= count


def test_call_greets_on_connect_then_answers_the_callers_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tts = FakeTts()
    use_dialogue_fakes(monkeypatch, tts)
    browser = ScriptedBrowser([bytes(1024)], leave_when=ended_lines(2))

    asyncio.run(app_module.call(browser))

    greeting, ask_problem = (app_module.TEMPLATES.lines[k] for k in ("greeting", "ask_problem"))
    assert browser.sent[:2] == [
        {"type": "config", "sample_rate": 8000, "tts_sample_rate": 16000},
        {"type": "connected"},
    ]
    # "ರಸ್ತೆ" alone names no problem the keyword stand-in knows, so the agent asks again.
    assert lines_of(browser.sent, "dialogue", "keys") == [["greeting"], ["ask_problem"]]
    assert lines_of(browser.sent, "transcript", "text") == ["ರಸ್ತೆ"]
    assert lines_of(browser.sent, "agent_text", "text") == [greeting, ask_problem]
    assert tts.spoken == [greeting, ask_problem]
    assert tts.closed


def test_a_whole_call_ends_with_a_ticket(
    monkeypatch: pytest.MonkeyPatch, ticket_store: TicketStore
) -> None:
    use_dialogue_fakes(monkeypatch, FakeTts(), ScriptedDecider(ROADS_IN_W04))
    # The caller stays on the line until the agent hangs up, so the hang-up is what is tested.
    browser = ScriptedBrowser([bytes(1024)], leave_when=lambda sent: False)

    asyncio.run(app_module.call(browser))

    closing = [m for m in browser.sent if isinstance(m, dict) and m["type"] == "dialogue"][-1]
    assert closing["keys"] == ["read_back_ward", "closing"]
    assert (closing["state"], closing["ticket"]["ward"]) == ("closed", "w04")
    # The agent hangs up once its last line has been heard.
    assert browser.sent[-1] == {"type": "call_ended", "reason": "ticket"}
    assert browser.closed_with == 1000
    # The ticket and the caller's turn are stored, the number matches the one read back.
    (ticket,) = ticket_store.tickets()
    assert ticket["number"] == closing["ticket"]["number"]
    assert (ticket["department"], ticket["ward"]) == ("roads", "w04")
    assert ticket["decider_revision"] == "scripted"
    assert ticket_store.transcript(ticket["call_id"]) == ["ರಸ್ತೆ"]


def test_a_late_prisma_event_during_the_hang_up_is_not_sent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stream = FakeStream()
    use_dialogue_fakes(monkeypatch, FakeTts(), ScriptedDecider(ROADS_IN_W04))
    monkeypatch.setattr(app_module, "stt_stream_factory", lambda settings: stream)

    def late_event_on_hang_up(sent: list) -> bool:
        if {"type": "call_ended", "reason": "ticket"} in sent:
            stream._events.put_nowait(SttEvent(type="transcript", text="ಹಲೋ"))
        return False  # never leaves: the server hangs up

    browser = ScriptedBrowser([bytes(1024)], leave_when=late_event_on_hang_up)
    asyncio.run(app_module.call(browser))  # ScriptedBrowser raises on a send after close

    assert browser.sent[-1] == {"type": "call_ended", "reason": "ticket"}
    assert browser.closed_with == 1000


def test_a_failing_decider_hands_the_call_off(monkeypatch: pytest.MonkeyPatch) -> None:
    use_dialogue_fakes(monkeypatch, FakeTts(), ScriptedDecider(RuntimeError("model crashed")))
    browser = ScriptedBrowser([bytes(1024)], leave_when=ended_lines(2))

    asyncio.run(app_module.call(browser))

    assert lines_of(browser.sent, "dialogue", "keys") == [["greeting"], ["handoff_unclear"]]
    assert lines_of(browser.sent, "dialogue", "reason")[-1] == "decider_error"


def test_the_agent_says_one_moment_while_a_slow_decision_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class SlowDecider(ScriptedDecider):
        def decide(self, transcript: str, screening: Screening) -> Decision:
            time.sleep(0.2)
            return super().decide(transcript, screening)

    use_dialogue_fakes(monkeypatch, FakeTts(), SlowDecider(ROADS_IN_W04))
    monkeypatch.setattr(app_module, "hold_on_after_s", 0.02)
    browser = ScriptedBrowser([bytes(1024)], leave_when=lambda sent: False)  # agent hangs up

    asyncio.run(app_module.call(browser))

    lines = app_module.TEMPLATES.lines
    assert lines_of(browser.sent, "agent_text", "text")[:3] == [
        lines["greeting"],
        lines["hold_on"],  # before the answer, while the decision runs
        app_module.TEMPLATES.line(
            "read_back_ward", ward="ಗಾಂಧಿನಗರ", department="ರಸ್ತೆ", ticket="1 0 0 1"
        ),
    ]


def test_a_safety_risk_is_handed_off_before_the_rest_is_decided(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    danger = dataclasses.replace(ROADS_IN_W04, safety_hazard=0.9)
    decider = ScriptedDecider(danger)
    use_dialogue_fakes(monkeypatch, FakeTts(), decider)
    monkeypatch.setattr(app_module, "hold_on_after_s", 0.0)

    def decided_in_full(sent: list) -> bool:
        return [] in lines_of(sent, "dialogue", "keys")

    browser = ScriptedBrowser([bytes(1024)], leave_when=decided_in_full)
    asyncio.run(app_module.call(browser))

    hand_off, update = [m for m in browser.sent if isinstance(m, dict) and m["type"] == "dialogue"][
        1:
    ]
    # Said on the safety answers alone, with no wait for the slow step and no "one moment".
    assert (hand_off["keys"], hand_off["reason"], hand_off["decision"]) == (
        ["safety"],
        "safety",
        None,
    )
    assert hand_off["screening"] == {
        "safety_hazard": 0.9,
        "urgency": "soon",
        "wants_human": 0.1,
        "out_of_scope": 0.1,
    }
    assert app_module.TEMPLATES.lines["hold_on"] not in lines_of(browser.sent, "agent_text", "text")
    # Then the rest, for the officer: nothing more is said.
    assert (update["state"], update["reason"]) == ("handed_off", "safety")
    assert update["decision"]["department"] == "roads"
    assert update["slots"]["ward"] == "w04"
    assert decider.decided == ["ರಸ್ತೆ"]


def test_the_turn_behind_a_safety_hand_off_is_kept_if_the_caller_hangs_up(
    monkeypatch: pytest.MonkeyPatch, ticket_store: TicketStore
) -> None:
    release = threading.Event()

    class SlowDecider(ScriptedDecider):
        def decide(self, transcript: str, screening: Screening) -> Decision:
            release.wait(5)
            return super().decide(transcript, screening)

    recorded = []
    store_turn = ticket_store.record_turn
    monkeypatch.setattr(
        ticket_store, "record_turn", lambda *turn: (recorded.append(turn), store_turn(*turn))
    )
    danger = dataclasses.replace(ROADS_IN_W04, safety_hazard=0.9)
    use_dialogue_fakes(monkeypatch, FakeTts(), SlowDecider(danger))
    # The caller leaves once told it is dangerous, while the rest is still being decided.
    browser = ScriptedBrowser([bytes(1024)], leave_when=ended_lines(2))

    async def scenario() -> None:
        await app_module.call(browser)
        release.set()
        await asyncio.sleep(0.1)  # let the decision thread finish

    asyncio.run(scenario())

    assert lines_of(browser.sent, "dialogue", "keys") == [["greeting"], ["safety"]]
    ((call_id, _, text, personal_id),) = recorded
    assert (text, personal_id) == ("ರಸ್ತೆ", 1.0)  # masked as if it held a personal number
    assert ticket_store.transcript(call_id) == ["ರಸ್ತೆ"]


@pytest.mark.parametrize("danger", [False, True], ids=["after_hold_on", "after_hand_off"])
def test_a_failing_second_step_hands_off_once(
    monkeypatch: pytest.MonkeyPatch, danger: bool
) -> None:
    class FailsAfterScreening(ScriptedDecider):
        def decide(self, transcript: str, screening: Screening) -> Decision:
            time.sleep(0.05)
            raise RuntimeError("model crashed")

    decision = dataclasses.replace(ROADS_IN_W04, safety_hazard=0.9 if danger else 0.1)
    use_dialogue_fakes(monkeypatch, FakeTts(), FailsAfterScreening(decision))
    monkeypatch.setattr(app_module, "hold_on_after_s", 0.01)
    # Greeting, then "one moment" and the hand-off, or the safety line alone.
    browser = ScriptedBrowser([bytes(1024)], leave_when=ended_lines(2 if danger else 3))

    asyncio.run(app_module.call(browser))

    expected = "safety" if danger else "handoff_unclear"
    assert lines_of(browser.sent, "dialogue", "keys") == [["greeting"], [expected]]


def test_a_decision_still_running_when_the_call_ends_files_nothing(
    monkeypatch: pytest.MonkeyPatch, ticket_store: TicketStore
) -> None:
    started, release = threading.Event(), threading.Event()

    class SlowDecider(ScriptedDecider):
        def decide(self, transcript: str, screening: Screening) -> Decision:
            started.set()
            release.wait(5)
            return super().decide(transcript, screening)

    use_dialogue_fakes(monkeypatch, FakeTts(), SlowDecider(ROADS_IN_W04))
    browser = ScriptedBrowser(
        [bytes(1024)],
        leave_when=lambda sent: started.is_set(),  # leave mid-decision
    )

    async def scenario() -> None:
        await app_module.call(browser)  # the browser leaves while the decision is running
        release.set()
        await asyncio.sleep(0.1)  # let the decision thread finish

    asyncio.run(scenario())

    assert lines_of(browser.sent, "dialogue", "keys") == [["greeting"]]
    assert ticket_store.tickets() == []  # no ticket was filed for a call that had ended


def test_fixed_lines_are_synthesised_once_across_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    tts = FakeTts()
    use_dialogue_fakes(monkeypatch, tts)
    monkeypatch.setattr(app_module, "stt_stream_factory", lambda settings: FakeStream())

    async def two_calls() -> None:
        for _ in range(2):
            await app_module.call(ScriptedBrowser([bytes(1024)], leave_when=ended_lines(2)))

    asyncio.run(two_calls())

    greeting = app_module.TEMPLATES.lines["greeting"]
    assert tts.spoken.count(greeting) == 1


def test_unexpected_tts_failure_still_ends_the_line_and_keeps_going() -> None:
    outbox = RecordingOutbox()

    class BuggyTts(FakeTts):
        async def stream(self, text: str, voice: str | None = None) -> AsyncIterator[bytes]:
            if text == "bad":
                raise KeyError("surprise")
            async for pcm in super().stream(text, voice):
                yield pcm

    async def scenario() -> None:
        speaker = Speaker(BuggyTts(), outbox, sample_rate=16000)
        speaker.say("bad")
        speaker.say("ಸರಿ")
        await wait_for(lambda: outbox.sent.count({"type": "agent_audio_end"}) == 2)
        await speaker.aclose()

    asyncio.run(scenario())

    assert outbox.sent[:4] == [
        {"type": "agent_text", "text": "bad"},
        {"type": "agent_audio_start", "sample_rate": 16000},
        {"type": "tts_error", "message": "KeyError"},
        {"type": "agent_audio_end"},
    ]
    assert outbox.sent[4] == {"type": "agent_text", "text": "ಸರಿ"}


def test_call_reports_tts_error_and_keeps_the_call_open(monkeypatch: pytest.MonkeyPatch) -> None:
    greeting = app_module.TEMPLATES.lines["greeting"]
    tts = FakeTts(fail_on=greeting)
    use_dialogue_fakes(monkeypatch, tts)
    browser = ScriptedBrowser([bytes(1024)], leave_when=ended_lines(2))

    asyncio.run(app_module.call(browser))

    assert {
        "type": "tts_error",
        "message": "Timbre SERVICE_UNAVAILABLE: busy (HTTP 503)",
    } in browser.sent
    # The failed greeting is not cached, and the call still answers the caller's turn.
    assert tts.spoken == [app_module.TEMPLATES.lines["ask_problem"]]
    assert browser.closed_with is None  # the browser left; the server never closed the call


def test_speaker_waits_until_the_browser_has_played_the_audio() -> None:
    outbox = RecordingOutbox()

    class LongTts(FakeTts):
        async def stream(self, text: str, voice: str | None = None) -> AsyncIterator[bytes]:
            yield bytes(9600)  # 0.3 s of 16-bit mono at 16 kHz, sent at once
            yield bytes(9600)

    async def scenario() -> float:
        speaker = Speaker(LongTts(), outbox, sample_rate=16000)
        loop = asyncio.get_running_loop()
        started = loop.time()
        speaker.say("ಸರಿ")
        await speaker.wait_until_heard()
        waited = loop.time() - started
        await speaker.aclose()
        return waited

    waited = asyncio.run(scenario())
    # Sent in an instant, but the browser needs 0.6 s to play it.
    assert 0.55 <= waited < 1.0
