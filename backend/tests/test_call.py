import asyncio
import threading
from collections.abc import AsyncIterator

import pytest
from fakes import ROADS_IN_W04, FakeStream, ScriptedBrowser, ScriptedDecider, use_fakes

import app as app_module
from audio.speaker import Speaker
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
        monkeypatch.setattr(app_module, "decider", decider)


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
    browser = ScriptedBrowser([bytes(1024)], leave_when=ended_lines(3))

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


def test_a_decision_still_running_when_the_call_ends_files_nothing(
    monkeypatch: pytest.MonkeyPatch, ticket_store: TicketStore
) -> None:
    started, release = threading.Event(), threading.Event()

    class SlowDecider:
        revision = "slow"

        def decide(self, transcript: str):
            started.set()
            release.wait(5)
            return ROADS_IN_W04

    use_dialogue_fakes(monkeypatch, FakeTts(), SlowDecider())
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
