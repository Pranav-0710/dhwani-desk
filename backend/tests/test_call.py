import asyncio
from collections.abc import AsyncIterator

import pytest
from fakes import FakeStream, ScriptedBrowser, use_fakes

import app as app_module
from audio.speaker import Speaker
from dialogue.echo import ECHO_PREFIX, EchoAgent
from gnani.timbre import TimbreError


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


def test_echo_agent_repeats_the_transcript_and_ignores_blank() -> None:
    agent = EchoAgent()

    assert agent.reply(" ರಸ್ತೆ ") == f"{ECHO_PREFIX}ರಸ್ತೆ"
    assert agent.reply("   ") is None


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


def test_call_echoes_each_transcript_as_speech(monkeypatch: pytest.MonkeyPatch) -> None:
    tts = FakeTts()
    use_fakes(monkeypatch, FakeStream(), tts)
    browser = ScriptedBrowser(
        [bytes(1024)], leave_when=lambda sent: {"type": "agent_audio_end"} in sent
    )

    asyncio.run(app_module.call(browser))

    reply = f"{ECHO_PREFIX}ರಸ್ತೆ"
    assert browser.sent[:2] == [
        {"type": "config", "sample_rate": 8000, "tts_sample_rate": 16000},
        {"type": "connected"},
    ]
    assert browser.sent[2]["text"] == "ರಸ್ತೆ"
    assert browser.sent[3:] == [
        {"type": "agent_text", "text": reply},
        {"type": "agent_audio_start", "sample_rate": 16000},
        bytes([1, 0]) * 4,
        bytes([2, 0]) * 4,
        {"type": "agent_audio_end"},
    ]
    assert tts.spoken == [reply]
    assert tts.closed


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
    reply = f"{ECHO_PREFIX}ರಸ್ತೆ"
    use_fakes(monkeypatch, FakeStream(), FakeTts(fail_on=reply))
    browser = ScriptedBrowser(
        [bytes(1024)], leave_when=lambda sent: {"type": "agent_audio_end"} in sent
    )

    asyncio.run(app_module.call(browser))

    assert browser.sent[-3:] == [
        {"type": "agent_audio_start", "sample_rate": 16000},
        {"type": "tts_error", "message": "Timbre SERVICE_UNAVAILABLE: busy (HTTP 503)"},
        {"type": "agent_audio_end"},
    ]
    assert browser.closed_with is None  # the browser left; the server never closed the call
