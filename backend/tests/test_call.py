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
        while not (
            outbox.sent and outbox.sent[-1] == {"type": "agent_audio_end"} and len(outbox.sent) > 4
        ):
            await asyncio.sleep(0.01)
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
        while b"\x01\x00" not in outbox.sent:
            await asyncio.sleep(0.01)
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
