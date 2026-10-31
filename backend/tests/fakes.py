"""Test doubles shared by the WebSocket route tests."""

import asyncio
import dataclasses
from types import TracebackType
from typing import Self

import pytest
from fastapi.testclient import TestClient

import app as app_module
from config import Settings
from dialogue.decision import Decision, Screening
from gnani.prisma import SttEvent

# A page the default ALLOWED_ORIGINS lets open the call sockets: the Vite dev server.
PHONE_ORIGIN = "http://localhost:5173"


def isolated_settings(**overrides: object) -> Settings:
    """Settings that ignore the developer's local .env."""
    return Settings(_env_file=None, **overrides)


class FakeStream:
    """Stands in for PrismaStream. Queue None into `_events` to simulate a clean close."""

    def __init__(self, enter_error: Exception | None = None) -> None:
        self.audio = bytearray()
        self.flushed = False
        self.discarded = False
        self.enter_error = enter_error
        self.audio_error: Exception | None = None
        self.close_code: int | None = None
        self.close_reason = ""
        self._events: asyncio.Queue[SttEvent | None] = asyncio.Queue()

    async def __aenter__(self) -> Self:
        if self.enter_error:
            raise self.enter_error
        self._events.put_nowait(SttEvent(type="connected"))
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        return None

    async def send_audio(self, pcm: bytes) -> None:
        if self.audio_error:
            raise self.audio_error
        self.audio.extend(pcm)
        self._events.put_nowait(SttEvent(type="transcript", text="ರಸ್ತೆ", latency_ms=120))

    async def flush(self) -> None:
        self.flushed = True

    def discard_pending(self) -> None:
        self.discarded = True

    def close_cleanly(self, code: int, reason: str) -> None:
        self.close_code, self.close_reason = code, reason
        self._events.put_nowait(None)

    async def events(self):
        while (event := await self._events.get()) is not None:
            yield event


def use_stream(monkeypatch: pytest.MonkeyPatch, stream: FakeStream) -> TestClient:
    monkeypatch.setattr(app_module, "get_settings", lambda: isolated_settings(gnani_api_key="t"))
    monkeypatch.setattr(app_module, "stt_stream_factory", lambda settings: stream)
    return TestClient(app_module.app, headers={"origin": PHONE_ORIGIN})


class ScriptedBrowser:
    """A browser socket as uvicorn presents it, for driving a route coroutine directly.

    Sends `frames` (bytes = audio, str = text frame), then disconnects normally once
    `leave_when(sent)` is true. Unlike TestClient, leaving never cancels the app mid-flight,
    so these tests are deterministic.
    """

    def __init__(
        self,
        frames: list[bytes | str],
        leave_when=lambda sent: True,
        origin: str | None = PHONE_ORIGIN,
    ) -> None:
        self.headers = {} if origin is None else {"origin": origin}
        self.frames = list(frames)
        self.leave_when = leave_when
        self.sent: list[dict | bytes] = []
        self.closed_with: int | None = None
        self.accepted = False

    async def accept(self) -> None:
        self.accepted = True

    async def receive(self) -> dict:
        await asyncio.sleep(0.01)
        if self.frames:
            frame = self.frames.pop(0)
            key = "bytes" if isinstance(frame, bytes) else "text"
            return {"type": "websocket.receive", key: frame}
        async with asyncio.timeout(5):
            while not self.leave_when(self.sent) and self.closed_with is None:
                await asyncio.sleep(0.01)
        return {"type": "websocket.disconnect", "code": self.closed_with or 1001}

    async def send_json(self, data: dict) -> None:
        self._check_open()
        self.sent.append(data)

    async def send_bytes(self, data: bytes) -> None:
        self._check_open()
        self.sent.append(data)

    def _check_open(self) -> None:
        # As Starlette does: sending on a socket the server has closed is an error.
        if self.closed_with is not None:
            raise RuntimeError("Cannot send once the socket is closed")

    async def close(self, code: int = 1000) -> None:
        self.closed_with = code


def use_fakes(monkeypatch: pytest.MonkeyPatch, stream: FakeStream, tts=None) -> None:
    """Point the app at fakes, for tests that drive route coroutines with ScriptedBrowser."""
    monkeypatch.setattr(app_module, "get_settings", lambda: isolated_settings(gnani_api_key="t"))
    monkeypatch.setattr(app_module, "stt_stream_factory", lambda settings: stream)
    if tts is not None:
        monkeypatch.setattr(app_module, "tts_client_factory", lambda settings: tts)


# A calm caller who has said nothing useful yet, and two decisions built from it.
NOTHING = Decision(
    department="other",
    department_confidence=0.2,
    ward="unknown",
    urgency="soon",
    safety_hazard=0.1,
    has_location=0.1,
    repeat_complaint=0.1,
    wants_human=0.1,
    caller_state="calm",
    out_of_scope=0.1,
    contains_personal_id=0.1,
)
ROADS = dataclasses.replace(NOTHING, department="roads", department_confidence=0.9)
ROADS_IN_W04 = dataclasses.replace(ROADS, ward="w04", has_location=0.9)


class ScriptedDecider:
    """Returns the queued decisions in order (raising any queued exception) and records what
    it was asked about. `screen` answers from the next decision; `decide` hands it over."""

    revision = "scripted"

    def __init__(self, *decisions: Decision | Exception) -> None:
        self.decisions = list(decisions)
        self.transcripts: list[str] = []
        self.decided: list[str] = []  # transcripts that got past the safety step

    def screen(self, transcript: str) -> Screening:
        self.transcripts.append(transcript)
        decision = self.decisions[0]
        if isinstance(decision, Exception):
            self.decisions.pop(0)
            raise decision
        return screening_of(decision)

    def decide(self, transcript: str, screening: Screening) -> Decision:
        self.decided.append(transcript)
        decision = self.decisions.pop(0)
        assert isinstance(decision, Decision) and screening == screening_of(decision)
        return decision


def screening_of(decision: Decision) -> Screening:
    return Screening(
        decision.safety_hazard, decision.urgency, decision.wants_human, decision.out_of_scope
    )
