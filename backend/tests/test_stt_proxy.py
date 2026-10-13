import asyncio
from types import TracebackType
from typing import Self

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

import app as app_module
from config import Settings
from gnani.prisma import PrismaOverflow, PrismaRejected, SttEvent


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
    return TestClient(app_module.app)


def test_proxy_relays_audio_up_and_transcripts_down(monkeypatch: pytest.MonkeyPatch) -> None:
    stream = FakeStream()
    client = use_stream(monkeypatch, stream)

    with client.websocket_connect("/ws/stt") as ws:
        assert ws.receive_json() == {"type": "config", "sample_rate": 8000}
        assert ws.receive_json() == {"type": "connected"}
        ws.send_bytes(b"\x00\x01" * 256)
        ws.send_text("flush")
        transcript = ws.receive_json()

    assert transcript == {
        "type": "transcript",
        "text": "ರಸ್ತೆ",
        "segment_index": None,
        "audio_duration_ms": None,
        "latency_ms": 120,
    }
    assert bytes(stream.audio) == b"\x00\x01" * 256
    assert stream.flushed


class FakeBrowser:
    """A browser socket as uvicorn presents it: audio, then a normal disconnect."""

    def __init__(self, frames: list[bytes]) -> None:
        self.incoming = [{"type": "websocket.receive", "bytes": f} for f in frames]
        self.incoming.append({"type": "websocket.disconnect", "code": 1001})
        self.sent: list[dict] = []

    async def accept(self) -> None:
        return None

    async def receive(self) -> dict:
        await asyncio.sleep(0.01)
        return self.incoming.pop(0)

    async def send_json(self, data: dict) -> None:
        self.sent.append(data)

    async def close(self, code: int = 1000) -> None:
        return None


def test_browser_leaving_drops_unsent_audio(monkeypatch: pytest.MonkeyPatch) -> None:
    stream = FakeStream()
    monkeypatch.setattr(app_module, "get_settings", lambda: isolated_settings(gnani_api_key="t"))
    monkeypatch.setattr(app_module, "stt_stream_factory", lambda settings: stream)
    browser = FakeBrowser([b"\x00" * 1024])

    asyncio.run(app_module.stt_proxy(browser))

    assert bytes(stream.audio) == b"\x00" * 1024
    assert stream.discarded
    assert browser.sent[0] == {"type": "config", "sample_rate": 8000}


def test_proxy_reports_clean_prisma_close(monkeypatch: pytest.MonkeyPatch) -> None:
    stream = FakeStream()
    stream.close_cleanly(1000, "session limit")  # queued before connect: thread-safe
    client = use_stream(monkeypatch, stream)

    with client.websocket_connect("/ws/stt") as ws:
        assert ws.receive_json()["type"] == "config"
        closed = ws.receive_json()
        with pytest.raises(WebSocketDisconnect) as disconnect:
            ws.receive_json()

    assert closed == {"type": "closed", "by": "prisma", "code": 1000, "reason": "session limit"}
    assert disconnect.value.code == 1000


def test_proxy_reports_rejected_key(monkeypatch: pytest.MonkeyPatch) -> None:
    client = use_stream(monkeypatch, FakeStream(enter_error=PrismaRejected(401)))

    with client.websocket_connect("/ws/stt") as ws:
        error = ws.receive_json()
        with pytest.raises(WebSocketDisconnect) as disconnect:
            ws.receive_json()

    assert error == {"type": "error", "message": "Prisma rejected the connection: HTTP 401"}
    assert disconnect.value.code == 1011


def test_proxy_reports_overflow(monkeypatch: pytest.MonkeyPatch) -> None:
    stream = FakeStream()
    stream.audio_error = PrismaOverflow()
    client = use_stream(monkeypatch, stream)

    with client.websocket_connect("/ws/stt") as ws:
        ws.receive_json()  # config
        ws.receive_json()  # connected
        ws.send_bytes(b"\x00" * 1024)
        error = ws.receive_json()

    assert error["type"] == "error"
    assert "real time" in error["message"]


def test_proxy_reports_missing_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(app_module, "get_settings", lambda: isolated_settings(gnani_api_key=""))
    client = TestClient(app_module.app)

    with client.websocket_connect("/ws/stt") as ws:
        message = ws.receive_json()
        with pytest.raises(WebSocketDisconnect) as disconnect:
            ws.receive_json()

    assert message == {"type": "error", "message": "GNANI_API_KEY is not set"}
    assert disconnect.value.code == 1011


def test_settings_parse_sample_rate_from_env_text(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GNANI_STT_SAMPLE_RATE", "16000")

    assert isolated_settings().gnani_stt_sample_rate == 16000


def test_settings_reject_non_telephony_sample_rate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GNANI_STT_SAMPLE_RATE", "44100")

    with pytest.raises(ValueError):
        isolated_settings()


def test_settings_reject_unsupported_bias_score(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GNANI_STT_BIAS_SCORE", "1.2")

    with pytest.raises(ValueError):
        isolated_settings()
