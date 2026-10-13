import asyncio
from types import TracebackType
from typing import Self

import pytest
from fastapi.testclient import TestClient

import app as app_module
from config import Settings
from gnani.prisma import SttEvent


def isolated_settings(**overrides: object) -> Settings:
    """Settings that ignore the developer's local .env."""
    return Settings(_env_file=None, **overrides)


class FakeStream:
    def __init__(self) -> None:
        self.audio = bytearray()
        self.flushed = False
        self._events: asyncio.Queue[SttEvent] = asyncio.Queue()

    async def __aenter__(self) -> Self:
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
        self.audio.extend(pcm)
        self._events.put_nowait(SttEvent(type="transcript", text="ರಸ್ತೆ", latency_ms=120))

    async def flush(self) -> None:
        self.flushed = True

    async def events(self):
        while True:
            yield await self._events.get()


@pytest.fixture
def fake_stream(monkeypatch: pytest.MonkeyPatch) -> FakeStream:
    stream = FakeStream()
    monkeypatch.setattr(app_module, "get_settings", lambda: isolated_settings(gnani_api_key="test"))
    monkeypatch.setattr(app_module, "stt_stream_factory", lambda settings: stream)
    return stream


def test_proxy_relays_audio_up_and_transcripts_down(fake_stream: FakeStream) -> None:
    client = TestClient(app_module.app)

    with client.websocket_connect("/ws/stt") as ws:
        assert ws.receive_json() == {"type": "config", "sample_rate": 8000}
        assert ws.receive_json() == {"type": "connected"}
        ws.send_bytes(b"\x00\x01" * 256)
        ws.send_text("flush")
        transcript = ws.receive_json()

    assert transcript["type"] == "transcript"
    assert transcript["text"] == "ರಸ್ತೆ"
    assert transcript["latency_ms"] == 120
    assert bytes(fake_stream.audio) == b"\x00\x01" * 256


def test_proxy_reports_missing_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(app_module, "get_settings", lambda: isolated_settings(gnani_api_key=""))
    client = TestClient(app_module.app)

    with client.websocket_connect("/ws/stt") as ws:
        message = ws.receive_json()

    assert message == {"type": "error", "message": "GNANI_API_KEY is not set"}


def test_settings_parse_sample_rate_from_env_text(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GNANI_STT_SAMPLE_RATE", "16000")

    assert isolated_settings().gnani_stt_sample_rate == 16000


def test_settings_reject_non_telephony_sample_rate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GNANI_STT_SAMPLE_RATE", "44100")

    with pytest.raises(ValueError):
        isolated_settings()
