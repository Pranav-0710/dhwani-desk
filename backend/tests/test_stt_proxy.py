import asyncio

import pytest
from fakes import (
    PHONE_ORIGIN,
    FakeStream,
    ScriptedBrowser,
    isolated_settings,
    use_fakes,
    use_stream,
)
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

import app as app_module
from gnani.prisma import PrismaOverflow, PrismaRejected


def test_proxy_relays_audio_up_and_transcripts_down(monkeypatch: pytest.MonkeyPatch) -> None:
    stream = FakeStream()
    use_fakes(monkeypatch, stream)
    audio = bytes([0, 1]) * 256
    browser = ScriptedBrowser(
        [audio, "flush"], leave_when=lambda sent: any(m.get("type") == "transcript" for m in sent)
    )

    asyncio.run(app_module.stt_proxy(browser))

    assert browser.sent == [
        {"type": "config", "sample_rate": 8000},
        {"type": "connected"},
        {
            "type": "transcript",
            "text": "ರಸ್ತೆ",
            "segment_index": None,
            "audio_duration_ms": None,
            "latency_ms": 120,
        },
    ]
    assert bytes(stream.audio) == audio
    assert stream.flushed
    assert stream.discarded  # browser left: unsent audio is dropped, not drained


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
    client = TestClient(app_module.app, headers={"origin": PHONE_ORIGIN})

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
