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
from gnani.prisma import RECOVERABLE_ERROR, PrismaOverflow, PrismaRejected, SttEvent


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

    # Prisma's own reason is logged, not sent to the browser.
    assert closed == {"type": "closed", "by": "prisma", "code": 1000}
    assert disconnect.value.code == 1000


def test_a_prisma_error_the_session_survives_reaches_the_browser_without_its_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stream = FakeStream()
    stream._events.put_nowait(SttEvent(type="error", message="account 42 has no credits left"))
    use_fakes(monkeypatch, stream)
    browser = ScriptedBrowser([], leave_when=lambda sent: len(sent) >= 3)

    asyncio.run(app_module.stt_proxy(browser))

    assert {"type": "error", "message": RECOVERABLE_ERROR} in browser.sent
    assert "credits" not in str(browser.sent)


def test_proxy_reports_rejected_key(monkeypatch: pytest.MonkeyPatch) -> None:
    client = use_stream(monkeypatch, FakeStream(enter_error=PrismaRejected(401)))

    with client.websocket_connect("/ws/stt") as ws:
        error = ws.receive_json()
        with pytest.raises(WebSocketDisconnect) as disconnect:
            ws.receive_json()

    # The vendor's reason is logged, not sent to the browser.
    assert error == app_module.STT_FAILED
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
    assert error == app_module.STT_FAILED


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
