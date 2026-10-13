import asyncio
import base64
import json

import httpx2
import pytest

from config import Settings
from gnani.timbre import TimbreClient, TimbreConfig, TimbreError


def sse(*messages: tuple[str | None, dict]) -> bytes:
    lines = []
    for event, data in messages:
        if event:
            lines.append(f"event: {event}")
        lines.append(f"data: {json.dumps(data)}")
        lines.append("")
    return ("\n".join(lines) + "\n").encode()


def run_with(handler, scenario, **config_overrides):
    """Run `scenario(client)` against a mocked Timbre API; returns its result."""
    config = TimbreConfig(api_key="k", base_url="https://tts.test/api/v1/tts", **config_overrides)

    async def go():
        async with TimbreClient(config, transport=httpx2.MockTransport(handler)) as client:
            return await scenario(client)

    return asyncio.run(go())


def test_request_body_asks_for_raw_16_bit_mono_pcm() -> None:
    body = TimbreConfig(api_key="k", voice="Saanvi", sample_rate=8000).request_body("ನಮಸ್ಕಾರ")

    assert body == {
        "text": "ನಮಸ್ಕಾರ",
        "model": "timbre-v2.5",
        "voice": "Saanvi",
        "language": "kn-IN",
        "speed": 1.0,
        "audio_config": {
            "sample_rate": 8000,
            "num_channels": 1,
            "sample_width": 2,
            "encoding": "linear_pcm",
            "container": "raw",
        },
    }


def test_request_body_rejects_unsupported_sample_rate() -> None:
    with pytest.raises(ValueError):
        TimbreConfig(api_key="k", sample_rate=11025).request_body("x")


def test_config_repr_never_shows_the_api_key() -> None:
    assert "SENTINEL-KEY" not in repr(TimbreConfig(api_key="SENTINEL-KEY"))


def test_synthesize_posts_to_inference_and_returns_pcm() -> None:
    seen: dict = {}

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen["url"] = str(request.url)
        seen["key"] = request.headers["X-API-Key-ID"]
        seen["body"] = json.loads(request.content)
        return httpx2.Response(200, content=b"\x01\x02" * 10)

    pcm = run_with(handler, lambda client: client.synthesize("ನಮಸ್ಕಾರ", voice="Kavin"))

    assert pcm == b"\x01\x02" * 10
    assert seen["url"] == "https://tts.test/api/v1/tts/inference"
    assert seen["key"] == "k"
    assert seen["body"]["voice"] == "Kavin"


def test_synthesize_raises_timbre_error_with_api_details() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        error = {"type": "INVALID_REQUEST_ERROR", "message": "unsupported voice"}
        return httpx2.Response(400, json={"success": False, "error": error})

    with pytest.raises(TimbreError) as caught:
        run_with(handler, lambda client: client.synthesize("x", voice="Nobody"))

    assert caught.value.status == 400
    assert caught.value.error_type == "INVALID_REQUEST_ERROR"
    assert "unsupported voice" in str(caught.value)


def test_network_failure_becomes_timbre_error() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError("boom", request=request)

    with pytest.raises(TimbreError) as caught:
        run_with(handler, lambda client: client.synthesize("x"))

    assert caught.value.error_type == "NETWORK_ERROR"


def chunk(index: int, audio: bytes) -> dict:
    return {"chunk_index": index, "audio": base64.b64encode(audio).decode(), "is_final": False}


def test_stream_decodes_sse_chunks_in_order() -> None:
    body = sse(
        ("start", {"status": "streaming_started", "text": "x"}),
        ("chunk", chunk(0, b"AAAA")),
        ("chunk", chunk(1, b"BB")),
        ("complete", {"chunk_index": 2, "audio": "", "is_final": True}),
    )
    seen: dict = {}

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen["url"] = str(request.url)
        return httpx2.Response(200, content=body, headers={"content-type": "text/event-stream"})

    async def collect(client: TimbreClient) -> list[bytes]:
        return [chunk async for chunk in client.stream("x")]

    assert run_with(handler, collect) == [b"AAAA", b"BB"]
    assert seen["url"] == "https://tts.test/api/v1/tts/sse"


def test_stream_accepts_unnamed_data_events() -> None:
    body = sse(
        (None, {"chunk_index": 0, "audio": base64.b64encode(b"CC").decode(), "is_final": False}),
        (None, {"chunk_index": 1, "audio": "", "is_final": True}),
    )

    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, content=body)

    async def collect(client: TimbreClient) -> list[bytes]:
        return [chunk async for chunk in client.stream("x")]

    assert run_with(handler, collect) == [b"CC"]


def test_stream_raises_on_error_event_and_http_error() -> None:
    error_event = sse(("error", {"error": {"type": "API_ERROR", "message": "engine down"}}))

    async def collect(client: TimbreClient) -> list[bytes]:
        return [chunk async for chunk in client.stream("x")]

    with pytest.raises(TimbreError, match="engine down"):
        run_with(lambda request: httpx2.Response(200, content=error_event), collect)

    forbidden = httpx2.Response(403, json={"error": {"type": "FORBIDDEN"}})
    with pytest.raises(TimbreError) as caught:
        run_with(lambda request: forbidden, collect)
    assert caught.value.status == 403


def test_settings_default_voice_when_env_value_is_blank(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GNANI_TTS_VOICE", "")

    assert Settings(_env_file=None).gnani_tts_voice == "Kavin"


def test_settings_reject_speed_outside_timbre_range(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GNANI_TTS_SPEED", "1.5")

    with pytest.raises(ValueError):
        Settings(_env_file=None)
