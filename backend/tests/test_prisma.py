import asyncio
import json
import time
from collections.abc import Awaitable, Callable
from http import HTTPStatus

import pytest
from websockets.asyncio.server import ServerConnection, serve

from gnani.prisma import (
    FRAME_BYTES,
    PrismaClosed,
    PrismaConfig,
    PrismaOverflow,
    PrismaRejected,
    PrismaStream,
    PrismaUnavailable,
    load_bias_words,
    parse_event,
)

Handler = Callable[[ServerConnection], Awaitable[None]]


async def with_fake_prisma(handler: Handler, scenario, **serve_kwargs):
    """Run `scenario(config)` against a local fake Prisma server."""
    async with serve(handler, "127.0.0.1", 0, **serve_kwargs) as server:
        port = server.sockets[0].getsockname()[1]
        config = PrismaConfig(api_key="k", url=f"ws://127.0.0.1:{port}", sample_rate=8000)
        return await scenario(config)


def test_headers_carry_required_fields_and_ascii_bias_list() -> None:
    config = PrismaConfig(
        api_key="key-123",
        lang_code="kn-IN",
        sample_rate=8000,
        bias_words=("ವಾರ್ಡ್", "ಚರಂಡಿ"),
        bias_score=1.5,
    )

    headers = config.headers()

    assert headers["x-api-key-id"] == "key-123"
    assert headers["lang_code"] == "kn-IN"
    assert headers["x-sample-rate"] == "8000"
    assert headers["x-format"] == "transcribe"
    assert headers["x-bias-score"] == "1.5"
    assert headers["x-bias-list"].isascii()
    assert json.loads(headers["x-bias-list"]) == ["ವಾರ್ಡ್", "ಚರಂಡಿ"]


def test_headers_omit_bias_when_no_words() -> None:
    headers = PrismaConfig(api_key="k", bias_score=1.0).headers()

    assert "x-bias-list" not in headers
    assert "x-bias-score" not in headers


def test_config_repr_never_shows_the_api_key() -> None:
    assert "SENTINEL-KEY" not in repr(PrismaConfig(api_key="SENTINEL-KEY"))


def test_parse_transcript_event() -> None:
    event = parse_event(
        json.dumps(
            {
                "type": "transcript",
                "text": "ವಾರ್ಡ್ ಫೋರ್",
                "segment_index": 2,
                "audio_duration_ms": 1200,
                "latency": 244,
            }
        )
    )

    assert event is not None
    assert event.type == "transcript"
    assert event.to_client() == {
        "type": "transcript",
        "text": "ವಾರ್ಡ್ ಫೋರ್",
        "segment_index": 2,
        "audio_duration_ms": 1200,
        "latency_ms": 244,
    }


def test_parse_event_treats_null_text_as_empty() -> None:
    event = parse_event('{"type": "transcript", "text": null}')

    assert event is not None
    assert event.text == ""


def test_parse_event_ignores_unknown_and_invalid_frames() -> None:
    assert parse_event('{"type": "something_new"}') is None
    assert parse_event("not json") is None
    assert parse_event("[1, 2]") is None


def test_load_bias_words_reads_file_and_tolerates_missing(tmp_path) -> None:
    path = tmp_path / "words.json"
    path.write_text('["ಕಸ", "ನೀರು"]', encoding="utf-8")

    assert load_bias_words(path) == ("ಕಸ", "ನೀರು")
    assert load_bias_words(tmp_path / "missing.json") == ()


def test_load_bias_words_rejects_non_list(tmp_path) -> None:
    path = tmp_path / "words.json"
    path.write_text('"ಕಸ"', encoding="utf-8")

    with pytest.raises(ValueError):
        load_bias_words(path)


def test_stream_sends_exact_frames_at_real_time_pace_and_yields_events() -> None:
    received: list[bytes] = []
    request_headers: dict[str, str] = {}

    async def fake_prisma(connection: ServerConnection) -> None:
        request_headers.update(connection.request.headers)
        await connection.send(json.dumps({"type": "connected", "config": {"chunk_size": 512}}))
        async for message in connection:
            received.append(message)
            if len(received) == 3:
                await connection.send(json.dumps({"type": "processing"}))
                await connection.send(json.dumps({"type": "transcript", "text": "ಕಸ"}))

    async def scenario(config: PrismaConfig) -> tuple[list[str], float]:
        started = time.monotonic()
        async with PrismaStream(config) as stream:
            await stream.send_audio(b"\x01" * 2500)
            await stream.flush()
            types = []
            async for event in stream.events():
                types.append(event.type)
                if event.type == "transcript":
                    break
        return types, time.monotonic() - started

    types, elapsed = asyncio.run(with_fake_prisma(fake_prisma, scenario))

    assert types == ["connected", "processing", "transcript"]
    assert [len(frame) for frame in received] == [FRAME_BYTES] * 3
    tail = 2500 - 2 * FRAME_BYTES
    assert received[-1] == b"\x01" * tail + b"\x00" * (FRAME_BYTES - tail)
    assert request_headers["x-api-key-id"] == "k"
    # 3 frames at 64 ms each (8 kHz): the last one cannot leave before ~128 ms.
    assert elapsed >= 0.12


def test_backlog_drains_faster_than_real_time() -> None:
    arrivals: list[float] = []

    async def fake_prisma(connection: ServerConnection) -> None:
        async for _ in connection:
            arrivals.append(time.monotonic())

    async def scenario(config: PrismaConfig) -> None:
        async with PrismaStream(config) as stream:
            # 20 frames arrive at once, as after a handshake delay or a network stall.
            await stream.send_audio(b"\x00" * FRAME_BYTES * 20)
            while len(arrivals) < 20:
                await asyncio.sleep(0.01)

    asyncio.run(with_fake_prisma(fake_prisma, scenario))

    real_time = 19 * 0.064
    assert arrivals[-1] - arrivals[0] < real_time * 0.9


def test_clean_close_ends_events_and_exposes_code() -> None:
    async def fake_prisma(connection: ServerConnection) -> None:
        await connection.send(json.dumps({"type": "connected"}))
        await connection.recv()
        await connection.close(1000, "session limit")

    async def scenario(config: PrismaConfig):
        async with PrismaStream(config) as stream:
            await stream.send_audio(b"\x00" * FRAME_BYTES * 3)
            types = [event.type async for event in stream.events()]
            sender = stream._sender
        assert sender is not None and sender.done()
        return types, stream.close_code, stream.close_reason, sender.exception()

    types, code, reason, sender_error = asyncio.run(with_fake_prisma(fake_prisma, scenario))

    assert types == ["connected"]
    assert (code, reason) == (1000, "session limit")
    assert sender_error is None


def test_abnormal_close_raises_prisma_closed() -> None:
    async def fake_prisma(connection: ServerConnection) -> None:
        await connection.send(json.dumps({"type": "connected"}))
        await connection.close(1011, "engine failure")

    async def scenario(config: PrismaConfig) -> None:
        async with PrismaStream(config) as stream:
            async for _ in stream.events():
                pass

    with pytest.raises(PrismaClosed) as caught:
        asyncio.run(with_fake_prisma(fake_prisma, scenario))

    assert caught.value.code == 1011


def test_rejected_handshake_raises_prisma_rejected() -> None:
    async def fake_prisma(connection: ServerConnection) -> None:
        return None

    def reject(connection, request):
        return connection.respond(HTTPStatus.UNAUTHORIZED, "bad key\n")

    async def scenario(config: PrismaConfig) -> None:
        async with PrismaStream(config):
            pass

    with pytest.raises(PrismaRejected) as caught:
        asyncio.run(with_fake_prisma(fake_prisma, scenario, process_request=reject))

    assert caught.value.status == 401


def test_unreachable_server_raises_prisma_unavailable() -> None:
    async def scenario() -> None:
        async with PrismaStream(PrismaConfig(api_key="k", url="ws://127.0.0.1:1")):
            pass

    with pytest.raises(PrismaUnavailable):
        asyncio.run(scenario())


def test_faster_than_real_time_client_gets_overflow() -> None:
    async def fake_prisma(connection: ServerConnection) -> None:
        async for _ in connection:
            pass

    async def scenario(config: PrismaConfig) -> None:
        async with PrismaStream(config) as stream:
            await stream.send_audio(b"\x00" * 8000 * 2 * 4)  # 4 s of 8 kHz audio at once

    with pytest.raises(PrismaOverflow):
        asyncio.run(with_fake_prisma(fake_prisma, scenario))
