import asyncio
import json
import time

from websockets.asyncio.server import ServerConnection, serve

from gnani.prisma import FRAME_BYTES, PrismaConfig, PrismaStream, load_bias_words, parse_event


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


def test_parse_event_ignores_unknown_and_invalid_frames() -> None:
    assert parse_event('{"type": "something_new"}') is None
    assert parse_event("not json") is None
    assert parse_event("[1, 2]") is None


def test_load_bias_words_reads_file_and_tolerates_missing(tmp_path) -> None:
    path = tmp_path / "words.json"
    path.write_text('["ಕಸ", "ನೀರು"]', encoding="utf-8")

    assert load_bias_words(path) == ("ಕಸ", "ನೀರು")
    assert load_bias_words(tmp_path / "missing.json") == ()


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

    async def scenario() -> tuple[list[str], float]:
        async with serve(fake_prisma, "127.0.0.1", 0) as server:
            port = server.sockets[0].getsockname()[1]
            config = PrismaConfig(api_key="k", url=f"ws://127.0.0.1:{port}", sample_rate=8000)
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

    types, elapsed = asyncio.run(scenario())

    assert types == ["connected", "processing", "transcript"]
    assert [len(frame) for frame in received] == [FRAME_BYTES] * 3
    assert received[-1].endswith(b"\x00")
    assert request_headers["x-api-key-id"] == "k"
    # 3 frames at 64 ms each (8 kHz): the last one cannot leave before ~128 ms.
    assert elapsed >= 0.12
