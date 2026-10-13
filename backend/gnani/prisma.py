"""Thin client for Gnani Prisma v2.5 realtime speech-to-text.

Protocol: https://docs.gnani.ai/api/STT/stt-websocket. Audio goes up as binary frames of
exactly 1,024 bytes (512 samples of 16-bit little-endian mono PCM) at real-time cadence;
results come back as JSON text frames with a `type` field.
"""

import asyncio
import json
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path
from types import TracebackType
from typing import Any, Literal, Self

from websockets.asyncio.client import ClientConnection, connect

FRAME_BYTES = 1024
SAMPLES_PER_FRAME = FRAME_BYTES // 2

EventType = Literal["connected", "processing", "transcript", "error", "settings_updated"]
EVENT_TYPES: frozenset[str] = frozenset(
    {"connected", "processing", "transcript", "error", "settings_updated"}
)


@dataclass(frozen=True)
class PrismaConfig:
    api_key: str
    url: str = "wss://api.vachana.ai/stt/v3/stream"
    lang_code: str = "kn-IN"
    sample_rate: int = 8000
    output_format: Literal["verbatim", "transcribe"] = "transcribe"
    bias_words: tuple[str, ...] = ()
    bias_score: float | None = None
    min_silence_ms: int | None = None

    def headers(self) -> dict[str, str]:
        headers = {
            "x-api-key-id": self.api_key,
            "lang_code": self.lang_code,
            "x-sample-rate": str(self.sample_rate),
            "x-format": self.output_format,
        }
        if self.bias_words:
            # ensure_ascii keeps Kannada words as \u escapes: HTTP headers must be ASCII.
            headers["x-bias-list"] = json.dumps(list(self.bias_words))
            if self.bias_score is not None:
                headers["x-bias-score"] = str(self.bias_score)
        if self.min_silence_ms is not None:
            headers["x-min-silence-ms"] = str(self.min_silence_ms)
        return headers


@dataclass(frozen=True)
class SttEvent:
    type: EventType
    text: str = ""
    segment_index: int | None = None
    audio_duration_ms: int | None = None
    latency_ms: int | None = None
    message: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

    def to_client(self) -> dict[str, Any]:
        """The fields the browser needs, without the raw server payload."""
        payload: dict[str, Any] = {"type": self.type}
        if self.type == "transcript":
            payload |= {
                "text": self.text,
                "segment_index": self.segment_index,
                "audio_duration_ms": self.audio_duration_ms,
                "latency_ms": self.latency_ms,
            }
        if self.message:
            payload["message"] = self.message
        return payload


def parse_event(payload: str) -> SttEvent | None:
    """Parse one server text frame; returns None for anything that is not a known event."""
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict) or data.get("type") not in EVENT_TYPES:
        return None
    return SttEvent(
        type=data["type"],
        text=data.get("text", ""),
        segment_index=data.get("segment_index"),
        audio_duration_ms=data.get("audio_duration_ms"),
        latency_ms=data.get("latency"),
        message=data.get("message", ""),
        raw=data,
    )


def load_bias_words(path: Path) -> tuple[str, ...]:
    if not path.exists():
        return ()
    words = json.loads(path.read_text(encoding="utf-8"))
    return tuple(str(w) for w in words)


class PrismaStream:
    """One realtime STT session. Use as an async context manager.

    `send_audio` accepts PCM of any length; it is cut into 1,024-byte frames and sent at
    real-time pace so bursts from the browser never reach Prisma as a burst.
    """

    def __init__(self, config: PrismaConfig) -> None:
        self.config = config
        self._ws: ClientConnection | None = None
        self._buffer = bytearray()
        self._frames: asyncio.Queue[bytes | None] = asyncio.Queue()
        self._sender: asyncio.Task[None] | None = None
        self._frame_seconds = SAMPLES_PER_FRAME / config.sample_rate

    async def __aenter__(self) -> Self:
        self._ws = await connect(self.config.url, additional_headers=self.config.headers())
        self._sender = asyncio.create_task(self._send_loop())
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()

    async def send_audio(self, pcm: bytes) -> None:
        self._buffer.extend(pcm)
        while len(self._buffer) >= FRAME_BYTES:
            self._frames.put_nowait(bytes(self._buffer[:FRAME_BYTES]))
            del self._buffer[:FRAME_BYTES]

    async def flush(self) -> None:
        """Send any partial frame, padded with silence."""
        if self._buffer:
            self._frames.put_nowait(bytes(self._buffer).ljust(FRAME_BYTES, b"\x00"))
            self._buffer.clear()

    async def events(self) -> AsyncIterator[SttEvent]:
        assert self._ws is not None, "use PrismaStream as an async context manager"
        async for message in self._ws:
            if isinstance(message, str) and (event := parse_event(message)):
                yield event

    async def update_bias(self, words: list[str], score: float | None = None) -> None:
        assert self._ws is not None
        bias: dict[str, Any] = {"list": words}
        if score is not None:
            bias["score"] = score
        await self._ws.send(json.dumps({"type": "update_settings", "bias": bias}))

    async def aclose(self) -> None:
        if self._sender is not None and not self._sender.done():
            await self.flush()
            self._frames.put_nowait(None)
            try:
                await asyncio.wait_for(self._sender, timeout=5)
            except TimeoutError:
                self._sender.cancel()
        if self._ws is not None:
            await self._ws.close()

    async def _send_loop(self) -> None:
        assert self._ws is not None
        next_send = time.monotonic()
        while (frame := await self._frames.get()) is not None:
            now = time.monotonic()
            if next_send > now:
                await asyncio.sleep(next_send - now)
            else:
                next_send = now
            await self._ws.send(frame)
            next_send += self._frame_seconds
