"""Thin client for Gnani Prisma v2.5 realtime speech-to-text.

Protocol: https://docs.gnani.ai/api/STT/stt-websocket. Audio goes up as binary frames of
exactly 1,024 bytes (512 samples of 16-bit little-endian mono PCM) at real-time cadence;
results come back as JSON text frames with a `type` field. There is no end-of-stream
message: either side closes, and sessions are capped at 15 minutes.
"""

import asyncio
import contextlib
import json
import logging
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path
from types import TracebackType
from typing import Any, Literal, Self, get_args

from websockets.asyncio.client import ClientConnection, connect
from websockets.exceptions import ConnectionClosed, ConnectionClosedError, InvalidStatus

FRAME_BYTES = 1024
SAMPLES_PER_FRAME = FRAME_BYTES // 2
# A backlog longer than this many frames is drained faster than real time.
CATCHUP_FRAMES = 4
CATCHUP_PACE = 0.75
# More buffered audio than this means the client is not streaming in real time.
MAX_BUFFERED_SECONDS = 3.0

EventType = Literal["connected", "processing", "transcript", "error", "settings_updated"]
EVENT_TYPES: frozenset[str] = frozenset(get_args(EventType))

# websockets logs every handshake header at DEBUG, including the API key.
logging.getLogger("websockets.client").setLevel(logging.INFO)


class PrismaError(Exception):
    """Base class for failures of the Prisma session."""


class PrismaRejected(PrismaError):
    def __init__(self, status: int) -> None:
        super().__init__(f"Prisma rejected the connection: HTTP {status}")
        self.status = status


class PrismaUnavailable(PrismaError):
    def __init__(self, reason: str) -> None:
        super().__init__(f"Prisma connection failed: {reason}")


class PrismaClosed(PrismaError):
    def __init__(self, code: int | None, reason: str) -> None:
        super().__init__(f"Prisma closed the session abnormally: {code} {reason}".strip())
        self.code = code
        self.reason = reason


class PrismaOverflow(PrismaError):
    def __init__(self) -> None:
        super().__init__(
            f"More than {MAX_BUFFERED_SECONDS:g} s of audio buffered: send audio in real time"
        )


@dataclass(frozen=True)
class PrismaConfig:
    api_key: str = field(repr=False)
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
        text=data.get("text") or "",
        segment_index=data.get("segment_index"),
        audio_duration_ms=data.get("audio_duration_ms"),
        latency_ms=data.get("latency"),
        message=data.get("message") or "",
        raw=data,
    )


def load_bias_words(path: Path) -> tuple[str, ...]:
    if not path.exists():
        return ()
    words = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(words, list) or not all(isinstance(w, str) for w in words):
        raise ValueError(f"{path} must contain a JSON list of strings")
    return tuple(words)


class PrismaStream:
    """One realtime STT session. Use as an async context manager.

    `send_audio` accepts PCM of any length; it is cut into 1,024-byte frames and sent at
    real-time pace. A backlog (handshake delay, network stall) is drained slightly faster
    than real time; a client that keeps sending faster than real time gets PrismaOverflow.
    """

    def __init__(self, config: PrismaConfig) -> None:
        self.config = config
        self._ws: ClientConnection | None = None
        self._buffer = bytearray()
        self._frames: asyncio.Queue[bytes | None] = asyncio.Queue()
        self._sender: asyncio.Task[None] | None = None
        self._frame_seconds = SAMPLES_PER_FRAME / config.sample_rate
        self._max_frames = int(MAX_BUFFERED_SECONDS / self._frame_seconds)

    @property
    def close_code(self) -> int | None:
        return self._ws.close_code if self._ws is not None else None

    @property
    def close_reason(self) -> str:
        return (self._ws.close_reason or "") if self._ws is not None else ""

    async def __aenter__(self) -> Self:
        try:
            self._ws = await connect(self.config.url, additional_headers=self.config.headers())
        except InvalidStatus as exc:
            raise PrismaRejected(exc.response.status_code) from exc
        except (OSError, TimeoutError, ConnectionClosed) as exc:
            raise PrismaUnavailable(type(exc).__name__) from exc
        self._sender = asyncio.create_task(self._send_loop())
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if exc is not None:
            self.discard_pending()
        await self.aclose()

    async def send_audio(self, pcm: bytes) -> None:
        self._buffer.extend(pcm)
        while len(self._buffer) >= FRAME_BYTES:
            if self._frames.qsize() >= self._max_frames:
                raise PrismaOverflow()
            self._frames.put_nowait(bytes(self._buffer[:FRAME_BYTES]))
            del self._buffer[:FRAME_BYTES]

    async def flush(self) -> None:
        """Send any partial frame, padded with silence."""
        if self._buffer:
            self._frames.put_nowait(bytes(self._buffer).ljust(FRAME_BYTES, b"\x00"))
            self._buffer.clear()

    def discard_pending(self) -> None:
        """Drop audio not yet sent, e.g. when the listener has gone and nobody needs it."""
        self._buffer.clear()
        while not self._frames.empty():
            self._frames.get_nowait()

    async def events(self) -> AsyncIterator[SttEvent]:
        """Server events until Prisma closes. A clean close ends the iterator; check
        `close_code` / `close_reason` afterwards. An abnormal close raises PrismaClosed."""
        assert self._ws is not None, "use PrismaStream as an async context manager"
        try:
            async for message in self._ws:
                if isinstance(message, str) and (event := parse_event(message)):
                    yield event
        except ConnectionClosedError as exc:
            code = exc.rcvd.code if exc.rcvd else None
            reason = exc.rcvd.reason if exc.rcvd else ""
            raise PrismaClosed(code, reason) from exc

    async def update_bias(self, words: list[str], score: float | None = None) -> None:
        assert self._ws is not None
        bias: dict[str, Any] = {"list": words}
        if score is not None:
            bias["score"] = score
        await self._ws.send(json.dumps({"type": "update_settings", "bias": bias}))

    async def aclose(self) -> None:
        try:
            if self._sender is not None and not self._sender.done():
                await self.flush()
                self._frames.put_nowait(None)
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(self._sender, timeout=5)
        finally:
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
            try:
                await self._ws.send(frame)
            except ConnectionClosed:
                # The receive side reports the close; nothing more to send.
                return
            pace = CATCHUP_PACE if self._frames.qsize() > CATCHUP_FRAMES else 1.0
            next_send += self._frame_seconds * pace
