"""Thin client for Gnani Timbre v2.5 text-to-speech.

Docs: https://docs.gnani.ai/api/TTS/tts-inference (one response) and
https://docs.gnani.ai/api/TTS/tts-sse (streamed). Both return 16-bit mono PCM here
(`container: raw`), so callers can play or forward it without decoding a container.
"""

import base64
import binascii
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

import httpx2

SAMPLE_RATES = frozenset({8000, 16000, 22050, 24000, 44100, 48000})
DEFAULT_VOICE = "Saanvi"
# Tight enough that a stalled call fails fast; the caller adds an overall deadline.
DEFAULT_TIMEOUT = httpx2.Timeout(connect=3.0, read=5.0, write=5.0, pool=2.0)


class TimbreError(Exception):
    """Any Timbre failure. `error_type` is the API's type, or NETWORK_ERROR, TIMEOUT,
    PROTOCOL_ERROR or INCOMPLETE_STREAM for failures detected on our side."""

    def __init__(self, status: int | None, error_type: str, message: str) -> None:
        suffix = f" (HTTP {status})" if status else ""
        super().__init__(f"Timbre {error_type}: {message}{suffix}")
        self.status = status
        self.error_type = error_type


@dataclass(frozen=True)
class TimbreConfig:
    api_key: str = field(repr=False)
    base_url: str = "https://api.vachana.ai/api/v1/tts"
    model: str = "timbre-v2.5"
    voice: str = DEFAULT_VOICE
    language: str = "kn-IN"
    sample_rate: int = 16000
    speed: float = 1.0

    def request_body(self, text: str, voice: str | None = None) -> dict[str, Any]:
        if not text.strip():
            raise ValueError("Timbre needs non-empty text")
        if self.sample_rate not in SAMPLE_RATES:
            raise ValueError(f"Timbre sample_rate must be one of {sorted(SAMPLE_RATES)}")
        return {
            "text": text,
            "model": self.model,
            "voice": voice or self.voice,
            "language": self.language,
            "speed": self.speed,
            "audio_config": {
                "sample_rate": self.sample_rate,
                "num_channels": 1,
                "sample_width": 2,
                "encoding": "linear_pcm",
                "container": "raw",
            },
        }


class TimbreClient:
    """Use as an async context manager; one HTTP connection pool per client."""

    def __init__(
        self,
        config: TimbreConfig,
        transport: httpx2.AsyncBaseTransport | None = None,
        timeout: httpx2.Timeout = DEFAULT_TIMEOUT,
    ) -> None:
        if not config.api_key:
            raise ValueError("Timbre needs an API key")
        self.config = config
        self._http = httpx2.AsyncClient(
            base_url=config.base_url,
            headers={"X-API-Key-ID": config.api_key},
            timeout=timeout,
            transport=transport,
        )

    async def __aenter__(self) -> "TimbreClient":
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._http.aclose()

    async def synthesize(self, text: str, voice: str | None = None) -> bytes:
        """The whole utterance as 16-bit mono PCM at `config.sample_rate`."""
        body = self.config.request_body(text, voice)
        try:
            response = await self._http.post("/inference", json=body)
        except httpx2.HTTPError as exc:
            raise _transport_error(exc) from exc
        if response.status_code != 200:
            raise _error_from(response.status_code, response.content)
        return response.content

    async def stream(self, text: str, voice: str | None = None) -> AsyncIterator[bytes]:
        """PCM chunks as Timbre produces them, for low time to first audio.

        Every chunk holds whole 16-bit samples. Raises TimbreError if the stream ends before
        Timbre's final event, so a cut-off utterance is never mistaken for a complete one.

        To stop early (caller barges in) without leaking the HTTP response, iterate inside
        `async with contextlib.aclosing(client.stream(text)) as chunks:`, and wrap the whole
        thing in `asyncio.timeout(...)` for an overall deadline.
        """
        body = self.config.request_body(text, voice)
        carry = b""
        try:
            async with self._http.stream("POST", "/sse", json=body) as response:
                if response.status_code != 200:
                    raise _error_from(response.status_code, await response.aread())
                async for event, data in _sse_events(response.aiter_lines()):
                    if event == "error":
                        raise _error_from(None, data.encode())
                    if event == "start":
                        continue
                    audio, is_final = _parse_chunk(data)
                    pcm = carry + audio
                    whole = len(pcm) - len(pcm) % 2
                    carry = pcm[whole:]
                    if whole:
                        yield pcm[:whole]
                    if is_final or event == "complete":
                        return
        except httpx2.HTTPError as exc:
            raise _transport_error(exc) from exc
        raise TimbreError(None, "INCOMPLETE_STREAM", "stream ended before the final event")


def _parse_chunk(data: str) -> tuple[bytes, bool]:
    try:
        payload = json.loads(data)
        audio = payload.get("audio") or ""
        return base64.b64decode(audio, validate=True), bool(payload.get("is_final"))
    except (ValueError, TypeError, AttributeError, binascii.Error) as exc:
        raise TimbreError(None, "PROTOCOL_ERROR", f"bad audio event: {data[:80]}") from exc


async def _sse_events(lines: AsyncIterator[str]) -> AsyncIterator[tuple[str | None, str]]:
    """Minimal Server-Sent Events parser: yields (event name, joined data) per message."""
    event: str | None = None
    data: list[str] = []
    async for line in lines:
        if line == "":
            if data:
                yield event, "\n".join(data)
            event, data = None, []
        elif line.startswith(":"):
            continue
        else:
            name, _, value = line.partition(":")
            value = value.removeprefix(" ")
            if name == "event":
                event = value
            elif name == "data":
                data.append(value)
    if data:
        yield event, "\n".join(data)


def _transport_error(exc: httpx2.HTTPError) -> TimbreError:
    kind = "TIMEOUT" if isinstance(exc, httpx2.TimeoutException) else "NETWORK_ERROR"
    return TimbreError(None, kind, type(exc).__name__)


def _error_from(status: int | None, body: bytes) -> TimbreError:
    """Accepts {"error": {"type", "message"}} and the flat {"type", "message"} shape."""
    try:
        parsed = json.loads(body)
    except ValueError:
        parsed = None
    if isinstance(parsed, dict):
        nested = parsed.get("error")
        source = nested if isinstance(nested, dict) else parsed
        error_type = source.get("type") or "API_ERROR"
        message = source.get("message") or body[:200].decode(errors="replace")
        return TimbreError(status, str(error_type), str(message))
    return TimbreError(status, "API_ERROR", body[:200].decode(errors="replace"))
