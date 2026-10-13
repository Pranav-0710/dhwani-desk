"""Thin client for Gnani Timbre v2.5 text-to-speech.

Docs: https://docs.gnani.ai/api/TTS/tts-inference (one response) and
https://docs.gnani.ai/api/TTS/tts-sse (streamed). Both return 16-bit mono PCM here
(`container: raw`), so callers can play or forward it without decoding a container.
"""

import base64
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

import httpx2

SAMPLE_RATES = frozenset({8000, 16000, 22050, 24000, 44100, 48000})


class TimbreError(Exception):
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
    voice: str = "Kavin"
    language: str = "kn-IN"
    sample_rate: int = 16000
    speed: float = 1.0
    timeout_s: float = 30.0

    def request_body(self, text: str, voice: str | None = None) -> dict[str, Any]:
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

    def __init__(self, config: TimbreConfig, transport: httpx2.AsyncBaseTransport | None = None):
        self.config = config
        self._http = httpx2.AsyncClient(
            base_url=config.base_url,
            headers={"X-API-Key-ID": config.api_key},
            timeout=config.timeout_s,
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
        try:
            response = await self._http.post(
                "/inference", json=self.config.request_body(text, voice)
            )
        except httpx2.HTTPError as exc:
            raise TimbreError(None, "NETWORK_ERROR", type(exc).__name__) from exc
        if response.status_code != 200:
            raise _error_from(response.status_code, response.content)
        return response.content

    async def stream(self, text: str, voice: str | None = None) -> AsyncIterator[bytes]:
        """PCM chunks as Timbre produces them, for low time-to-first-audio."""
        body = self.config.request_body(text, voice)
        try:
            async with self._http.stream("POST", "/sse", json=body) as response:
                if response.status_code != 200:
                    raise _error_from(response.status_code, await response.aread())
                async for event, data in _sse_events(response.aiter_lines()):
                    if event == "error":
                        raise _error_from(None, data.encode())
                    if event == "chunk" or (event is None and "audio" in data):
                        payload = json.loads(data)
                        if payload.get("audio"):
                            yield base64.b64decode(payload["audio"])
                        if payload.get("is_final"):
                            return
                    elif event == "complete":
                        return
        except httpx2.HTTPError as exc:
            raise TimbreError(None, "NETWORK_ERROR", type(exc).__name__) from exc


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


def _error_from(status: int | None, body: bytes) -> TimbreError:
    try:
        error = json.loads(body).get("error", {})
        return TimbreError(status, error.get("type", "API_ERROR"), error.get("message", ""))
    except (ValueError, AttributeError):
        return TimbreError(status, "API_ERROR", body[:200].decode(errors="replace"))
