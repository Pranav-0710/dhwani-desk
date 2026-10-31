import asyncio
from collections.abc import Callable
from typing import Any

from fastapi import WebSocket


class Outbox:
    """Serialises sends to one browser socket: transcripts and agent audio come from
    different tasks, and frames from two tasks must never interleave mid-send.

    `watch`, when given, sees every JSON message too (not the audio), as it is sent: the call
    route copies them to the supervisor consoles this way."""

    def __init__(
        self, websocket: WebSocket, watch: Callable[[dict[str, Any]], None] | None = None
    ) -> None:
        self._websocket = websocket
        self._watch = watch
        self._lock = asyncio.Lock()

    async def json(self, payload: dict[str, Any]) -> None:
        if self._watch is not None:
            self._watch(payload)
        async with self._lock:
            await self._websocket.send_json(payload)

    async def audio(self, pcm: bytes) -> None:
        async with self._lock:
            await self._websocket.send_bytes(pcm)
