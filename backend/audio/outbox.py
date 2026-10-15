import asyncio
from typing import Any

from fastapi import WebSocket


class Outbox:
    """Serialises sends to one browser socket: transcripts and agent audio come from
    different tasks, and frames from two tasks must never interleave mid-send."""

    def __init__(self, websocket: WebSocket) -> None:
        self._websocket = websocket
        self._lock = asyncio.Lock()

    async def json(self, payload: dict[str, Any]) -> None:
        async with self._lock:
            await self._websocket.send_json(payload)

    async def audio(self, pcm: bytes) -> None:
        async with self._lock:
            await self._websocket.send_bytes(pcm)

    async def close(self, code: int = 1000) -> None:
        async with self._lock:
            await self._websocket.close(code=code)
