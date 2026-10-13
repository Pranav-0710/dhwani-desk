import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator
from typing import Protocol

from audio.outbox import Outbox
from gnani.timbre import TimbreError

logger = logging.getLogger("dhwani.speaker")

# Longest an agent line may take from request to last chunk before it is abandoned.
UTTERANCE_DEADLINE_S = 20.0


class SpeechStream(Protocol):
    def stream(self, text: str, voice: str | None = None) -> AsyncIterator[bytes]: ...


class Speaker:
    """Speaks agent lines one at a time, in order, over the browser socket.

    For each line the browser receives `agent_text`, `agent_audio_start` (sample rate),
    binary PCM frames, then `agent_audio_end`. A Timbre failure sends `tts_error` and moves
    on to the next line: the call itself carries on.
    """

    def __init__(self, tts: SpeechStream, outbox: Outbox, sample_rate: int) -> None:
        self._tts = tts
        self._outbox = outbox
        self._sample_rate = sample_rate
        self._lines: asyncio.Queue[str] = asyncio.Queue()
        self._worker = asyncio.create_task(self._run())

    def say(self, text: str) -> None:
        self._lines.put_nowait(text)

    async def aclose(self) -> None:
        self._worker.cancel()
        # The worker may also have died because the browser socket closed; either is fine.
        await asyncio.gather(self._worker, return_exceptions=True)

    async def _run(self) -> None:
        while True:
            text = await self._lines.get()
            await self._speak(text)

    async def _speak(self, text: str) -> None:
        await self._outbox.json({"type": "agent_text", "text": text})
        await self._outbox.json({"type": "agent_audio_start", "sample_rate": self._sample_rate})
        try:
            async with asyncio.timeout(UTTERANCE_DEADLINE_S):
                async with contextlib.aclosing(self._tts.stream(text)) as chunks:
                    async for pcm in chunks:
                        await self._outbox.audio(pcm)
        except (TimbreError, TimeoutError) as exc:
            logger.warning("TTS failed: %s", exc)
            await self._outbox.json({"type": "tts_error", "message": str(exc) or "timed out"})
        await self._outbox.json({"type": "agent_audio_end"})
