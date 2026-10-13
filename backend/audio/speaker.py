import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator
from typing import Protocol

from starlette.websockets import WebSocketDisconnect

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
    binary PCM frames, then `agent_audio_end` — always, even when synthesis fails, because
    the browser mutes its mic between start and end. A failure also sends `tts_error`; the
    call carries on with the next line. Lines still queued when the call ends are dropped.

    The queue is unbounded: the echo agent says one line per transcript segment. The
    dialogue layer (#9) should keep it to one reply per caller turn.
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
        await asyncio.gather(self._worker, return_exceptions=True)

    async def _run(self) -> None:
        while True:
            text = await self._lines.get()
            try:
                await self._speak(text)
            except (WebSocketDisconnect, RuntimeError, OSError):
                # The browser has gone: nobody to speak to.
                return
            except Exception:
                logger.exception("Speaking a line failed; continuing with the next one")

    async def _speak(self, text: str) -> None:
        await self._outbox.json({"type": "agent_text", "text": text})
        await self._outbox.json({"type": "agent_audio_start", "sample_rate": self._sample_rate})
        try:
            async with asyncio.timeout(UTTERANCE_DEADLINE_S):
                async with contextlib.aclosing(self._tts.stream(text)) as chunks:
                    async for pcm in chunks:
                        await self._outbox.audio(pcm)
        except asyncio.CancelledError:
            raise
        except (TimbreError, TimeoutError) as exc:
            logger.warning("TTS failed: %s", exc)
            await self._outbox.json({"type": "tts_error", "message": str(exc) or "timed out"})
        except Exception as exc:
            logger.exception("Unexpected TTS failure")
            await self._outbox.json({"type": "tts_error", "message": type(exc).__name__})
        await self._outbox.json({"type": "agent_audio_end"})
