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

    The queue is unbounded; the dialogue queues one reply (a few lines) per caller turn.
    """

    def __init__(self, tts: SpeechStream, outbox: Outbox, sample_rate: int) -> None:
        self._tts = tts
        self._outbox = outbox
        self._sample_rate = sample_rate
        self._lines: asyncio.Queue[str] = asyncio.Queue()
        # Event-loop time by which the browser will have played everything sent so far: it
        # plays chunks back to back from the first one's arrival, and Timbre sends faster
        # than real time, so the audio is still playing after the last chunk has gone.
        self._heard_by = 0.0
        self._worker = asyncio.create_task(self._run())

    def say(self, text: str) -> None:
        self._lines.put_nowait(text)

    async def wait_until_heard(self) -> None:
        """Return once every queued line has been sent and should have finished playing."""
        await self._lines.join()
        loop = asyncio.get_running_loop()
        await asyncio.sleep(max(0.0, self._heard_by - loop.time()))

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
            finally:
                self._lines.task_done()

    async def _speak(self, text: str) -> None:
        await self._outbox.json({"type": "agent_text", "text": text})
        await self._outbox.json({"type": "agent_audio_start", "sample_rate": self._sample_rate})
        try:
            async with asyncio.timeout(UTTERANCE_DEADLINE_S):
                async with contextlib.aclosing(self._tts.stream(text)) as chunks:
                    async for pcm in chunks:
                        await self._outbox.audio(pcm)
                        self._played(pcm)
        except asyncio.CancelledError:
            raise
        except (TimbreError, TimeoutError) as exc:
            logger.warning("TTS failed: %s", exc)
            await self._outbox.json({"type": "tts_error", "message": str(exc) or "timed out"})
        except Exception as exc:
            logger.exception("Unexpected TTS failure")
            await self._outbox.json({"type": "tts_error", "message": type(exc).__name__})
        await self._outbox.json({"type": "agent_audio_end"})

    def _played(self, pcm: bytes) -> None:
        now = asyncio.get_running_loop().time()
        self._heard_by = max(now, self._heard_by) + len(pcm) / (2 * self._sample_rate)
