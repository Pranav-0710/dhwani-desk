"""Joins Prisma's transcript segments into caller turns."""

import asyncio

# Prisma ends a segment at a short pause, so one sentence can arrive in two or three pieces.
# A turn ends when no new segment arrives for this long; every extra 100 ms here is 100 ms
# before the agent answers, so #22 tunes it against real calls.
TURN_PAUSE_S = 0.6


class TurnCollector:
    """`add()` each transcript segment; finished turns come out of `turns`, in order.

    Finishing a turn only queues its text, so a slow decision on one turn never holds up or
    swallows the segments of the next.
    """

    def __init__(self, pause_s: float = TURN_PAUSE_S) -> None:
        self.pause_s = pause_s
        self.turns: asyncio.Queue[str] = asyncio.Queue()
        self._parts: list[str] = []
        self._timer: asyncio.Task[None] | None = None

    def add(self, segment: str) -> None:
        if not segment.strip():
            return
        self._parts.append(segment.strip())
        if self._timer:
            self._timer.cancel()
        self._timer = asyncio.create_task(self._finish_after_pause())

    async def aclose(self) -> None:
        if self._timer:
            self._timer.cancel()
            await asyncio.gather(self._timer, return_exceptions=True)

    async def _finish_after_pause(self) -> None:
        await asyncio.sleep(self.pause_s)
        # Nothing awaits between here and the end, so a new segment cannot cancel a turn
        # half-finished: it either restarts the pause above or starts the next turn.
        text, self._parts, self._timer = " ".join(self._parts), [], None
        self.turns.put_nowait(text)
