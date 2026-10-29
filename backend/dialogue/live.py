"""The calls in progress, and the supervisor consoles watching them.

Each call route registers its call here. Every JSON message its caller's browser gets is copied
to every console, and a console's officer commands are queued for that call's route to carry
out, so only the call's own route ever changes its dialogue.
"""

import asyncio
import contextlib
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

from dialogue.machine import Dialogue

# Messages queued for a console that has stopped reading before it is cut off: it reconnects
# and starts again from the list of calls.
MAX_BACKLOG = 1000

CommandKind = Literal["take_over", "say", "hand_back"]


@dataclass
class OfficerCommand:
    kind: CommandKind
    text: str = ""
    # Resolved by the call route once carried out, or failed with the reason it could not be.
    # Cancelled by the console when it stops waiting, so the call route skips it.
    done: asyncio.Future[None] = field(
        default_factory=lambda: asyncio.get_running_loop().create_future()
    )

    def __post_init__(self) -> None:
        # A failure nobody is waiting for any more is expected, not worth a warning.
        self.done.add_done_callback(lambda done: done.cancelled() or done.exception())


@dataclass
class LiveCall:
    dialogue: Dialogue
    started_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    commands: asyncio.Queue[OfficerCommand] = field(default_factory=asyncio.Queue)

    def summary(self) -> dict[str, Any]:
        return {
            "call_id": self.dialogue.call_id,
            "started_at": self.started_at,
            "state": self.dialogue.state.value,
            "reason": self.dialogue.reason,
            "officer": self.dialogue.officer,
            "slots": asdict(self.dialogue.slots),
            # What the caller has said so far, for a console that joins mid-call.
            "turns": list(self.dialogue.turns),
        }


class Watcher:
    """One console's queue of messages. None in the queue means it fell too far behind."""

    def __init__(self) -> None:
        self.messages: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue(MAX_BACKLOG + 1)

    def put(self, message: dict[str, Any]) -> bool:
        """Queue a message; False once the console has fallen too far behind."""
        if self.messages.qsize() >= MAX_BACKLOG:
            while not self.messages.empty():
                self.messages.get_nowait()
            self.messages.put_nowait(None)
            return False
        self.messages.put_nowait(message)
        return True


class LiveCalls:
    def __init__(self) -> None:
        self._calls: dict[str, LiveCall] = {}
        self._watchers: set[Watcher] = set()

    def __iter__(self) -> Iterator[LiveCall]:
        return iter(list(self._calls.values()))

    def get(self, call_id: str) -> LiveCall | None:
        return self._calls.get(call_id)

    def open(self, dialogue: Dialogue) -> LiveCall:
        call = LiveCall(dialogue)
        self._calls[dialogue.call_id] = call
        self._broadcast({"type": "call_started", **call.summary()})
        return call

    def close(self, call_id: str) -> None:
        call = self._calls.pop(call_id, None)
        if call is None:
            return
        # Commands nobody will carry out now.
        while not call.commands.empty():
            command = call.commands.get_nowait()
            if not command.done.done():
                command.done.set_exception(LookupError("The call has ended"))
        self._broadcast({"type": "call_closed", "call_id": call_id})

    def publish(self, call_id: str, message: dict[str, Any]) -> None:
        """Copy one message a caller's browser got to every console."""
        self._broadcast({"type": "call_event", "call_id": call_id, "event": message})

    @contextlib.contextmanager
    def watch(self) -> Iterator[Watcher]:
        watcher = Watcher()
        self._watchers.add(watcher)
        try:
            yield watcher
        finally:
            self._watchers.discard(watcher)

    def _broadcast(self, message: dict[str, Any]) -> None:
        for watcher in list(self._watchers):
            if not watcher.put(message):
                self._watchers.discard(watcher)
