"""Stopping background tasks from a coroutine that may itself be cancelled at the same moment."""

import asyncio
from typing import Any


def _retrieve_error(task: asyncio.Task[Any]) -> None:
    if not task.cancelled():
        task.exception()


async def cancel_and_wait(*tasks: asyncio.Task[Any]) -> None:
    """Cancel these tasks and wait until they have ended; whatever they ended with is dropped.

    Not `asyncio.gather`: cancelled while a child has yet to stop, it raises the child's
    CancelledError in place of the one sent to the caller. That loses the message by which an anyio
    cancel scope around the caller knows the cancellation as its own, so it escapes the scope.
    """
    if not tasks:
        return
    for task in tasks:
        task.cancel()
        task.add_done_callback(_retrieve_error)
    await asyncio.wait(tasks)
