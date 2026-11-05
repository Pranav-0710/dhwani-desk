import asyncio
import gc
from typing import Any

import anyio
import pytest

from cancelling import cancel_and_wait


def test_a_cancel_arriving_during_the_wait_stays_with_the_scope_that_sent_it() -> None:
    """The scope's cancel lands before the child has stepped: how a closing socket's test ends."""

    async def scenario() -> bool:
        with anyio.CancelScope() as scope:
            child = asyncio.create_task(asyncio.sleep(3600))
            await asyncio.sleep(0)
            asyncio.get_running_loop().call_soon(scope.cancel)
            await cancel_and_wait(child)
        return scope.cancelled_caught

    assert asyncio.run(scenario()) is True


def test_every_task_has_ended_when_it_returns() -> None:
    async def slow_to_stop() -> None:
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            await asyncio.sleep(0.01)
            raise

    async def scenario() -> list[bool]:
        tasks = [asyncio.create_task(slow_to_stop()) for _ in range(2)]
        await asyncio.sleep(0)
        await cancel_and_wait(*tasks)
        return [task.done() for task in tasks]

    assert asyncio.run(scenario()) == [True, True]


def test_a_task_that_failed_neither_raises_here_nor_warns_later() -> None:
    async def fails() -> None:
        raise ValueError("gone")

    async def scenario() -> list[dict[str, Any]]:
        unretrieved: list[dict[str, Any]] = []
        asyncio.get_running_loop().set_exception_handler(lambda _, ctx: unretrieved.append(ctx))
        task = asyncio.create_task(fails())
        await asyncio.sleep(0)
        await cancel_and_wait(task)
        del task
        gc.collect()
        return unretrieved

    assert asyncio.run(scenario()) == []


@pytest.mark.parametrize("started", [True, False])
def test_it_copes_with_no_tasks_and_with_a_task_that_has_not_started(started: bool) -> None:
    async def scenario() -> None:
        await cancel_and_wait()
        task = asyncio.create_task(asyncio.sleep(3600))
        if started:
            await asyncio.sleep(0)
        await cancel_and_wait(task)
        assert task.cancelled()

    asyncio.run(scenario())
