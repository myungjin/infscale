"""Timer class."""

import asyncio
from asyncio import Task
from typing import Any, Callable


class Timer:
    """Timer Class."""

    def __init__(self, timeout: int, callback: Callable, *args: Any, **kwargs: Any):
        """Initialize timer instance.

        *args and **kwargs are parameters for callback
        """
        self._timeout = timeout
        self._callback = callback
        self._args = args
        self._kwargs = kwargs

        self._task = self.create()

    def _create(self) -> Task:
        return asyncio.create_task(self._wait())

    async def _wait(self) -> None:
        await asyncio.sleep(self._timeout)

        if asyncio.iscoroutinefunction(self._callback):
            await self._callback(*self._args, **self._kwargs)
        else:
            self._callback(*self._args, **self._kwargs)

    async def cancel(self) -> None:
        """Cancel the timer."""
        await self._task.cancel()

    def renew(self) -> None:
        """Renew the timer."""
        self.cancel()

        self._task = self._create()
