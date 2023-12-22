"""AgentContext class."""
from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Union

from infscale.constants import HEART_BEAT_PERIOD
from infscale.utils.timer import Timer

if TYPE_CHECKING:
    from grpc import ServicerContext
    from infscale.controller.controller import Controller

DEFAULT_TIMEOUT = 1.5 * HEART_BEAT_PERIOD


class AgentContext:
    """Agent Context class."""

    def __init__(self, ctrl: Controller, id: str):
        """Initialize instance."""
        self.ctrl = ctrl
        self.id: str = id

        self.grpc_ctx = None
        self.grpc_ctx_event = asyncio.Event()

        self.alive: bool = False
        self.timer: Timer = None

    def get_grpc_ctx(self) -> Union[ServicerContext, None]:
        """Return grpc context (i.e., servicer context)."""
        return self.grpc_ctx

    def set_grpc_ctx(self, ctx: ServicerContext):
        """Set grpc servicer context."""
        self.grpc_ctx = ctx

    def get_grpc_ctx_event(self) -> asyncio.Event:
        """Return grpc context event."""
        return self.grpc_ctx_event

    def set_grpc_ctx_event(self):
        """Set event for grpc context.

        This will release the event.
        """
        self.grpc_ctx_event.set()

    def keep_alive(self):
        """Set agent's status to alive."""
        self.alive = True

        if not self.timer:
            self.timer = Timer(DEFAULT_TIMEOUT, self.ctrl.delete_agent_context, self.id)
        self.timer.renew()

    def reset(self):
        """Reset the agent context state."""
        self.alive = False
        self.timer = None

        self.grpc_ctx = None
        self.set_grpc_ctx_event()
