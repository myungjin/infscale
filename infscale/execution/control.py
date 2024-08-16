# Copyright 2024 Cisco Systems, Inc. and its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""Control channel class."""

import asyncio
from asyncio import StreamReader, StreamWriter

from infscale import get_logger

logger = get_logger()


MESSAGE_SIZE = 100


class Channel:
    """Control Channel class."""

    def __init__(self, rank: int, world_size: int, addr: str, port: int):
        """Initialize an instance."""
        self.rank = rank
        self.world_size = world_size
        self.addr = addr
        self.port = port

        self.peers: dict[int, tuple[StreamReader, StreamWriter]] = {}

    async def _setup_server(self, setup_done: asyncio.Event) -> None:
        server = await asyncio.start_server(self._handle_client, self.addr, self.port)

        addrs = ", ".join(str(sock.getsockname()) for sock in server.sockets)
        logger.info(f"Serving on {addrs}")

        setup_done.set()
        logger.info("set the setup_done event")

        async with server:
            await server.serve_forever()

    async def _handle_client(self, reader: StreamReader, writer: StreamWriter) -> None:
        data = await reader.read(MESSAGE_SIZE)
        message = data.decode()
        peer_rank = int(message)

        logger.info(f"peer rank: {peer_rank}")
        # save reader and writer streams for peer rank
        self.peers[peer_rank] = (reader, writer)

    async def _setup_client(self, setup_done: asyncio.Event) -> None:
        logger.info(f"setting up a client: {self.addr}:{self.port}")
        for i in range(3):
            try:
                reader, writer = await asyncio.open_connection(self.addr, self.port)
            except Exception as e:
                logger.warning(f"try {i+1}: exception occurred: {e}")
                await asyncio.sleep(3)

        # send my rank to rank 0
        message = f"{self.rank}"
        writer.write(message.encode())
        await writer.drain()
        logger.info(f"sent rank info({message}) to server")
        # server is always rank 0
        self.peers[0] = (reader, writer)

        setup_done.set()
        logger.info("set the setup_done event")

    async def wait_readiness(self):
        """Wait until control channel is fully configured."""
        if self.rank != 0:
            # this is client, configuration is done during calling self.setup()
            # nothing to do
            return

        while len(self.peers) != self.world_size - 1:
            await asyncio.sleep(1)

    async def setup(self) -> None:
        """Set up the channel."""
        setup_done = asyncio.Event()

        if self.rank == 0:
            _ = asyncio.create_task(self._setup_server(setup_done))
        else:
            _ = asyncio.create_task(self._setup_client(setup_done))

        # wait until setting up either server or client is done
        await setup_done.wait()

        logger.info("channel setup is done")

    async def send(self, rank: int, message: str) -> None:
        """Send control information to a receiver."""
        _, writer = self.peers[rank]
        writer.write(message.encode())
        await writer.drain()

    async def recv(self, rank: int) -> str:
        """Receive control information from a sender."""
        reader, _ = self.peers[rank]
        data = await reader.read(MESSAGE_SIZE)
        message = data.decode()

        return message

    async def sync(self, rank: int, mode="send"):
        """Synchronize send/recv."""
        if mode == "send":
            await self.send(rank, mode)
            _ = await self.recv(rank)
        else:  # ack
            _ = await self.recv(rank)
            await self.send(rank, mode)
