# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""IPC for communication between the main process that connects to the frontend."""

import asyncio
import json
import logging
import multiprocessing as mp
import uuid

from fastapi import WebSocket
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class Message(BaseModel):
    """Message."""

    source: str
    event: str
    data: dict

    def to_json(self) -> str:
        """Convert the message to a JSON string."""
        return json.dumps({"source": self.source, "event": self.event, "data": self.data})


class IPCConnection:
    """Pipe for IPC messages."""

    def __init__(self, event_queue: "mp.Queue[Message]") -> None:
        """The IPC pipe reads from the pipe and writes to the event queue."""
        self.id = uuid.uuid4()
        self.reader, self.writer = mp.Pipe()
        self.event_queue = event_queue

    def read(self) -> Message | None:
        """Read a message from the process.

        Only read the message if it is not from the same process handle.
        """
        if self.reader.poll():
            message: Message = self.reader.recv()
            if message.source != str(self.id):
                return message
        return None

    def send(self, message: Message) -> None:
        """Send a message to the IPC pipe."""
        if message.source != str(self.id):
            self.writer.send(message)

    def broadcast(self, event: str, data: dict) -> None:
        """Broadcast a message to the event queue."""
        self.event_queue.put(Message(source=str(self.id), event=event, data=data))


class IPCConnectionManager:
    """Manages WebSocket connections and broadcasts events to all connected clients.

    Currently it does not have specific topic routing, all messages are broadcast to all connected clients.
    """

    def __init__(self) -> None:
        self.websocket_clients: list[WebSocket] = []
        self.process_clients: list[IPCConnection] = []
        self.stop_event: asyncio.Event = asyncio.Event()
        self.event_queue = mp.Queue(maxsize=1000)
        self.broadcaster_task: asyncio.Task[None] = asyncio.create_task(
            self.broadcast_daemon(self.stop_event, self.event_queue)
        )

    def cleanup(self) -> None:
        """Shutdown the IPC."""
        logger.info("Shutting down IPC...")
        self._stop_event_queue_broadcaster()
        self.websocket_clients.clear()
        self.process_clients.clear()
        logger.info("IPC shutdown complete")

    async def _stop_event_queue_broadcaster(self) -> None:
        """Stop the event queue broadcaster."""
        self.stop_event.set()
        if self.event_queue_broadcaster_task is not None:
            self.event_queue_broadcaster_task.cancel()
            try:
                await self.event_queue_broadcaster_task
            except asyncio.CancelledError:
                pass

    async def connect_websocket(self, websocket_connection: WebSocket) -> None:
        """Connect a client to the WebSocket server."""
        await websocket_connection.accept()
        self.websocket_clients.append(websocket_connection)
        logger.info("Frontend client connected. Total clients: %d", len(self.websocket_clients))

    async def disconnect(self, connection: WebSocket | IPCConnection) -> None:
        """Disconnect a client from the WebSocket server."""
        if isinstance(connection, WebSocket):
            self.websocket_clients.remove(connection)
        elif isinstance(connection, IPCConnection):
            self.process_clients.remove(connection)
        else:
            raise ValueError("Invalid connection type")
        logger.info("Frontend client disconnected. Total clients: %d", len(self.websocket_clients))

    def create_ipc_pipe(self) -> IPCConnection:
        """Get a new IPC pipe.

        This is a convenience method to create a new IPC pipe and connect it to the IPC.
        """
        ipc_pipe = IPCConnection(self.event_queue)
        self.process_clients.append(ipc_pipe)
        return ipc_pipe

    async def _broadcast(self, message: Message) -> None:
        """Send a message to all connected clients and all non-generated processes."""
        if not self.websocket_clients:
            logger.debug("No websocket clients connected, message not sent")
        if not self.process_clients:
            logger.debug("No process clients connected, message not sent")

        for websocket_client in self.websocket_clients:
            try:
                await websocket_client.send_text(json.dumps({"event": message.event, "data": message.data}))
                logger.info("Sent event '%s' to websocket client %s", message.event, websocket_client.client)
            except Exception as e:
                logger.error("Error sending to websocket client: %s", e)
        for process_client in self.process_clients:
            try:
                logger.info("Sending event '%s' to process client %s", message.event, process_client.id)
                process_client.send(message)
            except Exception as e:
                logger.error("Error sending to process client: %s", e)

    async def add_event_to_queue(self, event: str, data: dict) -> None:
        """Add an event to the event queue."""
        logger.info("Adding event '%s' to event queue", event)
        self.event_queue.put(Message(source=str(uuid.uuid4()), event=event, data=data))

    async def broadcast_daemon(self, stop_event: asyncio.Event, event_queue: "mp.Queue[Message]") -> None:
        """Background task to poll event queue and broadcast to WebSocket clients."""
        while not stop_event.is_set():
            try:
                # Go over all the process clients and read the messages
                if not event_queue.empty():
                    try:
                        event_msg = event_queue.get_nowait()
                        await self._broadcast(event_msg)
                        logger.info("Broadcast event '%s' to clients", event_msg.event)
                    except Exception as e:
                        logger.error("Error processing event from queue: %s", e)
            except Exception as e:
                logger.error("Error in event queue broadcaster: %s", e)
            # Small sleep to avoid busy-waiting
            await asyncio.sleep(0.1)
        logger.info("Event queue broadcaster stopped")
