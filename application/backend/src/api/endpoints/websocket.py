# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""WebSocket API Endpoints"""

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect

from api.dependencies import get_ipc
from communication.ipc import IPCConnectionManager

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/ws", tags=["WebSocket"])


@router.websocket("/")
async def websocket_endpoint(
    websocket: WebSocket,
    ipc: Annotated[IPCConnectionManager, Depends(get_ipc)],
) -> None:
    """WebSocket endpoint for frontend plugin communication."""
    logger.info("Connecting client to WebSocket")
    await ipc.connect_websocket(websocket)
    logger.info("Client connected to WebSocket")

    try:
        # Keep connection alive and receive messages from frontend
        while True:
            message = await websocket.receive_json()
            event = message.get("event")
            data = message.get("data", {})
            if event:
                logger.info("Received event '%s' from client", event)
                await ipc.add_event_to_queue(event, data)
            else:
                logger.error("Invalid message from client: %s", message)
    except WebSocketDisconnect:
        logger.info("Client disconnected from WebSocket")
        await ipc.disconnect(websocket)
    except Exception as e:
        logger.error("WebSocket error: %s", e)
        await ipc.disconnect(websocket)
