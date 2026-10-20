"""WebSocket server that streams live rocket state to a viewer client."""

from __future__ import annotations
import asyncio
import json
import websockets
from .state import ViewerState

_clients: set = set()
_latest: dict = {}


async def _handler(websocket) -> None:
    _clients.add(websocket)
    try:
        if _latest:
            await websocket.send(json.dumps(_latest))
        async for _ in websocket:
            pass  # viewer is read-only
    finally:
        _clients.discard(websocket)


async def broadcast(vs: ViewerState) -> None:
    global _latest
    _latest = vs.to_dict()
    if _clients:
        msg = json.dumps(_latest)
        await asyncio.gather(*[c.send(msg) for c in list(_clients)], return_exceptions=True)


def start_server(host: str = "localhost", port: int = 8765) -> None:
    """Start the WebSocket server (blocking)."""
    async def _serve():
        async with websockets.serve(_handler, host, port):
            print(f"Viewer bridge listening on ws://{host}:{port}")
            await asyncio.Future()  # run forever

    asyncio.run(_serve())
