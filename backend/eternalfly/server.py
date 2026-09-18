"""FastAPI app streaming ReadingSession.tick() results over a WebSocket as JSON."""

import asyncio
from dataclasses import asdict

from fastapi import FastAPI, WebSocket
from starlette.websockets import WebSocketDisconnect


def tick_result_to_json(tick_result) -> dict:
    """Convert a TickResult into a plain JSON-serializable dict mirroring its fields."""
    return asdict(tick_result)


def create_app(session, tick_interval_seconds: float = 0.05) -> FastAPI:
    """Build a FastAPI app that streams `session.tick()` results over `/ws` as JSON,
    one message per tick, until the client disconnects."""
    app = FastAPI()

    @app.websocket("/ws")
    async def stream_ticks(websocket: WebSocket) -> None:
        """Accept the connection and repeatedly send tick results until disconnected."""
        await websocket.accept()
        try:
            while True:
                tick_result = session.tick()
                await websocket.send_json(tick_result_to_json(tick_result))
                await asyncio.sleep(tick_interval_seconds)
        except WebSocketDisconnect:
            return

    return app
