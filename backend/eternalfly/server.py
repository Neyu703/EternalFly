"""FastAPI app streaming ReadingSession.tick() results over a WebSocket as JSON."""

import asyncio
import pathlib
from dataclasses import asdict

from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from starlette.websockets import WebSocketDisconnect

from eternalfly.text_encoder import load_and_tokenize_file


def tick_result_to_json(tick_result) -> dict:
    """Convert a TickResult into a plain JSON-serializable dict mirroring its fields."""
    return asdict(tick_result)


class LoadBookRequest(BaseModel):
    """Request body for POST /load-book: an absolute path to a .txt or .epub file."""

    path: str


def create_app(session, tick_interval_seconds: float = 0.05) -> FastAPI:
    """Build a FastAPI app that streams `session.tick()` results over `/ws` as JSON,
    one message per tick, until the client disconnects."""
    app = FastAPI()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "tauri://localhost", "http://tauri.localhost"],
        allow_methods=["POST"],
        allow_headers=["Content-Type"],
    )

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

    @app.post("/load-book")
    async def load_book(request: LoadBookRequest) -> dict:
        """Load and tokenize the file at request.path, then feed its tokens into the
        running session as the new book, restarting word progress from the beginning."""
        try:
            tokens = load_and_tokenize_file(pathlib.Path(request.path))
            session.load_new_text(tokens)
        except FileNotFoundError as missing_file_error:
            raise HTTPException(status_code=404, detail=str(missing_file_error)) from missing_file_error
        except ValueError as invalid_book_error:
            raise HTTPException(status_code=400, detail=str(invalid_book_error)) from invalid_book_error
        return {"status": "ok", "total_words": len(tokens)}

    return app
