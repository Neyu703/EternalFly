"""FastAPI app streaming ReadingSession.tick() results over a WebSocket as JSON."""

import asyncio
import pathlib
import random
from dataclasses import asdict

from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from starlette.websockets import WebSocketDisconnect

from eternalfly.calibre_library import list_books
from eternalfly.text_encoder import load_and_tokenize_file

AUTOPLAY_MODE_OFF = "off"
AUTOPLAY_MODE_RESTART = "restart"
AUTOPLAY_MODE_SHUFFLE = "shuffle"


def tick_result_to_json(tick_result) -> dict:
    """Convert a TickResult into a plain JSON-serializable dict mirroring its fields."""
    return asdict(tick_result)


class LoadBookRequest(BaseModel):
    """Request body for POST /load-book: an absolute path to a .txt or .epub file."""

    path: str


def _apply_control_message(session, message: dict, current_autoplay_mode: str) -> str:
    """Apply one client control message to the session and return the resulting
    autoplay mode (unchanged unless the message sets it)."""
    message_type = message.get("type")
    if message_type == "set_paused":
        session.set_paused(bool(message.get("paused", False)))
    elif message_type == "set_speed_multiplier":
        session.set_speed_multiplier(float(message.get("value", 1.0)))
    elif message_type == "set_autoplay_mode":
        return message.get("mode", AUTOPLAY_MODE_OFF)
    return current_autoplay_mode


def _advance_past_finished_book(session, autoplay_mode: str, calibre_library_path: pathlib.Path | None) -> None:
    """React to the book having just finished, per the client's chosen autoplay mode.
    Does nothing when autoplay is off, so a finished book simply stays finished."""
    if autoplay_mode == AUTOPLAY_MODE_RESTART:
        session.restart()
    elif autoplay_mode == AUTOPLAY_MODE_SHUFFLE:
        _load_random_calibre_book(session, calibre_library_path)


def _load_random_calibre_book(session, calibre_library_path: pathlib.Path | None) -> None:
    """Load a random book from the configured Calibre library into the session. Falls
    back to restarting the current book if no library is configured, it has no
    loadable books, or the randomly chosen book fails to load (e.g. a corrupt file) —
    an unattended autoplay session must never crash on a single bad library entry."""
    try:
        books = list_books(calibre_library_path) if calibre_library_path is not None else []
        if not books:
            raise ValueError("no books available in the Calibre library")
        chosen_book = random.choice(books)
        tokens = load_and_tokenize_file(chosen_book.file_path)
        session.load_new_text(tokens)
    except (FileNotFoundError, ValueError):
        session.restart()


def create_app(
    session,
    tick_interval_seconds: float = 0.05,
    calibre_library_path: pathlib.Path | None = None,
) -> FastAPI:
    """Build a FastAPI app that streams `session.tick()` results over `/ws` as JSON,
    one message per tick, until the client disconnects."""
    app = FastAPI()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "tauri://localhost", "http://tauri.localhost"],
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

    @app.websocket("/ws")
    async def stream_ticks(websocket: WebSocket) -> None:
        """Accept the connection, repeatedly send tick results until disconnected, and
        concurrently apply playback control messages (pause, speed, autoplay mode) the
        client sends on the same socket."""
        await websocket.accept()
        autoplay_mode = AUTOPLAY_MODE_OFF

        async def receive_control_messages() -> None:
            """Apply each incoming control message to the session until disconnected."""
            nonlocal autoplay_mode
            try:
                while True:
                    message = await websocket.receive_json()
                    autoplay_mode = _apply_control_message(session, message, autoplay_mode)
            except WebSocketDisconnect:
                return

        receiver_task = asyncio.create_task(receive_control_messages())
        try:
            while True:
                tick_result = session.tick()
                if tick_result.book_finished:
                    # Level-triggered, not edge-triggered: restart()/load_new_text() make
                    # the very next tick's book_finished False again, so this fires at most
                    # once per finish — but it still fires promptly even if the autoplay
                    # mode was only set by the client *after* the book had already finished.
                    _advance_past_finished_book(session, autoplay_mode, calibre_library_path)
                await websocket.send_json(tick_result_to_json(tick_result))
                await asyncio.sleep(tick_interval_seconds)
        except WebSocketDisconnect:
            return
        finally:
            receiver_task.cancel()

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

    @app.get("/calibre-books")
    async def get_calibre_books() -> dict:
        """List books available in the configured Calibre library, resolved to
        absolute file paths the frontend can pass straight to /load-book. Returns
        an empty list if no library path was configured."""
        if calibre_library_path is None:
            return {"books": []}
        try:
            books = list_books(calibre_library_path)
        except FileNotFoundError as missing_library_error:
            raise HTTPException(status_code=404, detail=str(missing_library_error)) from missing_library_error
        return {
            "books": [
                {
                    "book_id": book.book_id,
                    "title": book.title,
                    "author": book.author,
                    "file_path": str(book.file_path),
                }
                for book in books
            ]
        }

    return app
