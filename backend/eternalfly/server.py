"""FastAPI app streaming ReadingSession.advance() results over a WebSocket as JSON."""

import asyncio
import contextlib
import pathlib
import random
import time
from collections.abc import Callable
from dataclasses import asdict

import numpy
import torch
from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from starlette.websockets import WebSocketDisconnect

from eternalfly.calibre_library import list_books
from eternalfly.folder_library import list_books_in_folder
from eternalfly.text_encoder import load_and_tokenize_file

AUTOPLAY_MODE_OFF = "off"
AUTOPLAY_MODE_RESTART = "restart"
AUTOPLAY_MODE_SHUFFLE = "shuffle"

DEFAULT_LOAD_BOOK_TIMEOUT_SECONDS = 30.0
DEFAULT_WORDS_PER_MINUTE = 150.0
DEFAULT_LOOKAHEAD_WORD_COUNT = 50
DEFAULT_LOOKAHEAD_INTERVAL_SECONDS = 1.0
DEFAULT_SPIKE_CLOUD_MAX_NEURON_COUNT = 20_000
# Small on purpose: this is the exposure window for how much learning a killed process
# (not a clean shutdown - see run_server.py's atexit hook and this module's lifespan
# hook for those) can lose. The saved npz is a few hundred bytes, so saving often is
# effectively free - the interval exists to avoid a save on literally every word, not
# because saves are expensive.
DEFAULT_MEMORY_SAVE_WORD_INTERVAL = 20


def encode_fired_neuron_indices(fired_neuron_indices: torch.Tensor) -> bytes:
    """Encode this frame's fired-neuron indices (see
    ReadingSession.last_frame_fired_neuron_indices) as raw little-endian uint32 bytes -
    the frontend's spike-cloud visualization reads this directly into a
    Uint32Array (see useWebSocketTickData.ts), far cheaper to parse per-frame than JSON
    would be for what can be thousands of indices."""
    return numpy.asarray(fired_neuron_indices.cpu(), dtype="<u4").tobytes()


def frame_result_to_json(frame_result, achieved_words_per_minute: float) -> dict:
    """Convert a FrameResult into a plain JSON-serializable dict, plus the server-
    measured (not session-known - the session only knows simulated time, not real
    wall-clock time) achieved reading pace."""
    return {**asdict(frame_result), "achieved_words_per_minute": achieved_words_per_minute}


def compute_step_count(elapsed_seconds: float, words_per_minute: float, sim_ms_per_word: float, dt_ms: float) -> int:
    """How many raw dt_ms simulation steps to advance for elapsed_seconds of real
    wall-clock time, targeting words_per_minute - each word always gets exactly
    sim_ms_per_word of simulated brain time (never sped up or slowed down: see
    ReadingSession), so "faster" reading means advancing more sim time per real
    second, not compressing a word's own processing. Never negative; returns 0 for
    non-positive elapsed_seconds or words_per_minute.
    """
    if elapsed_seconds <= 0 or words_per_minute <= 0:
        return 0
    target_sim_ms = elapsed_seconds * (words_per_minute / 60.0) * sim_ms_per_word
    return max(0, round(target_sim_ms / dt_ms))


def compute_achieved_words_per_minute(words_delta: int, elapsed_seconds: float) -> float:
    """The REAL reading pace actually delivered this frame (words_delta completed in
    elapsed_seconds of real wall-clock time) - can honestly read below the requested
    words_per_minute if the hardware can't simulate fast enough (see
    scripts/audit_brain.py for measured steps/s on real hardware). Returns 0.0 for
    non-positive elapsed_seconds."""
    if elapsed_seconds <= 0:
        return 0.0
    return (words_delta / elapsed_seconds) * 60.0


async def _run_lookahead_loop(session, word_count: int, interval_seconds: float) -> None:
    """Periodically call session.precompute_upcoming_words() so the multilingual
    embedding model almost never has to run synchronously on the simulation's own
    critical path (see reading_session.py). Runs forever until the task running it is
    cancelled (see create_app's startup handler)."""
    while True:
        await asyncio.to_thread(session.precompute_upcoming_words, word_count)
        await asyncio.sleep(interval_seconds)


class LoadBookRequest(BaseModel):
    """Request body for POST /load-book: an absolute path to a .txt or .epub file."""

    path: str


def _apply_control_message(message: dict, current_autoplay_mode: str, current_words_per_minute: float) -> tuple[str, float]:
    """Parse one client control message and return the resulting (autoplay_mode,
    words_per_minute) - each unchanged unless the message sets it. Pure: neither
    words_per_minute nor autoplay_mode is applied to the session directly anymore -
    words_per_minute only feeds compute_step_count, and autoplay_mode only gates
    _advance_past_finished_book, both called separately by stream_ticks.
    set_paused is handled by the caller too (it gates whether advance() is even
    called, not something the session itself tracks)."""
    message_type = message.get("type")
    if message_type == "set_words_per_minute":
        current_words_per_minute = float(message.get("value", current_words_per_minute))
    elif message_type == "set_autoplay_mode":
        current_autoplay_mode = message.get("mode", AUTOPLAY_MODE_OFF)
    return current_autoplay_mode, current_words_per_minute


def _advance_past_finished_book(
    session, autoplay_mode: str, calibre_library_path: pathlib.Path | None, save_memory_fn: Callable[[object], None] | None
) -> None:
    """React to the book having just finished, per the client's chosen autoplay mode.
    Does nothing when autoplay is off, so a finished book simply stays finished."""
    if autoplay_mode == AUTOPLAY_MODE_RESTART:
        session.restart()
    elif autoplay_mode == AUTOPLAY_MODE_SHUFFLE:
        _load_random_calibre_book(session, calibre_library_path, save_memory_fn)


def _load_random_calibre_book(
    session, calibre_library_path: pathlib.Path | None, save_memory_fn: Callable[[object], None] | None
) -> None:
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
        if save_memory_fn is not None:
            save_memory_fn(session)  # persist the finished book's learning before switching away from it
        session.load_new_text(tokens)
    except (FileNotFoundError, ValueError):
        session.restart()


def create_app(
    session,
    frame_interval_seconds: float = 0.1,
    calibre_library_path: pathlib.Path | None = None,
    load_book_timeout_seconds: float = DEFAULT_LOAD_BOOK_TIMEOUT_SECONDS,
    lookahead_word_count: int = DEFAULT_LOOKAHEAD_WORD_COUNT,
    lookahead_interval_seconds: float = DEFAULT_LOOKAHEAD_INTERVAL_SECONDS,
    spike_cloud_max_neuron_count: int = DEFAULT_SPIKE_CLOUD_MAX_NEURON_COUNT,
    memory_save_word_interval: int = DEFAULT_MEMORY_SAVE_WORD_INTERVAL,
    load_memory_fn: Callable[[object], None] | None = None,
    save_memory_fn: Callable[[object], None] | None = None,
    delete_persisted_memory_fn: Callable[[], None] | None = None,
) -> FastAPI:
    """Build a FastAPI app that streams `session.advance()` results over `/ws` as
    JSON, roughly every frame_interval_seconds of real time, until the client
    disconnects. session.advance() itself runs in a worker thread (asyncio.to_thread)
    so the real LIF/embedding computation never blocks the event loop - control
    messages and /load-book keep working while a frame is mid-simulation. A background
    task periodically calls session.precompute_upcoming_words() so the multilingual
    embedding model almost never has to run synchronously on the simulation's own
    critical path (see reading_session.py).

    load_memory_fn/save_memory_fn/delete_persisted_memory_fn are dependency-injected
    (see brain_loader.load_memory_into_session/save_memory/reset_memory) so this module
    stays testable against a fake session with no real filesystem cache - None (the
    default) skips persistence entirely, matching every test in test_server.py."""

    @contextlib.asynccontextmanager
    async def lifespan(_app: FastAPI):
        if load_memory_fn is not None:
            await asyncio.to_thread(load_memory_fn, session)
        lookahead_task = asyncio.create_task(_run_lookahead_loop(session, lookahead_word_count, lookahead_interval_seconds))
        yield
        lookahead_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await lookahead_task
        if save_memory_fn is not None:
            await asyncio.to_thread(save_memory_fn, session)

    app = FastAPI(lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "tauri://localhost", "http://tauri.localhost"],
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

    @app.websocket("/ws")
    async def stream_ticks(websocket: WebSocket) -> None:
        """Accept the connection, repeatedly advance and send frame results until
        disconnected, and concurrently apply playback control messages (pause, words-
        per-minute, autoplay mode) the client sends on the same socket."""
        await websocket.accept()
        autoplay_mode = AUTOPLAY_MODE_OFF
        words_per_minute = DEFAULT_WORDS_PER_MINUTE
        is_paused = False

        async def receive_control_messages() -> None:
            """Apply each incoming control message until disconnected."""
            nonlocal autoplay_mode, words_per_minute, is_paused
            try:
                while True:
                    message = await websocket.receive_json()
                    if message.get("type") == "set_paused":
                        is_paused = bool(message.get("paused", False))
                    else:
                        autoplay_mode, words_per_minute = _apply_control_message(message, autoplay_mode, words_per_minute)
            except WebSocketDisconnect:
                return

        receiver_task = asyncio.create_task(receive_control_messages())
        try:
            last_frame_time = time.monotonic()
            words_before = None  # None until the first frame has a genuine elapsed-time baseline to compare against
            words_since_last_memory_save = 0
            while True:
                frame_start = time.monotonic()
                elapsed_seconds = frame_start - last_frame_time
                last_frame_time = frame_start

                # While paused, step_count is always 0 - advance(0) is a cheap no-op
                # that still returns a valid (unchanged) FrameResult, so the very first
                # frame works the same whether or not a "set_paused" arrives before it.
                step_count = 0 if is_paused else compute_step_count(elapsed_seconds, words_per_minute, session.sim_ms_per_word, session.dt_ms)
                result = await asyncio.to_thread(session.advance, step_count)
                if result.book_finished:
                    # Level-triggered, not edge-triggered: restart()/load_new_text()
                    # make the very next advance()'s book_finished False again, so this
                    # fires at most once per finish - but it still fires promptly even
                    # if autoplay was only set after the book had already finished.
                    await asyncio.to_thread(
                        _advance_past_finished_book, session, autoplay_mode, calibre_library_path, save_memory_fn
                    )
                # words_before is None only for the very first frame: word_index starts
                # at 0 (words_read=1, even before any real simulated time has elapsed),
                # so there is no genuine "words produced over elapsed_seconds" baseline
                # yet to divide by - report 0.0 rather than a spurious huge/tiny rate.
                words_delta = 0 if words_before is None else max(0, result.words_read - words_before)
                achieved_words_per_minute = (
                    0.0 if words_before is None else compute_achieved_words_per_minute(result.words_read - words_before, elapsed_seconds)
                )
                words_before = result.words_read

                # Periodic persistence, independent of book changes/shutdown (see
                # brain_loader.save_memory) - words_delta is clamped to 0 rather than
                # going negative across a book change (words_read resets to 1 there),
                # so this can only undercount across that one frame, never overcount.
                if save_memory_fn is not None:
                    words_since_last_memory_save += words_delta
                    if words_since_last_memory_save >= memory_save_word_interval:
                        await asyncio.to_thread(save_memory_fn, session)
                        words_since_last_memory_save = 0

                await websocket.send_json(frame_result_to_json(result, achieved_words_per_minute))
                fired_neuron_indices = session.last_frame_fired_neuron_indices(spike_cloud_max_neuron_count)
                await websocket.send_bytes(encode_fired_neuron_indices(fired_neuron_indices))
                elapsed_this_iteration = time.monotonic() - frame_start
                await asyncio.sleep(max(0.0, frame_interval_seconds - elapsed_this_iteration))
        except WebSocketDisconnect:
            return
        finally:
            receiver_task.cancel()

    @app.post("/load-book")
    async def load_book(request: LoadBookRequest) -> dict:
        """Load and tokenize the file at request.path, then feed its tokens into the
        running session as the new book, restarting word progress from the beginning.

        Parsing runs in a worker thread (not directly on the event loop), so a slow or
        pathological book doesn't freeze the live WebSocket stream while it loads, and
        a timeout guarantees this request always resolves instead of hanging the
        frontend's "loading" state forever on a book that never finishes parsing."""
        try:
            tokens = await asyncio.wait_for(
                asyncio.to_thread(load_and_tokenize_file, pathlib.Path(request.path)),
                timeout=load_book_timeout_seconds,
            )
            if save_memory_fn is not None:
                await asyncio.to_thread(save_memory_fn, session)  # persist the outgoing book's learning first
            await asyncio.to_thread(session.load_new_text, tokens)
        except FileNotFoundError as missing_file_error:
            raise HTTPException(status_code=404, detail=str(missing_file_error)) from missing_file_error
        except ValueError as invalid_book_error:
            raise HTTPException(status_code=400, detail=str(invalid_book_error)) from invalid_book_error
        except asyncio.TimeoutError as timeout_error:
            raise HTTPException(
                status_code=504,
                detail=f"Loading this book took longer than {load_book_timeout_seconds:.0f}s",
            ) from timeout_error
        return {"status": "ok", "total_words": len(tokens)}

    @app.post("/reset-memory")
    async def reset_memory() -> dict:
        """Reset the fly's real KC->MBON synapses back to their un-learned initial
        weights (see reading_session.py's ReadingSession.reset_memory) and delete any
        persisted memory.npz, so nothing learned survives a restart either."""
        await asyncio.to_thread(session.reset_memory)
        if delete_persisted_memory_fn is not None:
            await asyncio.to_thread(delete_persisted_memory_fn)
        return {"status": "ok"}

    @app.get("/books-in-folder")
    async def get_books_in_folder(path: str) -> dict:
        """List .epub/.txt files directly inside the given folder, resolved to
        absolute file paths the frontend can pass straight to /load-book. Lets the
        user load an ad-hoc folder of books instead of a full Calibre library."""
        try:
            books = list_books_in_folder(pathlib.Path(path))
        except FileNotFoundError as missing_folder_error:
            raise HTTPException(status_code=404, detail=str(missing_folder_error)) from missing_folder_error
        return {
            "books": [
                {"file_name": book.file_name, "file_path": str(book.file_path)} for book in books
            ]
        }

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
