import asyncio
import contextlib
import sqlite3
import threading
import time

import ebooklib.epub
import pytest
import torch
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from eternalfly.reading_session import FrameResult
from eternalfly.server import (
    _run_lookahead_loop,
    compute_achieved_words_per_minute,
    compute_step_count,
    create_app,
    encode_fired_neuron_indices,
    frame_result_to_json,
)

_ZERO_EMOTIONS = {
    "joy": 0.0,
    "trust": 0.0,
    "fear": 0.0,
    "surprise": 0.0,
    "sadness": 0.0,
    "disgust": 0.0,
    "anger": 0.0,
    "anticipation": 0.0,
}
_ZERO_REGION_ACTIVITY = {"reward": 0.0, "punishment": 0.0, "arousal": 0.0}
_ZERO_BEHAVIORS = {"escape": 0.0, "feeding": 0.0, "backing": 0.0, "turn_left": 0.0, "turn_right": 0.0}

SAMPLE_FRAME_RESULT = FrameResult(
    current_word="hello",
    page_progress=0.5,
    words_read=5,
    total_words=10,
    emotions={
        "joy": 0.1, "trust": 0.2, "fear": 0.3, "surprise": 0.4,
        "sadness": 0.5, "disgust": 0.6, "anger": 0.7, "anticipation": 0.8,
    },
    rating_0_10=6.5,
    learned_valence=0.3,
    region_activity={"reward": 0.1, "punishment": 0.2, "arousal": 0.3},
    behaviors=_ZERO_BEHAVIORS,
    senses={"sweet": 0.4},
    neuropil_activity={"ME_L": 0.4, "MB_CA_R": 0.5},
    steps_simulated=150,
    spikes_per_second=1234.5,
    wants_new_book=False,
    book_finished=False,
)


def _receive_frame(websocket) -> dict:
    """Receive one full frame: the JSON FrameResult message, then the binary spike-
    cloud message stream_ticks always sends right after it (see server.py) - tests
    that only care about the JSON side still need to drain the binary one so the next
    receive_json() doesn't desync onto it."""
    message = websocket.receive_json()
    websocket.receive_bytes()
    return message


def test_frame_result_to_json_returns_dict_with_exact_keys_and_values_plus_achieved_wpm():
    result = frame_result_to_json(SAMPLE_FRAME_RESULT, achieved_words_per_minute=142.0)

    assert result == {
        "current_word": "hello",
        "page_progress": 0.5,
        "words_read": 5,
        "total_words": 10,
        "emotions": {
            "joy": 0.1, "trust": 0.2, "fear": 0.3, "surprise": 0.4,
            "sadness": 0.5, "disgust": 0.6, "anger": 0.7, "anticipation": 0.8,
        },
        "rating_0_10": 6.5,
        "learned_valence": 0.3,
        "region_activity": {"reward": 0.1, "punishment": 0.2, "arousal": 0.3},
        "behaviors": _ZERO_BEHAVIORS,
        "senses": {"sweet": 0.4},
        "neuropil_activity": {"ME_L": 0.4, "MB_CA_R": 0.5},
        "steps_simulated": 150,
        "spikes_per_second": 1234.5,
        "wants_new_book": False,
        "book_finished": False,
        "achieved_words_per_minute": 142.0,
    }


def test_compute_step_count_targets_the_requested_words_per_minute():
    # 1 second elapsed, 60 wpm (1 word/s), sim_ms_per_word=150, dt_ms=1
    # -> target_sim_ms = 1 * 1 * 150 = 150 steps.
    assert compute_step_count(elapsed_seconds=1.0, words_per_minute=60.0, sim_ms_per_word=150.0, dt_ms=1.0) == 150


def test_compute_step_count_scales_with_elapsed_time():
    assert compute_step_count(elapsed_seconds=2.0, words_per_minute=60.0, sim_ms_per_word=150.0, dt_ms=1.0) == 300


def test_compute_step_count_returns_zero_for_non_positive_elapsed_seconds():
    assert compute_step_count(elapsed_seconds=0.0, words_per_minute=150.0, sim_ms_per_word=150.0, dt_ms=1.0) == 0
    assert compute_step_count(elapsed_seconds=-1.0, words_per_minute=150.0, sim_ms_per_word=150.0, dt_ms=1.0) == 0


def test_compute_step_count_returns_zero_for_non_positive_words_per_minute():
    assert compute_step_count(elapsed_seconds=1.0, words_per_minute=0.0, sim_ms_per_word=150.0, dt_ms=1.0) == 0


def test_compute_step_count_respects_dt_ms_larger_than_one():
    # target_sim_ms = 1 * 1 * 150 = 150, dt_ms=2 -> 75 steps.
    assert compute_step_count(elapsed_seconds=1.0, words_per_minute=60.0, sim_ms_per_word=150.0, dt_ms=2.0) == 75


def test_compute_achieved_words_per_minute_converts_words_delta_to_per_minute_rate():
    # 5 words in 2 seconds -> 2.5 words/s -> 150 words/min.
    assert compute_achieved_words_per_minute(words_delta=5, elapsed_seconds=2.0) == pytest.approx(150.0)


def test_compute_achieved_words_per_minute_returns_zero_for_non_positive_elapsed_seconds():
    assert compute_achieved_words_per_minute(words_delta=5, elapsed_seconds=0.0) == 0.0


def test_encode_fired_neuron_indices_round_trips_via_numpy_uint32():
    import numpy

    encoded = encode_fired_neuron_indices(torch.tensor([1, 5, 139254], dtype=torch.int64))

    decoded = numpy.frombuffer(encoded, dtype="<u4")
    assert decoded.tolist() == [1, 5, 139254]


def test_encode_fired_neuron_indices_empty_tensor_encodes_to_empty_bytes():
    encoded = encode_fired_neuron_indices(torch.tensor([], dtype=torch.int64))

    assert encoded == b""


def test_run_lookahead_loop_calls_precompute_upcoming_words_repeatedly():
    calls: list[int] = []

    class FakeLookaheadSession:
        def precompute_upcoming_words(self, word_count: int) -> None:
            calls.append(word_count)

    async def run_a_few_iterations() -> None:
        task = asyncio.create_task(_run_lookahead_loop(FakeLookaheadSession(), word_count=7, interval_seconds=0))
        while len(calls) < 3:
            await asyncio.sleep(0)
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    asyncio.run(run_a_few_iterations())

    assert calls[:3] == [7, 7, 7]


class FakeIncrementingSession:
    """Fake session whose successive advance() calls return distinguishably different
    results, ignoring step_count's actual value (real elapsed-time-derived step counts
    aren't deterministic enough to assert on at this level - see test_server.py's
    compute_step_count tests for that math in isolation)."""

    sim_ms_per_word = 150.0
    dt_ms = 1.0

    def __init__(self):
        self._call_count = 0
        self.received_tokens: list[str] | None = None

    def load_new_text(self, tokens: list[str]) -> None:
        """Record tokens, so this fake can also exercise /load-book alongside /ws."""
        self.received_tokens = tokens

    def precompute_upcoming_words(self, word_count: int) -> None:
        """No-op: exercised by create_app's background lookahead loop."""

    def last_frame_fired_neuron_indices(self, max_count: int) -> torch.Tensor:
        """Return a distinguishable fired-neuron index per call, so the spike-cloud
        binary message can be checked to actually reflect the latest advance() call."""
        return torch.tensor([self._call_count], dtype=torch.int64)

    def advance(self, step_count: int) -> FrameResult:
        """Return a FrameResult whose rating increases by one on each successive call."""
        self._call_count += 1
        return FrameResult(
            current_word=f"word{self._call_count}",
            page_progress=0.1 * self._call_count,
            words_read=self._call_count,
            total_words=10,
            emotions=_ZERO_EMOTIONS,
            rating_0_10=float(self._call_count),
            learned_valence=0.0,
            region_activity=_ZERO_REGION_ACTIVITY,
            behaviors=_ZERO_BEHAVIORS,
            senses={},
            neuropil_activity={},
            steps_simulated=step_count,
            spikes_per_second=0.0,
            wants_new_book=False,
        )


def test_ws_first_message_reflects_first_advance_call():
    fake_session = FakeIncrementingSession()
    app = create_app(fake_session, frame_interval_seconds=0)
    client = TestClient(app)

    with client.websocket_connect("/ws") as websocket:
        first_message = _receive_frame(websocket)

    assert first_message["current_word"] == "word1"
    assert first_message["rating_0_10"] == 1.0
    assert first_message["total_words"] == 10
    # Regression check: the very first frame has no genuine elapsed-time baseline yet
    # (see stream_ticks) - it must report 0.0, never a spurious huge/tiny rate from
    # dividing by an almost-zero elapsed_seconds.
    assert first_message["achieved_words_per_minute"] == 0.0


def test_ws_second_message_reflects_second_advance_call_not_a_cached_first_result():
    fake_session = FakeIncrementingSession()
    app = create_app(fake_session, frame_interval_seconds=0)
    client = TestClient(app)

    with client.websocket_connect("/ws") as websocket:
        first_message = _receive_frame(websocket)
        second_message = _receive_frame(websocket)

    assert first_message["current_word"] == "word1"
    assert second_message["current_word"] == "word2"
    assert second_message["rating_0_10"] == 2.0


def test_ws_sends_the_spike_cloud_binary_message_right_after_each_json_frame():
    """End-to-end check that stream_ticks pairs every JSON frame with the binary
    fired-neuron-indices message from the same advance() call, in order - not a stale
    or swapped one (see encode_fired_neuron_indices's unit tests for the encoding
    itself, and last_frame_fired_neuron_indices's own tests in test_reading_session.py
    for where the indices come from)."""
    import numpy

    fake_session = FakeIncrementingSession()
    app = create_app(fake_session, frame_interval_seconds=0)
    client = TestClient(app)

    with client.websocket_connect("/ws") as websocket:
        first_json = websocket.receive_json()
        first_bytes = websocket.receive_bytes()
        second_json = websocket.receive_json()
        second_bytes = websocket.receive_bytes()

    assert first_json["current_word"] == "word1"
    assert numpy.frombuffer(first_bytes, dtype="<u4").tolist() == [1]
    assert second_json["current_word"] == "word2"
    assert numpy.frombuffer(second_bytes, dtype="<u4").tolist() == [2]


def test_app_startup_launches_the_background_lookahead_loop():
    fake_session = FakeIncrementingSession()
    calls = []
    fake_session.precompute_upcoming_words = lambda word_count: calls.append(word_count)
    app = create_app(fake_session, lookahead_word_count=9, lookahead_interval_seconds=0)

    # FastAPI's startup event (which schedules _run_lookahead_loop) only fires when
    # TestClient is used as a context manager - every other test in this file uses it
    # directly, deliberately not triggering startup (irrelevant to what they check).
    with TestClient(app) as client:
        with client.websocket_connect("/ws") as websocket:
            _receive_frame(websocket)

    assert 9 in calls


def test_ws_client_disconnect_mid_loop_does_not_raise_unhandled_exception():
    fake_session = FakeIncrementingSession()
    app = create_app(fake_session, frame_interval_seconds=0)
    client = TestClient(app)

    with client.websocket_connect("/ws") as websocket:
        _receive_frame(websocket)
        websocket.close()


class DisconnectingWebSocket:
    """Fake WebSocket whose send_json immediately raises WebSocketDisconnect, simulating
    a client that has already gone away by the time the server tries to send a frame."""

    async def accept(self) -> None:
        """No-op accept, mirroring the real WebSocket's accept() signature."""

    async def send_json(self, data: dict) -> None:
        """Simulate the transport discovering the client disconnected mid-send."""
        raise WebSocketDisconnect()

    async def receive_json(self) -> dict:
        """Simulate the transport discovering the client is already gone on receive too."""
        raise WebSocketDisconnect()


def test_ws_route_endpoint_returns_cleanly_when_send_raises_websocket_disconnect():
    fake_session = FakeIncrementingSession()
    app = create_app(fake_session, frame_interval_seconds=0)
    websocket_route = next(route for route in app.routes if getattr(route, "path", None) == "/ws")

    asyncio.run(websocket_route.endpoint(DisconnectingWebSocket()))


class FakeSessionTrackingLoadNewText:
    """Fake session recording load_new_text calls, mirroring ReadingSession's own
    empty-tokens guard so /load-book's error handling can be exercised without a real
    ReadingSession."""

    sim_ms_per_word = 150.0
    dt_ms = 1.0

    def __init__(self):
        self.received_tokens = None

    def load_new_text(self, tokens: list[str]) -> None:
        """Record tokens, raising like the real ReadingSession.load_new_text would."""
        if not tokens:
            raise ValueError("tokens must not be empty")
        self.received_tokens = tokens

    def precompute_upcoming_words(self, word_count: int) -> None:
        """No-op: exercised by create_app's background lookahead loop."""


def test_load_book_endpoint_returns_ok_and_total_words_for_valid_txt_file(tmp_path):
    text_path = tmp_path / "story.txt"
    text_path.write_text("The dragon flew.", encoding="utf-8")
    fake_session = FakeSessionTrackingLoadNewText()
    client = TestClient(create_app(fake_session))

    response = client.post("/load-book", json={"path": str(text_path)})

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "total_words": 3}
    assert fake_session.received_tokens == ["The", "dragon", "flew"]


def test_load_book_endpoint_returns_ok_for_valid_epub_file(tmp_path):
    epub_path = tmp_path / "tiny.epub"
    book = ebooklib.epub.EpubBook()
    book.set_identifier("test-id-123")
    book.set_title("Tiny Test Book")
    book.set_language("en")
    chapter = ebooklib.epub.EpubHtml(title="Chapter One", file_name="chapter_one.xhtml", lang="en")
    chapter.content = "<html><body><p>Sunlit meadows.</p></body></html>"
    book.add_item(chapter)
    book.toc = (chapter,)
    book.add_item(ebooklib.epub.EpubNcx())
    book.add_item(ebooklib.epub.EpubNav())
    book.spine = ["nav", chapter]
    ebooklib.epub.write_epub(str(epub_path), book)
    fake_session = FakeSessionTrackingLoadNewText()
    client = TestClient(create_app(fake_session))

    response = client.post("/load-book", json={"path": str(epub_path)})

    assert response.status_code == 200
    assert "Sunlit" in fake_session.received_tokens
    assert "meadows" in fake_session.received_tokens
    assert response.json() == {"status": "ok", "total_words": len(fake_session.received_tokens)}


def test_load_book_endpoint_returns_404_for_missing_file(tmp_path):
    missing_path = tmp_path / "missing.txt"
    fake_session = FakeSessionTrackingLoadNewText()
    client = TestClient(create_app(fake_session))

    response = client.post("/load-book", json={"path": str(missing_path)})

    assert response.status_code == 404
    assert fake_session.received_tokens is None


def test_load_book_endpoint_returns_400_for_unsupported_file_extension(tmp_path):
    unsupported_path = tmp_path / "story.pdf"
    unsupported_path.write_text("irrelevant", encoding="utf-8")
    fake_session = FakeSessionTrackingLoadNewText()
    client = TestClient(create_app(fake_session))

    response = client.post("/load-book", json={"path": str(unsupported_path)})

    assert response.status_code == 400
    assert fake_session.received_tokens is None


def test_load_book_endpoint_returns_400_when_file_tokenizes_to_an_empty_book(tmp_path):
    empty_text_path = tmp_path / "empty.txt"
    empty_text_path.write_text("--- ... !!!", encoding="utf-8")
    fake_session = FakeSessionTrackingLoadNewText()
    client = TestClient(create_app(fake_session))

    response = client.post("/load-book", json={"path": str(empty_text_path)})

    assert response.status_code == 400
    assert fake_session.received_tokens is None


def test_load_book_endpoint_returns_504_when_parsing_exceeds_the_timeout(tmp_path, monkeypatch):
    slow_path = tmp_path / "slow.txt"
    slow_path.write_text("word", encoding="utf-8")

    def slow_load_and_tokenize_file(path):
        """Stand in for a pathological book whose parsing never finishes in time."""
        time.sleep(1)
        return ["word"]

    monkeypatch.setattr("eternalfly.server.load_and_tokenize_file", slow_load_and_tokenize_file)
    fake_session = FakeSessionTrackingLoadNewText()
    client = TestClient(create_app(fake_session, load_book_timeout_seconds=0.05))

    response = client.post("/load-book", json={"path": str(slow_path)})

    assert response.status_code == 504
    assert fake_session.received_tokens is None


def test_load_book_endpoint_does_not_block_the_event_loop_while_parsing(tmp_path, monkeypatch):
    slow_path = tmp_path / "slow.txt"
    slow_path.write_text("word", encoding="utf-8")

    def slow_load_and_tokenize_file(path):
        """A blocking (non-async) call, matching real epub/txt parsing's sync API."""
        time.sleep(0.2)
        return ["word"]

    monkeypatch.setattr("eternalfly.server.load_and_tokenize_file", slow_load_and_tokenize_file)
    fake_session = FakeIncrementingSession()
    client = TestClient(create_app(fake_session, frame_interval_seconds=0))

    with client.websocket_connect("/ws") as websocket:
        _receive_frame(websocket)
        load_book_thread = threading.Thread(
            target=lambda: client.post("/load-book", json={"path": str(slow_path)})
        )
        load_book_thread.start()
        # If /load-book blocked the event loop, this would hang until the slow parse
        # finishes; it should instead keep receiving frames the whole time.
        second_frame = _receive_frame(websocket)
        load_book_thread.join(timeout=5)

    assert second_frame["current_word"] == "word2"
    assert not load_book_thread.is_alive()


def _create_minimal_calibre_library(tmp_path):
    """Build a minimal real Calibre-shaped metadata.db with a single EPUB book."""
    library_path = tmp_path / "calibre_library"
    library_path.mkdir()
    connection = sqlite3.connect(library_path / "metadata.db")
    connection.executescript(
        """
        CREATE TABLE books (id INTEGER PRIMARY KEY, title TEXT, path TEXT);
        CREATE TABLE authors (id INTEGER PRIMARY KEY, name TEXT);
        CREATE TABLE books_authors_link (book INTEGER, author INTEGER);
        CREATE TABLE data (id INTEGER PRIMARY KEY, book INTEGER, format TEXT, name TEXT);
        """
    )
    connection.execute("INSERT INTO books VALUES (1, 'Dune', 'Frank Herbert/Dune (1)')")
    connection.execute("INSERT INTO authors VALUES (1, 'Frank Herbert')")
    connection.execute("INSERT INTO books_authors_link VALUES (1, 1)")
    connection.execute("INSERT INTO data VALUES (1, 1, 'EPUB', 'dune')")
    connection.commit()
    connection.close()
    return library_path


def test_calibre_books_endpoint_returns_empty_list_when_no_library_configured():
    fake_session = FakeSessionTrackingLoadNewText()
    client = TestClient(create_app(fake_session))

    response = client.get("/calibre-books")

    assert response.status_code == 200
    assert response.json() == {"books": []}


def test_calibre_books_endpoint_returns_books_from_configured_library(tmp_path):
    library_path = _create_minimal_calibre_library(tmp_path)
    fake_session = FakeSessionTrackingLoadNewText()
    client = TestClient(create_app(fake_session, calibre_library_path=library_path))

    response = client.get("/calibre-books")

    assert response.status_code == 200
    assert response.json() == {
        "books": [
            {
                "book_id": 1,
                "title": "Dune",
                "author": "Frank Herbert",
                "file_path": str(library_path / "Frank Herbert/Dune (1)" / "dune.epub"),
            }
        ]
    }


def test_calibre_books_endpoint_returns_404_when_configured_library_path_has_no_metadata_db(tmp_path):
    missing_library_path = tmp_path / "no_such_library"
    fake_session = FakeSessionTrackingLoadNewText()
    client = TestClient(create_app(fake_session, calibre_library_path=missing_library_path))

    response = client.get("/calibre-books")

    assert response.status_code == 404


def test_books_in_folder_endpoint_returns_epub_and_txt_files_in_the_given_folder(tmp_path):
    (tmp_path / "Zebra.epub").write_bytes(b"")
    (tmp_path / "apple.txt").write_bytes(b"")
    (tmp_path / "cover.jpg").write_bytes(b"")
    fake_session = FakeSessionTrackingLoadNewText()
    client = TestClient(create_app(fake_session))

    response = client.get("/books-in-folder", params={"path": str(tmp_path)})

    assert response.status_code == 200
    assert response.json() == {
        "books": [
            {"file_name": "apple.txt", "file_path": str(tmp_path / "apple.txt")},
            {"file_name": "Zebra.epub", "file_path": str(tmp_path / "Zebra.epub")},
        ]
    }


def test_books_in_folder_endpoint_returns_404_for_a_missing_folder(tmp_path):
    missing_folder = tmp_path / "does_not_exist"
    fake_session = FakeSessionTrackingLoadNewText()
    client = TestClient(create_app(fake_session))

    response = client.get("/books-in-folder", params={"path": str(missing_folder)})

    assert response.status_code == 404


class FakeControllableSession:
    """Fake session for exercising playback control: records every advance() call, and
    can be scripted to report the book as finished from a given call onward (mirroring
    how a real ReadingSession's book_finished stays True)."""

    sim_ms_per_word = 150.0
    dt_ms = 1.0

    def __init__(self, finished_on_call: int | None = None):
        self._call_count = 0
        self._finished_on_call = finished_on_call
        self.advance_step_counts: list[int] = []
        self.restart_call_count = 0
        self.received_tokens: list[str] | None = None
        self.reset_memory_call_count = 0

    def advance(self, step_count: int) -> FrameResult:
        """Return a scripted FrameResult, finished from self._finished_on_call onward."""
        self.advance_step_counts.append(step_count)
        self._call_count += 1
        book_finished = self._finished_on_call is not None and self._call_count >= self._finished_on_call
        return FrameResult(
            current_word=None if book_finished else f"word{self._call_count}",
            page_progress=1.0 if book_finished else 0.1 * self._call_count,
            words_read=self._call_count,
            total_words=10,
            emotions=_ZERO_EMOTIONS,
            rating_0_10=float(self._call_count),
            learned_valence=0.0,
            region_activity=_ZERO_REGION_ACTIVITY,
            behaviors=_ZERO_BEHAVIORS,
            senses={},
            neuropil_activity={},
            steps_simulated=step_count,
            spikes_per_second=0.0,
            wants_new_book=False,
            book_finished=book_finished,
        )

    def restart(self) -> None:
        """Record the restart and reset progress, mirroring how the real
        ReadingSession.restart() makes the very next advance()'s book_finished False
        again. Also disarms further finishing: with frame_interval_seconds=0 the
        server-side loop can race arbitrarily far ahead of the test's receive_json()
        calls, so a test asserting "exactly once" needs the book to never finish a
        second time rather than racing a fixed number of extra frames against luck."""
        self.restart_call_count += 1
        self._call_count = 0
        self._finished_on_call = None

    def load_new_text(self, tokens: list[str]) -> None:
        """Record the tokens a shuffle-loaded book was swapped in with, and reset
        progress like the real ReadingSession.load_new_text() does (see restart()'s
        docstring for why finishing is also disarmed here)."""
        self.received_tokens = tokens
        self._call_count = 0
        self._finished_on_call = None

    def precompute_upcoming_words(self, word_count: int) -> None:
        """No-op: exercised by create_app's background lookahead loop."""

    def last_frame_fired_neuron_indices(self, max_count: int) -> torch.Tensor:
        """Return an empty tensor - this fake's control-message tests don't care about
        the spike-cloud binary message's contents, only that it doesn't crash."""
        return torch.tensor([], dtype=torch.int64)

    def reset_memory(self) -> None:
        """Record the reset, mirroring how the real ReadingSession.reset_memory() has
        no return value - see POST /reset-memory."""
        self.reset_memory_call_count += 1


def _drain_websocket_until(websocket, condition, max_messages: int = 500) -> None:
    """Keep receiving frame messages from websocket until condition() is true, raising
    if it isn't met within max_messages frames (bounds an otherwise-infinite poll loop)."""
    for _ in range(max_messages):
        if condition():
            return
        _receive_frame(websocket)
    if not condition():
        raise AssertionError(f"condition not met within {max_messages} frames")


def test_ws_set_paused_advances_with_zero_step_count_only():
    fake_session = FakeControllableSession()
    client = TestClient(create_app(fake_session, frame_interval_seconds=0))

    with client.websocket_connect("/ws") as websocket:
        _receive_frame(websocket)
        websocket.send_json({"type": "set_paused", "paused": True})
        # Drain enough frames to let the pause message actually be received and
        # applied (frame_interval_seconds=0 means the send loop can race ahead of the
        # receiver task), then check every advance() call from here on used step_count=0.
        for _ in range(20):
            _receive_frame(websocket)
        step_counts_once_paused = len(fake_session.advance_step_counts)

        for _ in range(20):
            _receive_frame(websocket)

    assert all(step_count == 0 for step_count in fake_session.advance_step_counts[step_counts_once_paused:])


def test_ws_set_words_per_minute_control_message_is_accepted_without_error():
    fake_session = FakeControllableSession()
    client = TestClient(create_app(fake_session, frame_interval_seconds=0))

    with client.websocket_connect("/ws") as websocket:
        websocket.send_json({"type": "set_words_per_minute", "value": 600.0})
        first_message = _receive_frame(websocket)

    assert first_message["current_word"] == "word1"


def test_ws_autoplay_mode_off_never_restarts_when_book_finishes():
    fake_session = FakeControllableSession(finished_on_call=20)
    client = TestClient(create_app(fake_session, frame_interval_seconds=0))

    with client.websocket_connect("/ws") as websocket:
        for _ in range(40):
            _receive_frame(websocket)

    assert fake_session.restart_call_count == 0


def test_ws_autoplay_mode_restart_restarts_session_exactly_once_when_book_finishes():
    fake_session = FakeControllableSession(finished_on_call=20)
    client = TestClient(create_app(fake_session, frame_interval_seconds=0))

    with client.websocket_connect("/ws") as websocket:
        websocket.send_json({"type": "set_autoplay_mode", "mode": "restart"})
        _drain_websocket_until(websocket, lambda: fake_session.restart_call_count > 0)

    assert fake_session.restart_call_count == 1


def _create_minimal_calibre_library_with_real_txt_book(tmp_path):
    """Build a minimal real Calibre-shaped metadata.db whose single book is an actual
    loadable .txt file on disk (unlike _create_minimal_calibre_library's placeholder
    row), so autoplay shuffle can genuinely load it."""
    library_path = tmp_path / "calibre_library"
    book_dir = library_path / "Frank Herbert" / "Dune (1)"
    book_dir.mkdir(parents=True)
    (book_dir / "dune.txt").write_text("Fear is the mind killer.", encoding="utf-8")
    connection = sqlite3.connect(library_path / "metadata.db")
    connection.executescript(
        """
        CREATE TABLE books (id INTEGER PRIMARY KEY, title TEXT, path TEXT);
        CREATE TABLE authors (id INTEGER PRIMARY KEY, name TEXT);
        CREATE TABLE books_authors_link (book INTEGER, author INTEGER);
        CREATE TABLE data (id INTEGER PRIMARY KEY, book INTEGER, format TEXT, name TEXT);
        """
    )
    connection.execute("INSERT INTO books VALUES (1, 'Dune', 'Frank Herbert/Dune (1)')")
    connection.execute("INSERT INTO authors VALUES (1, 'Frank Herbert')")
    connection.execute("INSERT INTO books_authors_link VALUES (1, 1)")
    connection.execute("INSERT INTO data VALUES (1, 1, 'TXT', 'dune')")
    connection.commit()
    connection.close()
    return library_path


def test_ws_autoplay_mode_shuffle_loads_a_calibre_book_when_book_finishes(tmp_path):
    library_path = _create_minimal_calibre_library_with_real_txt_book(tmp_path)
    fake_session = FakeControllableSession(finished_on_call=20)
    client = TestClient(create_app(fake_session, frame_interval_seconds=0, calibre_library_path=library_path))

    with client.websocket_connect("/ws") as websocket:
        websocket.send_json({"type": "set_autoplay_mode", "mode": "shuffle"})
        _drain_websocket_until(websocket, lambda: fake_session.received_tokens is not None)

    assert fake_session.received_tokens == ["Fear", "is", "the", "mind", "killer"]


def test_ws_autoplay_mode_shuffle_falls_back_to_restart_when_no_library_configured():
    fake_session = FakeControllableSession(finished_on_call=20)
    client = TestClient(create_app(fake_session, frame_interval_seconds=0))

    with client.websocket_connect("/ws") as websocket:
        websocket.send_json({"type": "set_autoplay_mode", "mode": "shuffle"})
        _drain_websocket_until(websocket, lambda: fake_session.restart_call_count > 0)

    assert fake_session.restart_call_count == 1
    assert fake_session.received_tokens is None


def test_ws_autoplay_mode_shuffle_saves_memory_before_loading_the_new_book(tmp_path):
    library_path = _create_minimal_calibre_library_with_real_txt_book(tmp_path)
    fake_session = FakeControllableSession(finished_on_call=20)
    saved_sessions = []
    client = TestClient(
        create_app(
            fake_session,
            frame_interval_seconds=0,
            calibre_library_path=library_path,
            save_memory_fn=saved_sessions.append,
        )
    )

    with client.websocket_connect("/ws") as websocket:
        websocket.send_json({"type": "set_autoplay_mode", "mode": "shuffle"})
        _drain_websocket_until(websocket, lambda: fake_session.received_tokens is not None)

    assert saved_sessions == [fake_session]


def test_load_book_endpoint_saves_memory_before_loading_the_new_book(tmp_path):
    text_path = tmp_path / "story.txt"
    text_path.write_text("The dragon flew.", encoding="utf-8")
    fake_session = FakeSessionTrackingLoadNewText()
    saved_sessions = []
    client = TestClient(create_app(fake_session, save_memory_fn=saved_sessions.append))

    client.post("/load-book", json={"path": str(text_path)})

    assert saved_sessions == [fake_session]


def test_load_book_endpoint_does_not_save_memory_when_no_save_memory_fn_is_configured(tmp_path):
    text_path = tmp_path / "story.txt"
    text_path.write_text("The dragon flew.", encoding="utf-8")
    fake_session = FakeSessionTrackingLoadNewText()
    client = TestClient(create_app(fake_session))

    response = client.post("/load-book", json={"path": str(text_path)})

    assert response.status_code == 200


def test_lifespan_loads_memory_on_startup():
    fake_session = FakeIncrementingSession()
    loaded_sessions = []
    app = create_app(fake_session, load_memory_fn=loaded_sessions.append)

    with TestClient(app) as client:
        with client.websocket_connect("/ws") as websocket:
            _receive_frame(websocket)

    assert loaded_sessions == [fake_session]


def test_lifespan_saves_memory_on_shutdown():
    fake_session = FakeIncrementingSession()
    saved_sessions = []
    app = create_app(fake_session, save_memory_fn=saved_sessions.append)

    with TestClient(app) as client:
        with client.websocket_connect("/ws") as websocket:
            _receive_frame(websocket)

    assert saved_sessions == [fake_session]


def test_ws_saves_memory_periodically_once_the_word_interval_is_crossed():
    fake_session = FakeIncrementingSession()
    saved_sessions = []
    client = TestClient(
        create_app(
            fake_session,
            frame_interval_seconds=0,
            memory_save_word_interval=2,
            save_memory_fn=saved_sessions.append,
        )
    )

    with client.websocket_connect("/ws") as websocket:
        _drain_websocket_until(websocket, lambda: len(saved_sessions) > 0)

    assert saved_sessions == [fake_session]


def test_reset_memory_endpoint_resets_session_and_deletes_persisted_file():
    fake_session = FakeControllableSession()
    deleted_call_count = []
    client = TestClient(create_app(fake_session, delete_persisted_memory_fn=lambda: deleted_call_count.append(1)))

    response = client.post("/reset-memory")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert fake_session.reset_memory_call_count == 1
    assert deleted_call_count == [1]


def test_reset_memory_endpoint_works_without_a_delete_persisted_memory_fn_configured():
    fake_session = FakeControllableSession()
    client = TestClient(create_app(fake_session))

    response = client.post("/reset-memory")

    assert response.status_code == 200
    assert fake_session.reset_memory_call_count == 1
