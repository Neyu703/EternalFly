import asyncio
import sqlite3

import ebooklib.epub
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from eternalfly.reading_session import TickResult
from eternalfly.server import create_app, tick_result_to_json

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

SAMPLE_TICK_RESULT = TickResult(
    current_word="hello",
    page_progress=0.5,
    words_read=5,
    total_words=10,
    emotions={
        "joy": 0.1,
        "trust": 0.2,
        "fear": 0.3,
        "surprise": 0.4,
        "sadness": 0.5,
        "disgust": 0.6,
        "anger": 0.7,
        "anticipation": 0.8,
    },
    rating_0_10=6.5,
    region_activity={"approach": 0.1, "avoidance": 0.2, "arousal": 0.3},
    neuropil_activity={"ME_L": 0.4, "MB_CA_R": 0.5},
    wants_new_book=False,
    book_finished=False,
)


def test_tick_result_to_json_returns_dict_with_exact_keys_and_values():
    result = tick_result_to_json(SAMPLE_TICK_RESULT)

    assert result == {
        "current_word": "hello",
        "page_progress": 0.5,
        "words_read": 5,
        "total_words": 10,
        "emotions": {
            "joy": 0.1,
            "trust": 0.2,
            "fear": 0.3,
            "surprise": 0.4,
            "sadness": 0.5,
            "disgust": 0.6,
            "anger": 0.7,
            "anticipation": 0.8,
        },
        "rating_0_10": 6.5,
        "region_activity": {"approach": 0.1, "avoidance": 0.2, "arousal": 0.3},
        "neuropil_activity": {"ME_L": 0.4, "MB_CA_R": 0.5},
        "wants_new_book": False,
        "book_finished": False,
    }


class FakeIncrementingSession:
    """Fake session whose successive tick() calls return distinguishably different results."""

    def __init__(self):
        self._tick_count = 0

    def tick(self) -> TickResult:
        """Return a TickResult whose rating increases by one on each successive call."""
        self._tick_count += 1
        return TickResult(
            current_word=f"word{self._tick_count}",
            page_progress=0.1 * self._tick_count,
            words_read=self._tick_count,
            total_words=10,
            emotions=_ZERO_EMOTIONS,
            rating_0_10=float(self._tick_count),
            region_activity={"approach": 0.0, "avoidance": 0.0, "arousal": 0.0},
            neuropil_activity={},
            wants_new_book=False,
        )


def test_ws_first_message_matches_json_of_first_tick():
    fake_session = FakeIncrementingSession()
    app = create_app(fake_session, tick_interval_seconds=0)
    client = TestClient(app)

    with client.websocket_connect("/ws") as websocket:
        first_message = websocket.receive_json()

    assert first_message == tick_result_to_json(
        TickResult(
            current_word="word1",
            page_progress=0.1,
            words_read=1,
            total_words=10,
            emotions=_ZERO_EMOTIONS,
            rating_0_10=1.0,
            region_activity={"approach": 0.0, "avoidance": 0.0, "arousal": 0.0},
            neuropil_activity={},
            wants_new_book=False,
        )
    )


def test_ws_second_message_reflects_second_tick_call_not_a_cached_first_result():
    fake_session = FakeIncrementingSession()
    app = create_app(fake_session, tick_interval_seconds=0)
    client = TestClient(app)

    with client.websocket_connect("/ws") as websocket:
        first_message = websocket.receive_json()
        second_message = websocket.receive_json()

    assert first_message["current_word"] == "word1"
    assert first_message["rating_0_10"] == 1.0
    assert second_message["current_word"] == "word2"
    assert second_message["rating_0_10"] == 2.0


def test_ws_client_disconnect_mid_loop_does_not_raise_unhandled_exception():
    fake_session = FakeIncrementingSession()
    app = create_app(fake_session, tick_interval_seconds=0)
    client = TestClient(app)

    with client.websocket_connect("/ws") as websocket:
        websocket.receive_json()
        websocket.close()


class DisconnectingWebSocket:
    """Fake WebSocket whose send_json immediately raises WebSocketDisconnect, simulating
    a client that has already gone away by the time the server tries to send a tick."""

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
    app = create_app(fake_session, tick_interval_seconds=0)
    websocket_route = next(route for route in app.routes if getattr(route, "path", None) == "/ws")

    asyncio.run(websocket_route.endpoint(DisconnectingWebSocket()))


class FakeSessionTrackingLoadNewText:
    """Fake session recording load_new_text calls, mirroring ReadingSession's own
    empty-tokens guard so /load-book's error handling can be exercised without a real
    ReadingSession."""

    def __init__(self):
        self.received_tokens = None

    def load_new_text(self, tokens: list[str]) -> None:
        """Record tokens, raising like the real ReadingSession.load_new_text would."""
        if not tokens:
            raise ValueError("tokens must not be empty")
        self.received_tokens = tokens


def test_load_book_endpoint_returns_ok_and_total_words_for_valid_txt_file(tmp_path):
    text_path = tmp_path / "story.txt"
    text_path.write_text("The dragon flew.", encoding="utf-8")
    fake_session = FakeSessionTrackingLoadNewText()
    client = TestClient(create_app(fake_session))

    response = client.post("/load-book", json={"path": str(text_path)})

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "total_words": 3}
    assert fake_session.received_tokens == ["the", "dragon", "flew"]


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
    assert "sunlit" in fake_session.received_tokens
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


class FakeControllableSession:
    """Fake session for exercising playback control messages: records every control
    call it receives, and can be scripted to report the book as finished from a given
    tick onward (mirroring how a real ReadingSession's book_finished stays True)."""

    def __init__(self, finished_on_tick: int | None = None):
        self._tick_count = 0
        self._finished_on_tick = finished_on_tick
        self.paused_calls: list[bool] = []
        self.speed_multiplier_calls: list[float] = []
        self.restart_call_count = 0
        self.received_tokens: list[str] | None = None

    def tick(self) -> TickResult:
        """Return a scripted TickResult, finished from self._finished_on_tick onward."""
        self._tick_count += 1
        book_finished = self._finished_on_tick is not None and self._tick_count >= self._finished_on_tick
        return TickResult(
            current_word=None if book_finished else f"word{self._tick_count}",
            page_progress=1.0 if book_finished else 0.1 * self._tick_count,
            words_read=self._tick_count,
            total_words=10,
            emotions=_ZERO_EMOTIONS,
            rating_0_10=float(self._tick_count),
            region_activity={"approach": 0.0, "avoidance": 0.0, "arousal": 0.0},
            neuropil_activity={},
            wants_new_book=False,
            book_finished=book_finished,
        )

    def set_paused(self, paused: bool) -> None:
        """Record the requested paused state."""
        self.paused_calls.append(paused)

    def set_speed_multiplier(self, multiplier: float) -> None:
        """Record the requested speed multiplier."""
        self.speed_multiplier_calls.append(multiplier)

    def restart(self) -> None:
        """Record the restart and reset word progress, mirroring how the real
        ReadingSession.restart() makes the very next tick's book_finished False again.
        Also disarms further finishing: with tick_interval_seconds=0 the server-side
        loop can race arbitrarily far ahead of the test's receive_json() calls, so a
        test asserting "exactly once" needs the book to never finish a second time
        rather than racing a fixed number of extra ticks against wall-clock luck."""
        self.restart_call_count += 1
        self._tick_count = 0
        self._finished_on_tick = None

    def load_new_text(self, tokens: list[str]) -> None:
        """Record the tokens a shuffle-loaded book was swapped in with, and reset word
        progress like the real ReadingSession.load_new_text() does (see restart()'s
        docstring for why finishing is also disarmed here)."""
        self.received_tokens = tokens
        self._tick_count = 0
        self._finished_on_tick = None


def _drain_websocket_until(websocket, condition, max_messages: int = 500) -> None:
    """Keep receiving tick messages from websocket until condition() is true, raising
    if it isn't met within max_messages ticks (bounds an otherwise-infinite poll loop)."""
    for _ in range(max_messages):
        if condition():
            return
        websocket.receive_json()
    if not condition():
        raise AssertionError(f"condition not met within {max_messages} ticks")


def test_ws_set_paused_control_message_calls_session_set_paused():
    fake_session = FakeControllableSession()
    client = TestClient(create_app(fake_session, tick_interval_seconds=0))

    with client.websocket_connect("/ws") as websocket:
        websocket.send_json({"type": "set_paused", "paused": True})
        _drain_websocket_until(websocket, lambda: True in fake_session.paused_calls)

    assert True in fake_session.paused_calls


def test_ws_set_speed_multiplier_control_message_calls_session_set_speed_multiplier():
    fake_session = FakeControllableSession()
    client = TestClient(create_app(fake_session, tick_interval_seconds=0))

    with client.websocket_connect("/ws") as websocket:
        websocket.send_json({"type": "set_speed_multiplier", "value": 2.5})
        _drain_websocket_until(websocket, lambda: 2.5 in fake_session.speed_multiplier_calls)

    assert 2.5 in fake_session.speed_multiplier_calls


def test_ws_autoplay_mode_off_never_restarts_when_book_finishes():
    fake_session = FakeControllableSession(finished_on_tick=20)
    client = TestClient(create_app(fake_session, tick_interval_seconds=0))

    with client.websocket_connect("/ws") as websocket:
        for _ in range(40):
            websocket.receive_json()

    assert fake_session.restart_call_count == 0


def test_ws_autoplay_mode_restart_restarts_session_exactly_once_when_book_finishes():
    # finished_on_tick=20 gives the receiver task plenty of ticks to apply the mode
    # before the book ever finishes. Once it restarts once, FakeControllableSession
    # disarms further finishing, so however far the send loop races ahead while the
    # test isn't looking, restart_call_count can never exceed 1.
    fake_session = FakeControllableSession(finished_on_tick=20)
    client = TestClient(create_app(fake_session, tick_interval_seconds=0))

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
    fake_session = FakeControllableSession(finished_on_tick=20)
    client = TestClient(create_app(fake_session, tick_interval_seconds=0, calibre_library_path=library_path))

    with client.websocket_connect("/ws") as websocket:
        websocket.send_json({"type": "set_autoplay_mode", "mode": "shuffle"})
        _drain_websocket_until(websocket, lambda: fake_session.received_tokens is not None)

    assert fake_session.received_tokens == ["fear", "is", "the", "mind", "killer"]


def test_ws_autoplay_mode_shuffle_falls_back_to_restart_when_no_library_configured():
    fake_session = FakeControllableSession(finished_on_tick=20)
    client = TestClient(create_app(fake_session, tick_interval_seconds=0))

    with client.websocket_connect("/ws") as websocket:
        websocket.send_json({"type": "set_autoplay_mode", "mode": "shuffle"})
        _drain_websocket_until(websocket, lambda: fake_session.restart_call_count > 0)

    assert fake_session.restart_call_count == 1
    assert fake_session.received_tokens is None
