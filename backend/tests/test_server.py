import asyncio

from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from eternalfly.reading_session import TickResult
from eternalfly.server import create_app, tick_result_to_json

SAMPLE_TICK_RESULT = TickResult(
    current_word="hello",
    page_progress=0.5,
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
    wants_new_book=False,
)


def test_tick_result_to_json_returns_dict_with_exact_keys_and_values():
    result = tick_result_to_json(SAMPLE_TICK_RESULT)

    assert result == {
        "current_word": "hello",
        "page_progress": 0.5,
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
        "wants_new_book": False,
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
            emotions={
                "joy": 0.0,
                "trust": 0.0,
                "fear": 0.0,
                "surprise": 0.0,
                "sadness": 0.0,
                "disgust": 0.0,
                "anger": 0.0,
                "anticipation": 0.0,
            },
            rating_0_10=float(self._tick_count),
            region_activity={"approach": 0.0, "avoidance": 0.0, "arousal": 0.0},
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
            emotions={
                "joy": 0.0,
                "trust": 0.0,
                "fear": 0.0,
                "surprise": 0.0,
                "sadness": 0.0,
                "disgust": 0.0,
                "anger": 0.0,
                "anticipation": 0.0,
            },
            rating_0_10=1.0,
            region_activity={"approach": 0.0, "avoidance": 0.0, "arousal": 0.0},
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


def test_ws_route_endpoint_returns_cleanly_when_send_raises_websocket_disconnect():
    fake_session = FakeIncrementingSession()
    app = create_app(fake_session, tick_interval_seconds=0)
    websocket_route = next(route for route in app.routes if getattr(route, "path", None) == "/ws")

    asyncio.run(websocket_route.endpoint(DisconnectingWebSocket()))
