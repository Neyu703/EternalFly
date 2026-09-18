import math

import pytest
import torch

from eternalfly.emotion_decoder import compute_emotions, compute_rating, pool_rates_to_valence_arousal
from eternalfly.lif import LIFParameters
from eternalfly.reading_session import ReadingSession, ReadingSessionConfig

NEURON_COUNT = 5
ZERO_ADJACENCY = torch.zeros((NEURON_COUNT, NEURON_COUNT))
POOL_INDICES = {
    "sensory_input": torch.tensor([0, 1], dtype=torch.int64),
    "approach": torch.tensor([2], dtype=torch.int64),
    "avoidance": torch.tensor([3], dtype=torch.int64),
    "arousal": torch.tensor([4], dtype=torch.int64),
}
LIF_PARAMETERS = LIFParameters(
    membrane_time_constant_ms=20.0,
    spike_threshold=1.0,
    reset_potential=0.0,
    refractory_period_ms=2.0,
    dt_ms=1.0,
)


def _make_config(**overrides) -> ReadingSessionConfig:
    defaults = dict(
        lif_parameters=LIF_PARAMETERS,
        ticks_per_word=2,
        input_current_scale=0.5,
        token_seed=0,
        engagement_window_size=3,
        engagement_threshold=0.3,
        min_ticks_before_boredom_check=3,
        device="cpu",
    )
    defaults.update(overrides)
    return ReadingSessionConfig(**defaults)


def test_tick_reports_current_word_and_page_progress_advancing_with_ticks_per_word():
    session = ReadingSession(NEURON_COUNT, ZERO_ADJACENCY, POOL_INDICES, ["hello", "world"], _make_config())

    first_tick = session.tick()
    second_tick = session.tick()
    third_tick = session.tick()

    assert first_tick.current_word == "hello"
    assert first_tick.page_progress == pytest.approx(0.5)
    assert second_tick.current_word == "hello"
    assert second_tick.page_progress == pytest.approx(0.5)
    assert third_tick.current_word == "world"
    assert third_tick.page_progress == pytest.approx(1.0)


def test_tick_returns_none_word_and_full_progress_after_book_finished():
    session = ReadingSession(NEURON_COUNT, ZERO_ADJACENCY, POOL_INDICES, ["hello"], _make_config(ticks_per_word=1))

    session.tick()
    after_book_end = session.tick()

    assert after_book_end.current_word is None
    assert after_book_end.page_progress == pytest.approx(1.0)


def test_tick_computes_emotions_and_rating_from_zero_pool_activity():
    session = ReadingSession(NEURON_COUNT, ZERO_ADJACENCY, POOL_INDICES, ["hello", "world"], _make_config())

    tick_result = session.tick()

    expected_valence, expected_arousal = pool_rates_to_valence_arousal(0.0, 0.0, 0.0)
    expected_emotions = compute_emotions(expected_valence, expected_arousal)
    expected_rating = compute_rating(expected_valence)

    assert tick_result.emotions == expected_emotions
    assert tick_result.rating_0_10 == pytest.approx(expected_rating)
    assert tick_result.region_activity == {"approach": 0.0, "avoidance": 0.0, "arousal": 0.0}


def test_tick_sets_wants_new_book_once_engagement_stays_low_long_enough():
    session = ReadingSession(
        NEURON_COUNT, ZERO_ADJACENCY, POOL_INDICES, ["hello", "world", "again"],
        _make_config(ticks_per_word=1, min_ticks_before_boredom_check=3),
    )

    results = [session.tick() for _ in range(4)]

    assert results[0].wants_new_book is False
    assert results[1].wants_new_book is False
    assert results[3].wants_new_book is True


def test_tick_does_not_flag_wants_new_book_before_min_ticks_reached():
    session = ReadingSession(
        NEURON_COUNT, ZERO_ADJACENCY, POOL_INDICES, ["hello"],
        _make_config(ticks_per_word=1, min_ticks_before_boredom_check=1000),
    )

    tick_result = session.tick()

    assert tick_result.wants_new_book is False


def test_pool_spike_rate_of_nan_is_never_produced_even_with_empty_pool():
    empty_pool_indices = dict(POOL_INDICES)
    empty_pool_indices["approach"] = torch.tensor([], dtype=torch.int64)
    session = ReadingSession(NEURON_COUNT, ZERO_ADJACENCY, empty_pool_indices, ["hello"], _make_config(ticks_per_word=1))

    tick_result = session.tick()

    assert not math.isnan(tick_result.region_activity["approach"])
    assert tick_result.region_activity["approach"] == 0.0
