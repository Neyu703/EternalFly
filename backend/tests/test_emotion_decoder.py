import pytest

from eternalfly.emotion_decoder import (
    ExponentialMovingAverage,
    compute_emotions,
    compute_rating,
    pool_rates_to_valence_arousal,
)


def test_exponential_moving_average_raises_on_non_positive_time_constant():
    with pytest.raises(ValueError, match="time_constant_ms"):
        ExponentialMovingAverage(time_constant_ms=0.0)


def test_exponential_moving_average_raises_on_non_positive_elapsed_ms():
    average = ExponentialMovingAverage(time_constant_ms=100.0)
    with pytest.raises(ValueError, match="elapsed_ms"):
        average.update(1.0, elapsed_ms=0.0)


def test_exponential_moving_average_first_update_returns_that_single_value():
    average = ExponentialMovingAverage(time_constant_ms=100.0)
    assert average.update(4.0, elapsed_ms=1.0) == 4.0


def test_exponential_moving_average_moves_toward_a_constant_input_over_time():
    average = ExponentialMovingAverage(time_constant_ms=10.0)
    average.update(0.0, elapsed_ms=1.0)
    for _ in range(200):
        result = average.update(1.0, elapsed_ms=1.0)
    assert result == pytest.approx(1.0, abs=1e-6)


def test_exponential_moving_average_decays_by_the_expected_factor_after_one_time_constant():
    import math

    time_constant_ms, dt_ms = 20.0, 1.0
    average = ExponentialMovingAverage(time_constant_ms=time_constant_ms)
    average.update(1.0, elapsed_ms=dt_ms)
    steps = round(time_constant_ms / dt_ms)
    result = 1.0
    for _ in range(steps):
        result = average.update(0.0, elapsed_ms=dt_ms)
    # After one time constant of decaying toward 0, an EMA should sit near 1/e of its start.
    assert result == pytest.approx(math.exp(-1.0), abs=0.02)


def test_exponential_moving_average_smaller_dt_relative_to_time_constant_still_converges():
    average = ExponentialMovingAverage(time_constant_ms=10.0)
    average.update(0.0, elapsed_ms=0.5)
    for _ in range(400):
        result = average.update(1.0, elapsed_ms=0.5)
    assert result == pytest.approx(1.0, abs=1e-6)


def test_exponential_moving_average_a_single_large_elapsed_ms_update_matches_many_small_ones():
    # A frame spanning 20ms in one update() call must behave like 20 separate 1ms
    # updates toward the same target value (this is what lets reading_session.py call
    # update() once per variable-sized frame instead of once per raw simulation step).
    time_constant_ms = 15.0
    single_call_average = ExponentialMovingAverage(time_constant_ms)
    single_call_average.update(0.0, elapsed_ms=1.0)
    result_single = single_call_average.update(1.0, elapsed_ms=20.0)

    many_calls_average = ExponentialMovingAverage(time_constant_ms)
    many_calls_average.update(0.0, elapsed_ms=1.0)
    result_many = 0.0
    for _ in range(20):
        result_many = many_calls_average.update(1.0, elapsed_ms=1.0)

    assert result_single == pytest.approx(result_many, abs=1e-9)


def test_pool_rates_to_valence_arousal_scales_positive_valence_by_its_own_ceiling():
    valence, arousal = pool_rates_to_valence_arousal(
        approach_pool_rate=0.3,
        avoidance_pool_rate=0.1,
        arousal_pool_rate=0.5,
        positive_valence_ceiling=0.4,
        negative_valence_ceiling=0.1,
        arousal_ceiling=1.0,
    )
    assert valence == pytest.approx(0.5)  # raw 0.2 / positive ceiling 0.4
    assert arousal == pytest.approx(0.5)


def test_pool_rates_to_valence_arousal_scales_negative_valence_by_its_own_ceiling():
    valence, _ = pool_rates_to_valence_arousal(
        approach_pool_rate=0.1,
        avoidance_pool_rate=0.3,
        arousal_pool_rate=0.0,
        positive_valence_ceiling=0.4,
        negative_valence_ceiling=0.1,
        arousal_ceiling=1.0,
    )
    assert valence == pytest.approx(-1.0)  # raw -0.2 / negative ceiling 0.1, clamped to -1.0


def test_pool_rates_to_valence_arousal_clamps_valence_and_arousal_to_their_bounds():
    valence, arousal = pool_rates_to_valence_arousal(
        approach_pool_rate=1.0,
        avoidance_pool_rate=0.0,
        arousal_pool_rate=1.0,
        positive_valence_ceiling=0.1,
        negative_valence_ceiling=0.1,
        arousal_ceiling=0.1,
    )
    assert valence == pytest.approx(1.0)
    assert arousal == pytest.approx(1.0)


def test_pool_rates_to_valence_arousal_returns_zero_for_zero_ceiling():
    valence, arousal = pool_rates_to_valence_arousal(
        approach_pool_rate=0.5,
        avoidance_pool_rate=0.0,
        arousal_pool_rate=0.5,
        positive_valence_ceiling=0.0,
        negative_valence_ceiling=0.0,
        arousal_ceiling=0.0,
    )
    assert valence == pytest.approx(0.0)
    assert arousal == pytest.approx(0.0)


def test_compute_emotions_at_joy_target_gives_joy_intensity_of_exactly_one():
    emotions = compute_emotions(valence=1.0, arousal=0.6)
    assert emotions["joy"] == pytest.approx(1.0)


def test_compute_emotions_at_a_target_suppresses_the_opposite_valence_emotion():
    # Regression guard: a linear (rather than sharply-falling-off) distance falloff left
    # every emotion, including opposite-valence ones, sitting at a similar mid-level
    # intensity regardless of how far its own target actually was.
    emotions = compute_emotions(valence=1.0, arousal=0.6)  # exactly the joy target
    assert emotions["joy"] == pytest.approx(1.0)
    assert emotions["disgust"] < 0.05  # disgust's target (-1.0, 0.5) is the opposite valence


def test_compute_emotions_far_from_all_targets_gives_nonnegative_low_intensities():
    emotions = compute_emotions(valence=10.0, arousal=10.0)
    for emotion_intensity in emotions.values():
        assert 0.0 <= emotion_intensity < 1e-10


def test_compute_emotions_returns_dict_with_exactly_the_eight_plutchik_keys():
    emotions = compute_emotions(valence=0.0, arousal=0.0)
    assert set(emotions.keys()) == {
        "joy",
        "trust",
        "fear",
        "surprise",
        "sadness",
        "disgust",
        "anger",
        "anticipation",
    }


def test_compute_rating_at_zero_valence_returns_midpoint_five():
    assert compute_rating(valence=0.0) == pytest.approx(5.0)


def test_compute_rating_at_valence_one_returns_ten():
    assert compute_rating(valence=1.0) == pytest.approx(10.0)


def test_compute_rating_at_valence_negative_one_returns_zero():
    assert compute_rating(valence=-1.0) == pytest.approx(0.0)


def test_compute_rating_clamps_out_of_range_valence_to_upper_bound():
    assert compute_rating(valence=2.0) == pytest.approx(10.0)


def test_compute_rating_clamps_out_of_range_valence_to_lower_bound():
    assert compute_rating(valence=-2.0) == pytest.approx(0.0)
