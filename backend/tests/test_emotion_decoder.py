import pytest

from eternalfly.emotion_decoder import (
    RollingAverage,
    compute_emotions,
    compute_rating,
    pool_rates_to_valence_arousal,
)


def test_rolling_average_raises_on_non_positive_window_size():
    with pytest.raises(ValueError):
        RollingAverage(window_size=0)


def test_rolling_average_raises_on_non_integer_window_size():
    with pytest.raises(ValueError):
        RollingAverage(window_size=2.5)


def test_rolling_average_first_update_returns_that_single_value():
    rolling_average = RollingAverage(window_size=3)
    assert rolling_average.update(4.0) == 4.0


def test_rolling_average_returns_mean_of_seen_values_before_window_fills():
    rolling_average = RollingAverage(window_size=3)
    rolling_average.update(2.0)
    assert rolling_average.update(4.0) == 3.0


def test_rolling_average_drops_oldest_values_once_window_is_full():
    rolling_average = RollingAverage(window_size=2)
    rolling_average.update(10.0)
    rolling_average.update(20.0)
    assert rolling_average.update(30.0) == 25.0


def test_rolling_average_running_sum_stays_correct_across_many_evictions():
    # Guards the O(1) running-sum implementation: repeatedly evicting values must not
    # let the running sum drift away from the true mean of the current window.
    window_size = 5
    rolling_average = RollingAverage(window_size=window_size)
    values = [float(value) for value in range(1, 51)]
    for value in values:
        result = rolling_average.update(value)
    expected_mean = sum(values[-window_size:]) / window_size
    assert result == pytest.approx(expected_mean)


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


def test_compute_emotions_far_from_all_targets_gives_nonnegative_low_intensities():
    emotions = compute_emotions(valence=10.0, arousal=10.0)
    for emotion_intensity in emotions.values():
        assert emotion_intensity == 0.0


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
