import pytest

from eternalfly.emotion_decoder import (
    PoolCalibration,
    RollingAverage,
    compute_emotions,
    compute_rating,
    emotions_to_valence,
    rate_above_rest,
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


CALIBRATION = PoolCalibration(resting_rate=0.05, peak_rate=0.25)
CALIBRATIONS = {"reward": CALIBRATION, "aversion": CALIBRATION, "arousal": CALIBRATION}


def test_rate_above_rest_maps_resting_rate_to_zero():
    assert rate_above_rest(0.05, CALIBRATION) == pytest.approx(0.0)


def test_rate_above_rest_maps_peak_rate_to_one():
    assert rate_above_rest(0.25, CALIBRATION) == pytest.approx(1.0)


def test_rate_above_rest_scales_linearly_between_rest_and_peak():
    assert rate_above_rest(0.10, CALIBRATION) == pytest.approx(0.25)


def test_rate_above_rest_clamps_below_rest_to_zero():
    assert rate_above_rest(0.01, CALIBRATION) == pytest.approx(0.0)


def test_rate_above_rest_clamps_above_peak_to_one():
    assert rate_above_rest(0.9, CALIBRATION) == pytest.approx(1.0)


def test_rate_above_rest_returns_zero_when_peak_is_not_above_rest():
    assert rate_above_rest(0.5, PoolCalibration(resting_rate=0.2, peak_rate=0.2)) == 0.0


def test_compute_emotions_returns_exactly_the_three_states_the_fly_brain_has():
    emotions = compute_emotions({"reward": 0.05, "aversion": 0.05, "arousal": 0.05}, CALIBRATIONS)
    assert set(emotions.keys()) == {"reward", "aversion", "arousal"}


def test_compute_emotions_reads_each_state_from_its_own_population():
    emotions = compute_emotions({"reward": 0.15, "aversion": 0.05, "arousal": 0.10}, CALIBRATIONS)
    assert emotions == pytest.approx({"reward": 0.5, "aversion": 0.0, "arousal": 0.25})


def test_compute_emotions_shows_reward_and_aversion_together_when_both_populations_fire():
    emotions = compute_emotions({"reward": 0.15, "aversion": 0.25, "arousal": 0.05}, CALIBRATIONS)
    assert emotions["reward"] == pytest.approx(0.5)
    assert emotions["aversion"] == pytest.approx(1.0)


def test_compute_emotions_uses_each_populations_own_calibration():
    calibrations = {**CALIBRATIONS, "aversion": PoolCalibration(resting_rate=0.0, peak_rate=0.1)}
    emotions = compute_emotions({"reward": 0.05, "aversion": 0.05, "arousal": 0.05}, calibrations)
    assert emotions["aversion"] == pytest.approx(0.5)
    assert emotions["reward"] == pytest.approx(0.0)


def test_emotions_to_valence_is_reward_minus_aversion():
    assert emotions_to_valence({"reward": 0.6, "aversion": 0.2, "arousal": 0.9}) == pytest.approx(0.4)


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
