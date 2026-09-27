"""Turns the smoothed firing rates of the fly brain's neuromodulatory populations into its
emotional states and a 0-10 rating."""

import collections
from dataclasses import dataclass

from eternalfly.validation import require_positive_int

# The emotional states the simulated fly brain has circuits for (see compute_emotions).
EMOTION_NAMES = ("reward", "aversion", "arousal")

# The dopamine rating's scale: 0 (all aversion) to MAX_RATING (all reward), with
# NEUTRAL_RATING in the middle.
MAX_RATING = 10.0
NEUTRAL_RATING = MAX_RATING / 2


def _clamp(value: float, low: float, high: float) -> float:
    """value limited to the range [low, high]."""
    return max(low, min(high, value))


class RollingAverage:
    """A fixed-window moving average over a stream of values.

    Maintains a running sum instead of resumming the whole window on every update, so
    update() stays O(1) even for very large window sizes (needed for a reading session
    to track sentiment over many words — one instance of this runs per tick for every
    tracked brain region, so an O(window_size) update would scale badly)."""

    def __init__(self, window_size: int):
        """Create a rolling average over the most recent window_size values.
        window_size must be a positive integer."""
        require_positive_int("window_size", window_size)
        self._window = collections.deque(maxlen=window_size)
        self._running_sum = 0.0

    def update(self, value: float) -> float:
        """Append value to the window and return the mean of the values currently
        held in the window (fewer than window_size before the window fills up)."""
        if len(self._window) == self._window.maxlen:
            self._running_sum -= self._window[0]
        self._window.append(value)
        self._running_sum += value
        return self._running_sum / len(self._window)


@dataclass(frozen=True)
class PoolCalibration:
    """Spike rates of one neuromodulatory population measured on the real connectome (see
    scripts/calibrate_sentiment.py): resting_rate while the fly reads emotionally neutral
    text, peak_rate while it reads the most strongly charged text."""

    resting_rate: float
    peak_rate: float


def rate_above_rest(rate: float, calibration: PoolCalibration) -> float:
    """How far rate has risen from the population's resting rate toward its peak rate, as
    0..1 (clamped). The population's spontaneous, network-driven firing is its resting
    state rather than an emotion, so it maps to 0. Returns 0.0 for a calibration whose
    peak is not above its resting rate."""
    span = calibration.peak_rate - calibration.resting_rate
    if span <= 0:
        return 0.0
    return _clamp((rate - calibration.resting_rate) / span, 0.0, 1.0)


def compute_emotions(pool_rates: dict[str, float], calibrations: dict[str, PoolCalibration]) -> dict[str, float]:
    """Return the intensities (0..1) of the only emotional states the simulated fly brain
    actually has circuits for, instead of human emotion categories projected onto it, each
    read straight from its own population's firing above rest: reward (the reward-coding
    PAM dopamine neurons of the mushroom body's medial lobe), aversion (the punishment-
    coding PPL1 dopamine neurons of its vertical lobe) and arousal (the octopamine neurons
    driving the central complex). pool_rates and calibrations are keyed by those three
    names; reward and aversion are independent, so mixed passages can show both."""
    return {name: rate_above_rest(pool_rates[name], calibrations[name]) for name in EMOTION_NAMES}


def emotions_to_valence(emotions: dict[str, float]) -> float:
    """Net valence in [-1, 1]: reward minus aversion (feeds compute_rating)."""
    return emotions["reward"] - emotions["aversion"]


def compute_rating(valence: float) -> float:
    """Map valence to a 0-MAX_RATING rating (NEUTRAL_RATING at valence 0), clamped to
    [0.0, MAX_RATING]."""
    return _clamp(NEUTRAL_RATING + NEUTRAL_RATING * valence, 0.0, MAX_RATING)
