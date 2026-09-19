"""Turns smoothed brain-region firing rates into Plutchik emotions and a 0-10 rating."""

import collections
import math

EMOTION_TARGETS = {
    "joy": (1.0, 0.6),
    "trust": (0.6, 0.2),
    "fear": (-0.8, 0.9),
    "surprise": (0.0, 1.0),
    "sadness": (-0.6, 0.1),
    "disgust": (-1.0, 0.5),
    "anger": (-0.7, 0.9),
    "anticipation": (0.5, 0.6),
}
DISTANCE_NORMALIZER = 2.5


class RollingAverage:
    """A fixed-window moving average over a stream of values.

    Maintains a running sum instead of resumming the whole window on every update, so
    update() stays O(1) even for very large window sizes (needed for a reading session
    to track sentiment over many words — one instance of this runs per tick for every
    tracked brain region, so an O(window_size) update would scale badly)."""

    def __init__(self, window_size: int):
        """Create a rolling average over the most recent window_size values.
        window_size must be a positive integer."""
        if not isinstance(window_size, int) or window_size <= 0:
            raise ValueError("window_size must be a positive integer")
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


def pool_rates_to_valence_arousal(
    approach_pool_rate: float, avoidance_pool_rate: float, arousal_pool_rate: float
) -> tuple[float, float]:
    """Convert approach/avoidance/arousal pool firing rates into a (valence, arousal)
    coordinate pair. Unclamped: callers are expected to pass already-normalized rates."""
    valence = approach_pool_rate - avoidance_pool_rate
    arousal = arousal_pool_rate
    return (valence, arousal)


def compute_emotions(valence: float, arousal: float) -> dict[str, float]:
    """Map a (valence, arousal) coordinate to intensities for all 8 Plutchik primary
    emotions, based on Euclidean distance to each emotion's fixed target coordinate."""
    emotion_intensities = {}
    for emotion_name, (target_valence, target_arousal) in EMOTION_TARGETS.items():
        distance_to_target = math.hypot(valence - target_valence, arousal - target_arousal)
        emotion_intensities[emotion_name] = max(0.0, 1.0 - distance_to_target / DISTANCE_NORMALIZER)
    return emotion_intensities


def compute_rating(valence: float) -> float:
    """Map valence to a 0-10 rating (5.0 + 5.0 * valence), clamped to [0.0, 10.0]."""
    raw_rating = 5.0 + 5.0 * valence
    return max(0.0, min(10.0, raw_rating))
