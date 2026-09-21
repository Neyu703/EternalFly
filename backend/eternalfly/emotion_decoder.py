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
# Controls how sharply intensity drops off with distance from an emotion's target in the
# Gaussian falloff below. Calibrated so a point clearly closest to one or two targets
# lights mostly those up, instead of all eight targets (which cluster within about a
# unit of each other) sitting at a similar mid-level intensity simultaneously.
EMOTION_FALLOFF_SIGMA = 0.45


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
    approach_pool_rate: float,
    avoidance_pool_rate: float,
    arousal_pool_rate: float,
    positive_valence_ceiling: float,
    negative_valence_ceiling: float,
    arousal_ceiling: float,
) -> tuple[float, float]:
    """Convert approach/avoidance/arousal pool firing rates into a (valence, arousal)
    coordinate pair, rescaled to [-1.0, 1.0] / [0.0, 1.0] against the network's own
    calibrated ceilings (see scripts/calibrate_sentiment.py) rather than the raw pool
    rates directly.

    The raw approach-minus-avoidance difference and raw arousal-pool rate only ever
    span a tiny sliver of their nominal range even under maximally extreme input (the
    dopaminergic/octopaminergic pools are a few hundred neurons out of ~139k, with
    spike rates bounded well under 1.0), so feeding them unscaled into compute_rating/
    compute_emotions makes the display barely move. positive_valence_ceiling and
    negative_valence_ceiling (both non-negative magnitudes) are the raw
    approach-minus-avoidance value that should map to +1.0 / -1.0 respectively (the
    positive and negative dopaminergic pools differ in size, so their ceilings differ
    too); arousal_ceiling is the raw arousal_pool_rate that should map to 1.0. Values
    beyond a ceiling clamp to +-1.0 / 1.0 rather than exceeding it."""
    raw_valence = approach_pool_rate - avoidance_pool_rate
    if raw_valence >= 0:
        valence = min(1.0, raw_valence / positive_valence_ceiling) if positive_valence_ceiling > 0 else 0.0
    else:
        valence = max(-1.0, raw_valence / negative_valence_ceiling) if negative_valence_ceiling > 0 else 0.0
    arousal = max(0.0, min(1.0, arousal_pool_rate / arousal_ceiling)) if arousal_ceiling > 0 else 0.0
    return (valence, arousal)


def compute_emotions(valence: float, arousal: float) -> dict[str, float]:
    """Map a (valence, arousal) coordinate to intensities for all 8 Plutchik primary
    emotions, using a Gaussian falloff from each emotion's fixed target coordinate (see
    EMOTION_FALLOFF_SIGMA) instead of a linear one, so only the target(s) actually close
    to the current coordinate light up rather than every emotion reading a similar
    mid-level intensity regardless of distance."""
    emotion_intensities = {}
    for emotion_name, (target_valence, target_arousal) in EMOTION_TARGETS.items():
        distance_to_target = math.hypot(valence - target_valence, arousal - target_arousal)
        emotion_intensities[emotion_name] = math.exp(-(distance_to_target**2) / (2 * EMOTION_FALLOFF_SIGMA**2))
    return emotion_intensities


def compute_rating(valence: float) -> float:
    """Map valence to a 0-10 rating (5.0 + 5.0 * valence), clamped to [0.0, 10.0]."""
    raw_rating = 5.0 + 5.0 * valence
    return max(0.0, min(10.0, raw_rating))
