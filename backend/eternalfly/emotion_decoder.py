"""Turns smoothed brain-region firing rates into Plutchik emotions and a 0-10 rating."""

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


class ExponentialMovingAverage:
    """A moving average over a stream of values, smoothed with a real time constant
    rather than a fixed value COUNT. Unlike a fixed dt, elapsed_ms is passed to each
    update() call rather than baked in at construction - reading_session.py's new
    event-driven engine advances in variable-sized frames (a server-controlled step
    count, not a fixed number of ticks per word), so a single call's real elapsed
    simulated time can vary from one update to the next."""

    def __init__(self, time_constant_ms: float):
        """Create an EMA with the given (positive) time constant."""
        if time_constant_ms <= 0:
            raise ValueError("time_constant_ms must be positive")
        self._time_constant_ms = time_constant_ms
        self._value = 0.0
        self._has_seen_a_value = False

    def update(self, value: float, elapsed_ms: float) -> float:
        """Fold value into the average (as if it held for the last elapsed_ms of
        simulated time) and return the new smoothed value. The very first update seeds
        the average at value itself, rather than decaying up from 0 (which would
        otherwise take several time constants to reach a steady input). Raises
        ValueError if elapsed_ms is not positive."""
        if elapsed_ms <= 0:
            raise ValueError("elapsed_ms must be positive")
        if not self._has_seen_a_value:
            self._value = value
            self._has_seen_a_value = True
        else:
            decay = math.exp(-elapsed_ms / self._time_constant_ms)
            self._value = decay * self._value + (1.0 - decay) * value
        return self._value


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
    calibrated ceilings (see scripts/audit_brain.py) rather than the raw pool
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
