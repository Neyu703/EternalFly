"""Reads the fly's observable state out of each simulation tick's spikes: its emotions and
dopamine rating, its brain regions' and whole brain's firing rates, and how engaged it has
been over a long stretch of reading."""

from dataclasses import dataclass

import torch

from eternalfly.emotion_decoder import (
    EMOTION_NAMES,
    MAX_RATING,
    PoolCalibration,
    RollingAverage,
    compute_emotions,
    compute_rating,
    emotions_to_valence,
)
from eternalfly.session_helpers import compute_pool_spike_rate, spike_fraction_to_hz


@dataclass(frozen=True)
class RegionSynapseWeights:
    """Brain regions to report firing rates for: their names and a (region count, neuron
    count) matrix whose row per region holds each neuron's share of that region's synapses
    (see connectome.build_region_synapse_weights), in the same order."""

    names: list[str]
    weights: torch.Tensor


@dataclass(frozen=True)
class Readout:
    """Everything read out of one tick: the emotion populations' raw rates, the displayed
    (short-window) emotions, rating and activity, and the long-window engagement score."""

    raw_rates: dict[str, float]
    emotions: dict[str, float]
    rating_0_10: float
    region_activity: dict[str, float]
    neuropil_activity: dict[str, float]
    firing_rate_hz: float
    smoothed_engagement: float


def _roll_rates(averages: dict[str, RollingAverage], rates: dict[str, float]) -> dict[str, float]:
    """Push each rate into its rolling average (matched by name) and return the new means."""
    return {name: average.update(rates[name]) for name, average in averages.items()}


class ActivityReadout:
    """Turns each tick's spikes into a Readout, smoothing over two independent timescales:
    a short display window, so the displayed rating/emotions/region activity visibly react
    to the sentence currently being read, and a long engagement window purely for judging
    whether the fly has been engaged over its recent reading as a whole."""

    def __init__(
        self,
        emotion_pool_indices: dict[str, torch.Tensor],
        emotion_calibrations: dict[str, PoolCalibration],
        display_window_size: int,
        engagement_window_size: int,
        dt_ms: float,
        region_synapse_weights: RegionSynapseWeights | None = None,
        device: str = "cpu",
    ):
        """Read the emotions from emotion_pool_indices (keyed by emotion name, index
        tensors into the network) against emotion_calibrations, report rates in Hz for a
        simulation step of dt_ms, and report region_synapse_weights' regions, if given."""
        self._emotion_pool_indices = emotion_pool_indices
        self._emotion_calibrations = emotion_calibrations
        self._dt_ms = dt_ms
        self._display_rate_averages = {name: RollingAverage(display_window_size) for name in EMOTION_NAMES}
        self._engagement_rate_averages = {name: RollingAverage(engagement_window_size) for name in EMOTION_NAMES}
        self._engagement_average = RollingAverage(engagement_window_size)
        self._region_names = region_synapse_weights.names if region_synapse_weights else []
        self._region_weights = region_synapse_weights.weights.to(device) if region_synapse_weights else None
        self._region_rate_averages = [RollingAverage(display_window_size) for _ in self._region_names]
        self._firing_rate_average = RollingAverage(display_window_size)

    def read(self, spikes: torch.Tensor) -> Readout:
        """Read one tick's spikes, advancing every rolling average by that tick."""
        raw_rates = {name: compute_pool_spike_rate(spikes, indices) for name, indices in self._emotion_pool_indices.items()}
        display_rates = _roll_rates(self._display_rate_averages, raw_rates)
        emotions, rating = self._emotions_and_rating(display_rates)
        return Readout(
            raw_rates=raw_rates,
            emotions=emotions,
            rating_0_10=rating,
            region_activity={
                "approach": self._to_hz(display_rates["reward"]),
                "avoidance": self._to_hz(display_rates["aversion"]),
                "arousal": emotions["arousal"],
            },
            neuropil_activity=self._update_region_rates(spikes),
            firing_rate_hz=self._to_hz(self._firing_rate_average.update(spikes.mean().item())),
            smoothed_engagement=self._update_engagement(raw_rates),
        )

    def _emotions_and_rating(self, pool_rates: dict[str, float]) -> tuple[dict[str, float], float]:
        """The emotions read from pool_rates (keyed by emotion name) against the
        calibrations, and the dopamine rating of their net valence."""
        emotions = compute_emotions(pool_rates, self._emotion_calibrations)
        return emotions, compute_rating(emotions_to_valence(emotions))

    def _to_hz(self, spike_fraction: float) -> float:
        """Convert a per-tick spike fraction into spikes per second for this time step."""
        return spike_fraction_to_hz(spike_fraction, self._dt_ms)

    def _update_region_rates(self, spikes: torch.Tensor) -> dict[str, float]:
        """Roll each region's synapse-weighted firing rate forward and return the smoothed
        rate in Hz per region name. Empty when no region_synapse_weights were given."""
        if self._region_weights is None:
            return {}
        region_spike_fractions = (self._region_weights @ spikes).tolist()
        return {
            region_name: self._to_hz(average.update(spike_fraction))
            for region_name, average, spike_fraction in zip(
                self._region_names, self._region_rate_averages, region_spike_fractions
            )
        }

    def _update_engagement(self, raw_rates: dict[str, float]) -> float:
        """Roll raw_rates through the long engagement window and return the smoothed
        engagement score: the mean of the long-window rating (as a fraction of MAX_RATING)
        and arousal."""
        emotions, rating = self._emotions_and_rating(_roll_rates(self._engagement_rate_averages, raw_rates))
        engagement_score = (rating / MAX_RATING + emotions["arousal"]) / 2.0
        return self._engagement_average.update(engagement_score)
