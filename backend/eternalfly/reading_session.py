"""Orchestrates the LIF simulation, text encoding and emotion decoding into a single
tick-by-tick 'the fly reads a book' session."""

from dataclasses import dataclass

import torch

from eternalfly.emotion_decoder import RollingAverage, compute_emotions, compute_rating, pool_rates_to_valence_arousal
from eternalfly.lif import LIFParameters, LIFState, create_initial_state, step
from eternalfly.session_helpers import (
    compute_pool_spike_rate,
    inject_currents_at_indices,
    word_index_for_tick,
)
from eternalfly.text_encoder import project_token_to_currents


@dataclass(frozen=True)
class ReadingSessionConfig:
    """Tunable parameters for a ReadingSession."""

    lif_parameters: LIFParameters
    ticks_per_word: int
    input_current_scale: float
    valence_weight: float
    token_seed: int
    engagement_window_size: int
    engagement_threshold: float
    min_ticks_before_boredom_check: int
    device: str = "cpu"


@dataclass(frozen=True)
class TickResult:
    """The observable outcome of advancing a ReadingSession by one tick."""

    current_word: str | None
    page_progress: float
    words_read: int
    total_words: int
    emotions: dict[str, float]
    rating_0_10: float
    region_activity: dict[str, float]
    neuropil_activity: dict[str, float]
    wants_new_book: bool
    book_finished: bool = False


class ReadingSession:
    """Ties the LIF network, text encoder and emotion decoder together, one tick at a time."""

    def __init__(
        self,
        neuron_count: int,
        adjacency_matrix: torch.Tensor,
        pool_indices: dict[str, torch.Tensor],
        tokens: list[str],
        config: ReadingSessionConfig,
        neuropil_pool_indices: dict[str, torch.Tensor] | None = None,
    ):
        """Create a session over tokens, using pool_indices["sensory_input"/"approach"/
        "avoidance"/"arousal"] (index tensors into the neuron_count-sized network).

        neuropil_pool_indices optionally maps arbitrary region names (e.g. real FlyWire
        neuropil codes) to index tensors, tracked purely for reporting live per-region
        activity via TickResult.neuropil_activity; they play no part in the emotion or
        engagement calculations."""
        self._neuron_count = neuron_count
        self._adjacency_matrix = adjacency_matrix
        self._sensory_pool_indices = pool_indices["sensory_input"].to(config.device)
        self._approach_pool_indices = pool_indices["approach"].to(config.device)
        self._avoidance_pool_indices = pool_indices["avoidance"].to(config.device)
        self._arousal_pool_indices = pool_indices["arousal"].to(config.device)
        self._tokens = tokens
        self._config = config

        self._state: LIFState = create_initial_state(neuron_count, device=config.device)
        self._previous_spikes = torch.zeros(neuron_count, device=config.device)
        self._tick_number = 0
        self._is_paused = False
        self._speed_multiplier = 1.0
        self._last_tick_result: TickResult | None = None

        self._approach_rate_average = RollingAverage(config.engagement_window_size)
        self._avoidance_rate_average = RollingAverage(config.engagement_window_size)
        self._arousal_rate_average = RollingAverage(config.engagement_window_size)
        self._engagement_average = RollingAverage(config.engagement_window_size)

        self._neuropil_pool_indices = {
            region_name: indices.to(config.device) for region_name, indices in (neuropil_pool_indices or {}).items()
        }
        self._neuropil_rate_averages = {
            region_name: RollingAverage(config.engagement_window_size) for region_name in self._neuropil_pool_indices
        }

    def _current_external_input(self, word_index: int, book_finished: bool) -> torch.Tensor:
        """Return this tick's injected current: the active word's projection, held for its
        entire ticks_per_word window (not just its first tick) so the signal has enough
        sustained drive to propagate through several synaptic hops before decaying away."""
        if book_finished:
            return torch.zeros(self._neuron_count, device=self._config.device)

        token_currents = project_token_to_currents(
            self._tokens[word_index],
            len(self._sensory_pool_indices),
            self._config.input_current_scale,
            self._config.token_seed,
            self._config.valence_weight,
        )
        return inject_currents_at_indices(
            self._neuron_count,
            self._sensory_pool_indices,
            torch.as_tensor(token_currents, dtype=torch.float32),
            self._config.device,
        )

    def _update_emotions(self, spikes: torch.Tensor) -> tuple[dict[str, float], float, dict[str, float]]:
        """Roll pool spike rates forward and derive this tick's emotions, rating and activity."""
        approach_rate = self._approach_rate_average.update(compute_pool_spike_rate(spikes, self._approach_pool_indices))
        avoidance_rate = self._avoidance_rate_average.update(compute_pool_spike_rate(spikes, self._avoidance_pool_indices))
        arousal_rate = self._arousal_rate_average.update(compute_pool_spike_rate(spikes, self._arousal_pool_indices))

        valence, arousal = pool_rates_to_valence_arousal(approach_rate, avoidance_rate, arousal_rate)
        emotions = compute_emotions(valence, arousal)
        rating = compute_rating(valence)
        region_activity = {"approach": approach_rate, "avoidance": avoidance_rate, "arousal": arousal_rate}
        return emotions, rating, region_activity

    def _update_neuropil_activity(self, spikes: torch.Tensor) -> dict[str, float]:
        """Roll each configured neuropil region's spike rate forward and return the
        smoothed activity per region name. Empty when no neuropil_pool_indices were given."""
        return {
            region_name: self._neuropil_rate_averages[region_name].update(compute_pool_spike_rate(spikes, indices))
            for region_name, indices in self._neuropil_pool_indices.items()
        }

    def _update_engagement(self, rating: float, arousal_rate: float) -> bool:
        """Track a smoothed engagement score and report whether the fly wants a new book."""
        engagement_score = (rating / 10.0 + arousal_rate) / 2.0
        smoothed_engagement = self._engagement_average.update(engagement_score)
        has_enough_history = self._tick_number >= self._config.min_ticks_before_boredom_check
        return has_enough_history and smoothed_engagement < self._config.engagement_threshold

    def load_new_text(self, tokens: list[str]) -> None:
        """Swap in a new book's tokens and restart word progress from tick 0, while
        deliberately keeping the simulated brain's ongoing LIF/engagement state intact
        across books — only the text feed changes."""
        if not tokens:
            raise ValueError("tokens must not be empty")
        self._tokens = tokens
        self._tick_number = 0

    def restart(self) -> None:
        """Restart the current book from its first word, keeping its tokens and the
        simulated brain's ongoing LIF/engagement state intact."""
        self._tick_number = 0

    def set_paused(self, paused: bool) -> None:
        """Pause or resume tick() advancing the simulation. While paused, tick() keeps
        returning the last computed TickResult instead of stepping the network."""
        self._is_paused = paused

    def set_speed_multiplier(self, multiplier: float) -> None:
        """Set how many effective ticks make up one word: higher values read faster.
        Raises ValueError if multiplier is not positive."""
        if multiplier <= 0:
            raise ValueError(f"speed multiplier must be positive, got {multiplier}")
        self._speed_multiplier = multiplier

    def _effective_ticks_per_word(self) -> int:
        """Return ticks_per_word scaled down by the current speed multiplier, never below 1."""
        return max(1, round(self._config.ticks_per_word / self._speed_multiplier))

    def tick(self) -> TickResult:
        """Advance the session by one simulation tick and return its observable result.
        While paused, returns the last result unchanged instead of advancing."""
        if self._is_paused and self._last_tick_result is not None:
            return self._last_tick_result

        word_index = word_index_for_tick(self._tick_number, self._effective_ticks_per_word())
        book_finished = word_index >= len(self._tokens)
        current_word = None if book_finished else self._tokens[word_index]

        external_input = self._current_external_input(word_index, book_finished)
        self._state, spikes = step(
            self._state, self._adjacency_matrix, external_input, self._previous_spikes, self._config.lif_parameters
        )
        self._previous_spikes = spikes

        emotions, rating, region_activity = self._update_emotions(spikes)
        neuropil_activity = self._update_neuropil_activity(spikes)
        wants_new_book = self._update_engagement(rating, region_activity["arousal"])

        page_progress = 1.0 if book_finished else (word_index + 1) / len(self._tokens)
        words_read = len(self._tokens) if book_finished else word_index + 1
        self._tick_number += 1

        tick_result = TickResult(
            current_word=current_word,
            page_progress=page_progress,
            words_read=words_read,
            total_words=len(self._tokens),
            emotions=emotions,
            rating_0_10=rating,
            region_activity=region_activity,
            neuropil_activity=neuropil_activity,
            wants_new_book=wants_new_book,
            book_finished=book_finished,
        )
        self._last_tick_result = tick_result
        return tick_result
