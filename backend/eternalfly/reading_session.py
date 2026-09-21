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
from eternalfly.text_encoder import project_arousal_to_currents, project_token_to_currents, project_valence_to_currents


@dataclass(frozen=True)
class ReadingSessionConfig:
    """Tunable parameters for a ReadingSession."""

    lif_parameters: LIFParameters
    ticks_per_word: int
    input_current_scale: float
    valence_weight: float
    arousal_weight: float
    token_seed: int
    engagement_window_size: int
    engagement_threshold: float
    min_ticks_before_boredom_check: int
    display_window_size: int  # short window for the live-displayed rating/emotions/region_activity, separate from engagement_window_size's long-run boredom judgment
    positive_valence_ceiling: float  # raw approach-minus-avoidance rate that maps to valence +1.0, see emotion_decoder.pool_rates_to_valence_arousal
    negative_valence_ceiling: float  # raw avoidance-minus-approach rate that maps to valence -1.0
    arousal_ceiling: float  # raw arousal-pool rate that maps to arousal 1.0
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
        "avoidance"/"arousal"/"valence_positive"/"valence_negative"/"arousal_input"]
        (index tensors into the neuron_count-sized network). valence_positive/
        valence_negative are real dopaminergic neurons synapsing onto the approach/
        avoidance compartments respectively, and arousal_input is real octopaminergic
        neurons synapsing onto the arousal compartment (see
        scripts/build_connectome_cache.py) — each word's sentiment actively excites
        the matching valence channel, and its sentiment *magnitude* (regardless of
        sign) actively excites arousal_input (see _current_external_input), rather
        than only shifting the sensory pool's own drive up or down.

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
        self._valence_positive_pool_indices = pool_indices["valence_positive"].to(config.device)
        self._valence_negative_pool_indices = pool_indices["valence_negative"].to(config.device)
        self._arousal_input_pool_indices = pool_indices["arousal_input"].to(config.device)
        self._tokens = tokens
        self._config = config

        self._state: LIFState = create_initial_state(neuron_count, device=config.device)
        self._previous_spikes = torch.zeros(neuron_count, device=config.device)
        self._tick_number = 0
        self._is_paused = False
        self._speed_multiplier = 1.0
        self._last_tick_result: TickResult | None = None

        # Two independent timescales over the same raw per-tick pool spike rates: a short
        # window so the displayed rating/emotions/region_activity visibly react to the
        # sentence currently being read, and a long window purely for judging whether the
        # fly has been engaged over its recent reading as a whole (wants_new_book).
        self._display_rate_averages = {
            "approach": RollingAverage(config.display_window_size),
            "avoidance": RollingAverage(config.display_window_size),
            "arousal": RollingAverage(config.display_window_size),
        }
        self._engagement_rate_averages = {
            "approach": RollingAverage(config.engagement_window_size),
            "avoidance": RollingAverage(config.engagement_window_size),
            "arousal": RollingAverage(config.engagement_window_size),
        }
        self._engagement_average = RollingAverage(config.engagement_window_size)

        self._neuropil_pool_indices = {
            region_name: indices.to(config.device) for region_name, indices in (neuropil_pool_indices or {}).items()
        }
        self._neuropil_rate_averages = {
            region_name: RollingAverage(config.display_window_size) for region_name in self._neuropil_pool_indices
        }

    def _current_external_input(self, word_index: int, book_finished: bool) -> torch.Tensor:
        """Return this tick's injected current: the active word's projection, held for its
        entire ticks_per_word window (not just its first tick) so the signal has enough
        sustained drive to propagate through several synaptic hops before decaying away.

        Combines two independent channels: generic per-token noise into the sensory
        pool (texture only, no sentiment), and the word's real sentiment actively
        exciting the matching dopaminergic valence pool (positive words excite
        valence_positive, negative words excite valence_negative, neutral words excite
        neither) — see text_encoder.project_valence_to_currents for why both valence
        directions need to be actively excitatory rather than one side just being
        "less input"."""
        if book_finished:
            return torch.zeros(self._neuron_count, device=self._config.device)

        token = self._tokens[word_index]

        sensory_currents = project_token_to_currents(
            token, len(self._sensory_pool_indices), self._config.input_current_scale, self._config.token_seed
        )
        external_input = inject_currents_at_indices(
            self._neuron_count,
            self._sensory_pool_indices,
            torch.as_tensor(sensory_currents, dtype=torch.float32),
            self._config.device,
        )

        for valence_pool_indices, channel in (
            (self._valence_positive_pool_indices, "positive"),
            (self._valence_negative_pool_indices, "negative"),
        ):
            valence_currents = project_valence_to_currents(
                token, len(valence_pool_indices), self._config.input_current_scale, self._config.valence_weight, channel
            )
            external_input = external_input + inject_currents_at_indices(
                self._neuron_count,
                valence_pool_indices,
                torch.as_tensor(valence_currents, dtype=torch.float32),
                self._config.device,
            )

        arousal_currents = project_arousal_to_currents(
            token, len(self._arousal_input_pool_indices), self._config.input_current_scale, self._config.arousal_weight
        )
        external_input = external_input + inject_currents_at_indices(
            self._neuron_count,
            self._arousal_input_pool_indices,
            torch.as_tensor(arousal_currents, dtype=torch.float32),
            self._config.device,
        )

        return external_input

    def _raw_pool_rates(self, spikes: torch.Tensor) -> dict[str, float]:
        """Return this tick's raw (unsmoothed) spike rate for each emotion-tracking pool,
        fed into both the short display window and the long engagement window below."""
        return {
            "approach": compute_pool_spike_rate(spikes, self._approach_pool_indices),
            "avoidance": compute_pool_spike_rate(spikes, self._avoidance_pool_indices),
            "arousal": compute_pool_spike_rate(spikes, self._arousal_pool_indices),
        }

    def _pool_rates_to_valence_arousal(self, approach_rate: float, avoidance_rate: float, arousal_rate: float) -> tuple[float, float]:
        """Convenience wrapper binding this session's calibrated ceilings (see
        ReadingSessionConfig) to emotion_decoder.pool_rates_to_valence_arousal."""
        return pool_rates_to_valence_arousal(
            approach_rate,
            avoidance_rate,
            arousal_rate,
            self._config.positive_valence_ceiling,
            self._config.negative_valence_ceiling,
            self._config.arousal_ceiling,
        )

    def _update_display_activity(self, raw_rates: dict[str, float]) -> tuple[dict[str, float], float, dict[str, float]]:
        """Roll raw_rates through the short display window and derive this tick's
        visibly-reactive emotions, rating and region_activity. region_activity's arousal
        is the same ceiling-normalized value emotions/rating use, so the displayed
        Erregung tile matches what actually drives the fly's mood; approach/avoidance
        stay as raw (smoothed) pool spike rates, which aren't independently displayed."""
        display_rates = {name: average.update(raw_rates[name]) for name, average in self._display_rate_averages.items()}
        valence, arousal = self._pool_rates_to_valence_arousal(
            display_rates["approach"], display_rates["avoidance"], display_rates["arousal"]
        )
        emotions = compute_emotions(valence, arousal)
        rating = compute_rating(valence)
        region_activity = {**display_rates, "arousal": arousal}
        return emotions, rating, region_activity

    def _update_engagement_rating_and_arousal(self, raw_rates: dict[str, float]) -> tuple[float, float]:
        """Roll raw_rates through the long engagement window and derive the smoothed
        rating/arousal used to judge whether the fly is bored (see _update_engagement)."""
        engagement_rates = {
            name: average.update(raw_rates[name]) for name, average in self._engagement_rate_averages.items()
        }
        valence, arousal = self._pool_rates_to_valence_arousal(
            engagement_rates["approach"], engagement_rates["avoidance"], engagement_rates["arousal"]
        )
        return compute_rating(valence), arousal

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

        raw_rates = self._raw_pool_rates(spikes)
        emotions, rating, region_activity = self._update_display_activity(raw_rates)
        neuropil_activity = self._update_neuropil_activity(spikes)
        engagement_rating, engagement_arousal = self._update_engagement_rating_and_arousal(raw_rates)
        wants_new_book = self._update_engagement(engagement_rating, engagement_arousal)

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
