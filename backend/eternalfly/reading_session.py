"""Orchestrates the LIF simulation, text encoding and emotion decoding into a single
tick-by-tick 'the fly reads a book' session."""

import logging
from dataclasses import dataclass

import torch

from eternalfly.emotion_decoder import (
    EMOTION_NAMES,
    PoolCalibration,
    RollingAverage,
    compute_emotions,
    compute_rating,
    emotions_to_valence,
)
from eternalfly.lif import LIFParameters, LIFState, create_initial_state, step
from eternalfly.session_helpers import (
    compute_pool_spike_rate,
    inject_currents_at_indices,
    spike_fraction_to_hz,
    word_index_for_tick,
)
from eternalfly.text_encoder import project_arousal_to_currents, project_token_to_currents, project_valence_to_currents
from eternalfly.word_activity_log import WordActivityLog

logger = logging.getLogger(__name__)


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
    emotion_calibrations: dict[str, PoolCalibration]  # resting/peak rate per emotion's population, see emotion_decoder.compute_emotions
    device: str = "cpu"


@dataclass(frozen=True)
class RegionSynapseWeights:
    """Brain regions to report firing rates for: their names and a (region count, neuron
    count) matrix whose row per region holds each neuron's share of that region's synapses
    (see connectome.build_region_synapse_weights), in the same order."""

    names: list[str]
    weights: torch.Tensor


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
    firing_rate_hz: float
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
        region_synapse_weights: RegionSynapseWeights | None = None,
    ):
        """Create a session over tokens, using pool_indices["sensory_input"/
        "valence_positive"/"valence_negative"/"arousal_input"] (index tensors into the
        neuron_count-sized network). valence_positive/valence_negative are the real
        dopaminergic neurons of the mushroom body's medial (reward) and vertical
        (punishment) lobes, and arousal_input the real octopaminergic neurons of the
        central complex (see scripts/build_connectome_cache.py) — each word's sentiment
        actively excites the matching valence population, and its sentiment *magnitude*
        (regardless of sign) actively excites arousal_input (see
        _current_external_input). The same three populations are read back out as the
        fly's reward, aversion and arousal, so what is shown is their actual firing,
        including everything the rest of the network feeds into them.

        region_synapse_weights optionally names brain regions (e.g. real FlyWire neuropil
        codes) whose synapse-weighted firing rate is reported live, in Hz, via
        TickResult.neuropil_activity; they play no part in the emotion or
        engagement calculations."""
        self._neuron_count = neuron_count
        self._adjacency_matrix = adjacency_matrix
        self._sensory_pool_indices = pool_indices["sensory_input"].to(config.device)
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
        self._display_rate_averages = {name: RollingAverage(config.display_window_size) for name in EMOTION_NAMES}
        self._engagement_rate_averages = {name: RollingAverage(config.engagement_window_size) for name in EMOTION_NAMES}
        self._engagement_average = RollingAverage(config.engagement_window_size)
        self._word_activity_log = WordActivityLog()

        self._region_names = region_synapse_weights.names if region_synapse_weights else []
        self._region_weights = region_synapse_weights.weights.to(config.device) if region_synapse_weights else None
        self._region_rate_averages = [RollingAverage(config.display_window_size) for _ in self._region_names]
        self._firing_rate_average = RollingAverage(config.display_window_size)

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
        """Return this tick's raw (unsmoothed) spike rate of each emotion's population,
        keyed by emotion name, fed into both the short display window and the long
        engagement window below."""
        return {
            "reward": compute_pool_spike_rate(spikes, self._valence_positive_pool_indices),
            "aversion": compute_pool_spike_rate(spikes, self._valence_negative_pool_indices),
            "arousal": compute_pool_spike_rate(spikes, self._arousal_input_pool_indices),
        }

    def _emotions_and_rating(self, pool_rates: dict[str, float]) -> tuple[dict[str, float], float]:
        """The emotions read from pool_rates (keyed by emotion name) against this session's
        calibrations, and the dopamine rating of their net valence."""
        emotions = compute_emotions(pool_rates, self._config.emotion_calibrations)
        return emotions, compute_rating(emotions_to_valence(emotions))

    def _update_display_activity(self, raw_rates: dict[str, float]) -> tuple[dict[str, float], float, dict[str, float]]:
        """Roll raw_rates through the short display window and derive this tick's
        visibly-reactive emotions, rating and region_activity.
        region_activity reports the reward and punishment dopamine neurons' smoothed firing
        rates in Hz as "approach"/"avoidance" and the arousal emotion as "arousal", so the
        displayed Arousal tile matches the Emotions card."""
        display_rates = {name: average.update(raw_rates[name]) for name, average in self._display_rate_averages.items()}
        emotions, rating = self._emotions_and_rating(display_rates)
        region_activity = {
            "approach": self._to_hz(display_rates["reward"]),
            "avoidance": self._to_hz(display_rates["aversion"]),
            "arousal": emotions["arousal"],
        }
        return emotions, rating, region_activity

    def _update_engagement_rating_and_arousal(self, raw_rates: dict[str, float]) -> tuple[float, float]:
        """Roll raw_rates through the long engagement window and derive the smoothed
        rating/arousal used to judge whether the fly is bored (see _update_engagement)."""
        engagement_rates = {
            name: average.update(raw_rates[name]) for name, average in self._engagement_rate_averages.items()
        }
        emotions, rating = self._emotions_and_rating(engagement_rates)
        return rating, emotions["arousal"]

    def _to_hz(self, spike_fraction: float) -> float:
        """Convert a per-tick spike fraction into spikes per second (see
        session_helpers.spike_fraction_to_hz) for this session's simulation time step."""
        return spike_fraction_to_hz(spike_fraction, self._config.lif_parameters.dt_ms)

    def _update_neuropil_activity(self, spikes: torch.Tensor) -> dict[str, float]:
        """Roll each configured region's synapse-weighted firing rate forward and return
        the smoothed rate in Hz per region name. Empty when no region_synapse_weights were
        given."""
        if self._region_weights is None:
            return {}
        region_spike_fractions = (self._region_weights @ spikes).tolist()
        return {
            region_name: self._to_hz(average.update(spike_fraction))
            for region_name, average, spike_fraction in zip(
                self._region_names, self._region_rate_averages, region_spike_fractions
            )
        }

    def _update_firing_rate(self, spikes: torch.Tensor) -> float:
        """Roll the whole brain's firing rate (the mean over every simulated neuron) forward
        through the display window and return it in Hz."""
        return self._to_hz(self._firing_rate_average.update(spikes.mean().item()))

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
        self._word_activity_log.close_book()
        self._tokens = tokens
        self._tick_number = 0
        logger.info("new book loaded: %d words", len(tokens))

    def restart(self) -> None:
        """Restart the current book from its first word, keeping its tokens and the
        simulated brain's ongoing LIF/engagement state intact."""
        self._word_activity_log.close_book()
        self._tick_number = 0
        logger.info("book restarted from its first word")

    def set_paused(self, paused: bool) -> None:
        """Pause or resume tick() advancing the simulation. While paused, tick() keeps
        returning the last computed TickResult instead of stepping the network."""
        self._is_paused = paused
        logger.info("reading %s", "paused" if paused else "resumed")

    def set_speed_multiplier(self, multiplier: float) -> None:
        """Set how many effective ticks make up one word: higher values read faster.
        Raises ValueError if multiplier is not positive."""
        if multiplier <= 0:
            raise ValueError(f"speed multiplier must be positive, got {multiplier}")
        self._speed_multiplier = multiplier
        logger.info("speed multiplier set to %g", multiplier)

    def _effective_ticks_per_word(self) -> int:
        """Return ticks_per_word scaled down by the current speed multiplier, never below 1."""
        return max(1, round(self._config.ticks_per_word / self._speed_multiplier))

    def _ticks_per_call(self) -> int:
        """How many raw simulation ticks a single public tick() call advances.

        Speeding up first shrinks ticks-per-word (see _effective_ticks_per_word) down to
        its floor of 1 raw tick per word - the fastest a word can be represented at all.
        Beyond that floor (speed_multiplier > ticks_per_word), further speed can only
        come from advancing multiple whole words' worth of ticks within one tick() call,
        so reading speed isn't permanently capped at "1 tick per word" once that floor
        is reached."""
        floor_speed = self._config.ticks_per_word
        if self._speed_multiplier <= floor_speed:
            return 1
        return round(self._speed_multiplier / floor_speed)

    def tick(self) -> TickResult:
        """Advance the session by one or more raw simulation ticks (see _ticks_per_call)
        and return the resulting observable state. While paused, returns the last result
        unchanged instead of advancing."""
        if self._is_paused and self._last_tick_result is not None:
            return self._last_tick_result

        result = self._advance_single_tick()
        for _ in range(self._ticks_per_call() - 1):
            if result.book_finished:
                break
            result = self._advance_single_tick()
        return result

    def _advance_single_tick(self) -> TickResult:
        """Advance the session by exactly one raw simulation tick and return its result."""
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
        self._word_activity_log.record(word_index, current_word, raw_rates, emotions, rating)
        neuropil_activity = self._update_neuropil_activity(spikes)
        firing_rate_hz = self._update_firing_rate(spikes)
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
            firing_rate_hz=firing_rate_hz,
            wants_new_book=wants_new_book,
            book_finished=book_finished,
        )
        self._last_tick_result = tick_result
        return tick_result
