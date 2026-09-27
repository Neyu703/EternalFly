"""Orchestrates the LIF simulation, text encoding and emotion decoding into a single
tick-by-tick 'the fly reads a book' session."""

import logging
from dataclasses import dataclass

import numpy
import torch

from eternalfly.activity_readout import ActivityReadout, RegionSynapseWeights
from eternalfly.emotion_decoder import PoolCalibration
from eternalfly.lif import LIFParameters, LIFState, create_initial_state, step
from eternalfly.session_helpers import inject_currents_at_indices
from eternalfly.text_encoder import (
    VALENCE_CHANNELS,
    project_arousal_to_currents,
    project_token_to_currents,
    project_valence_to_currents,
)
from eternalfly.word_activity_log import WordActivityLog

logger = logging.getLogger(__name__)

# Which pool_indices population each emotion is read from (see ReadingSession.__init__).
EMOTION_POOL_NAMES = {"reward": "valence_positive", "aversion": "valence_negative", "arousal": "arousal_input"}

# The text_encoder valence channel that excites each valence emotion's population.
VALENCE_CHANNEL_BY_EMOTION = dict(zip(("reward", "aversion"), VALENCE_CHANNELS))


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
        self._emotion_pool_indices = {
            emotion_name: pool_indices[pool_name].to(config.device) for emotion_name, pool_name in EMOTION_POOL_NAMES.items()
        }
        self._tokens = tokens
        self._config = config

        self._state: LIFState = create_initial_state(neuron_count, device=config.device)
        self._previous_spikes = torch.zeros(neuron_count, device=config.device)
        self._tick_number = 0
        self._word_index = 0
        self._ticks_into_word = 0
        self._is_paused = False
        self._speed_multiplier = 1.0
        self._last_tick_result: TickResult | None = None

        self._readout = ActivityReadout(
            self._emotion_pool_indices,
            config.emotion_calibrations,
            config.display_window_size,
            config.engagement_window_size,
            config.lif_parameters.dt_ms,
            region_synapse_weights,
            config.device,
        )
        self._word_activity_log = WordActivityLog()

    def _pool_input(self, pool_indices: torch.Tensor, currents: numpy.ndarray) -> torch.Tensor:
        """A full-network current tensor carrying currents at pool_indices, zero elsewhere."""
        return inject_currents_at_indices(
            self._neuron_count, pool_indices, torch.as_tensor(currents, dtype=torch.float32), self._config.device
        )

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
        config = self._config
        external_input = self._pool_input(
            self._sensory_pool_indices,
            project_token_to_currents(token, len(self._sensory_pool_indices), config.input_current_scale, config.token_seed),
        )
        for emotion_name, channel in VALENCE_CHANNEL_BY_EMOTION.items():
            valence_pool_indices = self._emotion_pool_indices[emotion_name]
            external_input += self._pool_input(
                valence_pool_indices,
                project_valence_to_currents(
                    token, len(valence_pool_indices), config.input_current_scale, config.valence_weight, channel
                ),
            )
        arousal_pool_indices = self._emotion_pool_indices["arousal"]
        external_input += self._pool_input(
            arousal_pool_indices,
            project_arousal_to_currents(token, len(arousal_pool_indices), config.input_current_scale, config.arousal_weight),
        )
        return external_input

    def _wants_new_book(self, smoothed_engagement: float) -> bool:
        """Whether the fly has read long enough to judge and stayed below the engagement
        threshold over its recent reading."""
        has_enough_history = self._tick_number >= self._config.min_ticks_before_boredom_check
        return has_enough_history and smoothed_engagement < self._config.engagement_threshold

    def _restart_word_progress(self) -> None:
        """Put the reading position back on the book's first word. The last result is
        dropped too, so a paused session shows the (new or restarted) book's first word
        instead of the result from before."""
        self._tick_number = 0
        self._word_index = 0
        self._ticks_into_word = 0
        self._last_tick_result = None

    def load_new_text(self, tokens: list[str]) -> None:
        """Swap in a new book's tokens and restart word progress from its first word, while
        deliberately keeping the simulated brain's ongoing LIF/engagement state intact
        across books — only the text feed changes."""
        if not tokens:
            raise ValueError("tokens must not be empty")
        self._word_activity_log.close_book()
        self._tokens = tokens
        self._restart_word_progress()
        logger.info("new book loaded: %d words", len(tokens))

    def restart(self) -> None:
        """Restart the current book from its first word, keeping its tokens and the
        simulated brain's ongoing LIF/engagement state intact."""
        self._word_activity_log.close_book()
        self._restart_word_progress()
        logger.info("book restarted from its first word")

    def set_paused(self, paused: bool) -> None:
        """Pause or resume tick() advancing the simulation. While paused, tick() keeps
        returning the last computed TickResult instead of stepping the network."""
        self._is_paused = paused
        logger.info("reading %s", "paused" if paused else "resumed")

    def set_speed_multiplier(self, multiplier: float) -> None:
        """Set how many effective ticks make up one word: higher values read faster. The
        word being read keeps its place; only how long each word lasts changes. Raises
        ValueError if multiplier is not positive."""
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

    def _advance_word_position(self) -> None:
        """Count one more tick on the current word, moving on to the next word once it has
        lasted the current effective ticks per word."""
        self._ticks_into_word += 1
        if self._ticks_into_word >= self._effective_ticks_per_word():
            self._word_index += 1
            self._ticks_into_word = 0

    def _advance_single_tick(self) -> TickResult:
        """Advance the session by exactly one raw simulation tick and return its result."""
        word_index = self._word_index
        book_finished = word_index >= len(self._tokens)
        current_word = None if book_finished else self._tokens[word_index]

        external_input = self._current_external_input(word_index, book_finished)
        self._state, spikes = step(
            self._state, self._adjacency_matrix, external_input, self._previous_spikes, self._config.lif_parameters
        )
        self._previous_spikes = spikes

        readout = self._readout.read(spikes)
        self._word_activity_log.record(word_index, current_word, readout.raw_rates, readout.emotions, readout.rating_0_10)
        tick_result = TickResult(
            current_word=current_word,
            page_progress=1.0 if book_finished else (word_index + 1) / len(self._tokens),
            words_read=len(self._tokens) if book_finished else word_index + 1,
            total_words=len(self._tokens),
            emotions=readout.emotions,
            rating_0_10=readout.rating_0_10,
            region_activity=readout.region_activity,
            neuropil_activity=readout.neuropil_activity,
            firing_rate_hz=readout.firing_rate_hz,
            wants_new_book=self._wants_new_book(readout.smoothed_engagement),
            book_finished=book_finished,
        )

        self._tick_number += 1
        if not book_finished:
            self._advance_word_position()
        self._last_tick_result = tick_result
        return tick_result
