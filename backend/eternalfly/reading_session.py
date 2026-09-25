"""Orchestrates the Shiu-validated event-driven LIF simulation, multilingual semantic
sensory encoding (semantic_encoder.py) and behavior/emotion decoding into a single
"the fly reads a book through its own real sensory neurons, and reacts through its own
real descending/motor neurons" session.

Text no longer injects a sentiment score straight into the exact pool it's read back
from (the old design's core bug - see reading_session.py's git history). Real sensory
receptor neurons (cell_groups.SENSORY_CHANNEL_SELECTORS) are Poisson-driven by how
strongly a word matches each real sense; the real connectome (synapses.py) propagates
that; emotions/rating are read from real downstream MBON approach/avoidance activity
(Aso et al. 2014's learned-value readout, distinct from the PAM/PPL1 pools context
valence teaches), not from the same pool being injected into.
"""

import math
import threading
from dataclasses import dataclass, replace

import numpy
import torch

from eternalfly.emotion_decoder import ExponentialMovingAverage, compute_emotions, compute_rating, normalize_rate, pool_rates_to_valence_arousal
from eternalfly.lif import SynapticLIFParameters, SynapticLIFState, create_initial_synaptic_state, poisson_forced_spikes, synaptic_step
from eternalfly.plasticity import compute_updated_edge_weights, decay_eligibility_trace, find_kc_to_mbon_edge_positions, mark_eligible_kcs
from eternalfly.readout import accumulate_spike_counts, build_group_readout_matrix, readout_rates
from eternalfly.semantic_encoder import ChannelCalibration, EmbedFn, WordEmbeddingCache, channel_drives, context_valence, semantic_odor
from eternalfly.synapses import EventDrivenSynapses, advance_delay_buffer, propagate_spikes

# Readout rows this session tracks beyond the 78 real neuropils (see
# connectome.neuropil_readout_matrix) - every real named cell group whose firing rate
# is reported, EXCLUDING the sensory input/odor groups (those are reported as `senses`,
# the CURRENT injected drive level, not a readout of their own resulting spike rate -
# see FrameResult.senses).
BEHAVIOR_NAMES = ["escape", "feeding", "backing", "turn_left", "turn_right"]
REGION_ACTIVITY_GROUP_NAMES = {"reward": "reward_pam", "punishment": "punishment_ppl1", "arousal": "arousal_oa"}
MBON_GROUP_NAMES = {"approach": "mbon_approach", "avoidance": "mbon_avoidance"}


@dataclass(frozen=True)
class SensoryChannelConfig:
    """One sensory channel's (or the shared remaining-olfactory-glomeruli pool's)
    Poisson-injection tuning: how many spikes/second its pool fires at full (1.0)
    drive, and how quickly that drive decays back toward 0 once a word stops
    refreshing it (see ReadingSession._decay_channel_levels)."""

    max_rate_hz: float
    decay_ms: float


@dataclass(frozen=True)
class ReadingSessionConfig:
    """Tunable parameters for a ReadingSession - see scripts/audit_brain.py for how
    these are calibrated against the real connectome."""

    lif_parameters: SynapticLIFParameters
    sim_ms_per_word: float
    channel_configs: dict[str, SensoryChannelConfig]  # keyed like semantic_encoder.SENSORY_CHANNEL_ANCHORS
    odor_config: SensoryChannelConfig
    odor_active_fraction: float
    context_window_word_count: int
    valence_injection_max_rate_hz: float
    display_time_constant_ms: float
    engagement_time_constant_ms: float
    engagement_threshold: float
    min_words_before_boredom_check: int
    positive_valence_ceiling: float
    negative_valence_ceiling: float
    arousal_ceiling: float
    behavior_ceilings: dict[str, float]  # keyed like BEHAVIOR_NAMES - see scripts/audit_brain.py
    plasticity_learning_rate: float
    plasticity_eligibility_time_constant_ms: float
    plasticity_recovery_time_constant_ms: float
    plasticity_update_interval_steps: int
    device: str = "cpu"


@dataclass(frozen=True)
class FrameResult:
    """The observable outcome of advancing a ReadingSession by one or more raw steps."""

    current_word: str | None
    page_progress: float
    words_read: int
    total_words: int
    emotions: dict[str, float]
    rating_0_10: float
    learned_valence: float  # -1..1, from real MBON approach-avoidance activity (see plasticity.py) - the fly's actual learned opinion, distinct from region_activity's raw instinct signal
    region_activity: dict[str, float]  # reward/punishment/arousal - the injected teaching signal's own activity
    behaviors: dict[str, float]  # real descending/motor readouts, normalized 0..1 against behavior_ceilings
    senses: dict[str, float]  # current word's channel drive levels (0..1), not a spike-rate readout
    neuropil_activity: dict[str, float]
    steps_simulated: int
    spikes_per_second: float
    wants_new_book: bool
    book_finished: bool = False


class ReadingSession:
    """Ties the event-driven Shiu LIF network, the multilingual semantic encoder and
    the real cell-type readouts together, one simulated word at a time."""

    def __init__(
        self,
        neuron_count: int,
        synapses: EventDrivenSynapses,
        cell_groups: dict[str, torch.Tensor],
        neuropil_readout_matrix: torch.Tensor,
        neuropil_names: list[str],
        odor_projection: torch.Tensor,
        channel_calibrations: dict[str, ChannelCalibration],
        positive_context_anchor: numpy.ndarray,
        negative_context_anchor: numpy.ndarray,
        embed_fn: EmbedFn,
        tokens: list[str],
        config: ReadingSessionConfig,
    ):
        """cell_groups holds every named group from cell_groups.npz (see
        scripts/build_connectome_cache.py): sensory_<channel>, olfactory_remaining,
        behavior_<name>, reward_pam, punishment_ppl1, arousal_oa, mbon_approach,
        mbon_avoidance, kenyon_cells - as index tensors into the neuron_count-sized
        network. neuropil_readout_matrix is connectome.neuropil_readout_matrix's real
        presynapse-weighted (neuropil_count x neuron_count) matrix, already on device.
        """
        self._neuron_count = neuron_count
        self._synapses = synapses
        self._config = config
        self._device = config.device
        self._tokens = tokens

        self._sensory_group_indices = {
            name.removeprefix("sensory_"): indices.to(config.device)
            for name, indices in cell_groups.items()
            if name.startswith("sensory_")
        }
        self._olfactory_remaining_indices = cell_groups["olfactory_remaining"].to(config.device)

        readout_group_indices = {
            **{f"behavior_{name}": cell_groups[f"behavior_{name}"] for name in BEHAVIOR_NAMES},
            **{key: cell_groups[group_name] for key, group_name in REGION_ACTIVITY_GROUP_NAMES.items()},
            **{key: cell_groups[group_name] for key, group_name in MBON_GROUP_NAMES.items()},
        }
        self._reward_pam_indices = cell_groups["reward_pam"].to(config.device)
        self._punishment_ppl1_indices = cell_groups["punishment_ppl1"].to(config.device)

        # KC->MBON dopamine-gated plasticity (see plasticity.py): avoidance-MBONs are
        # depressed by the real reward(PAM) drive, approach-MBONs by the real
        # punishment(PPL1) drive - the "compartment-proxy" dopamine signal
        # connectome.split_mbons_by_dopamine_input's docstring describes.
        kc_indices = cell_groups["kenyon_cells"].to(config.device)
        mbon_avoidance_indices = cell_groups["mbon_avoidance"].to(config.device)
        mbon_approach_indices = cell_groups["mbon_approach"].to(config.device)
        self._kc_indices = kc_indices
        self._kc_eligibility = torch.zeros(len(kc_indices), device=config.device)
        self._avoidance_edges = find_kc_to_mbon_edge_positions(
            synapses.crow_indices, synapses.col_indices, synapses.values, kc_indices, mbon_avoidance_indices
        )
        self._approach_edges = find_kc_to_mbon_edge_positions(
            synapses.crow_indices, synapses.col_indices, synapses.values, kc_indices, mbon_approach_indices
        )
        self._steps_since_plasticity_update = 0

        group_row_names, group_readout_matrix = build_group_readout_matrix(readout_group_indices, neuron_count, config.device)
        self._readout_row_names = neuropil_names + group_row_names
        self._readout_matrix = torch.cat([neuropil_readout_matrix.to(config.device), group_readout_matrix], dim=0)

        # semantic_odor's API is pure numpy; converting once here avoids a torch<->numpy
        # round trip on every single word.
        self._odor_projection_np = odor_projection.cpu().numpy()
        self._channel_calibrations = channel_calibrations
        self._positive_context_anchor = positive_context_anchor
        self._negative_context_anchor = negative_context_anchor
        self._embed_fn = embed_fn
        self._word_cache = WordEmbeddingCache(embed_fn)

        self._channel_decay_factors = {
            name: math.exp(-config.lif_parameters.dt_ms / channel_config.decay_ms)
            for name, channel_config in config.channel_configs.items()
        }
        self._odor_decay_factor = math.exp(-config.lif_parameters.dt_ms / config.odor_config.decay_ms)
        self._steps_per_word = max(1, round(config.sim_ms_per_word / config.lif_parameters.dt_ms))

        self._state: SynapticLIFState = create_initial_synaptic_state(neuron_count, config.lif_parameters, device=config.device)
        self._last_frame_spike_counts = torch.zeros(neuron_count, device=config.device)
        self._word_index = 0
        self._steps_into_word = 0
        self._words_simulated_total = 0
        self._channel_levels = dict.fromkeys(config.channel_configs, 0.0)
        self._odor_levels = torch.zeros(len(self._olfactory_remaining_indices), device=config.device)
        self._valence_injection_rate_pam = 0.0
        self._valence_injection_rate_ppl1 = 0.0

        self._display_emas: torch.Tensor | None = None
        self._engagement_average = ExponentialMovingAverage(config.engagement_time_constant_ms)
        self._engagement_approach_average = ExponentialMovingAverage(config.engagement_time_constant_ms)
        self._engagement_avoidance_average = ExponentialMovingAverage(config.engagement_time_constant_ms)
        self._engagement_arousal_average = ExponentialMovingAverage(config.engagement_time_constant_ms)

        # Guards every method below that reads or mutates session state: advance() runs
        # on the server's simulation worker thread, while precompute_upcoming_words()
        # (a background lookahead thread) and load_new_text()/restart() (an HTTP
        # request's own worker thread, via /load-book) can all run concurrently with it
        # otherwise - see server.py.
        self._lock = threading.Lock()

    @property
    def sim_ms_per_word(self) -> float:
        """How many simulated milliseconds of brain time each word always gets,
        regardless of playback speed (see server.py's compute_step_count)."""
        return self._config.sim_ms_per_word

    @property
    def dt_ms(self) -> float:
        """The raw simulation timestep size (see server.py's compute_step_count)."""
        return self._config.lif_parameters.dt_ms

    def precompute_upcoming_words(self, word_count: int) -> None:
        """Embed the next word_count not-yet-cached unique words starting at the
        current reading position, in one batch call (see semantic_encoder.WordEmbeddingCache).
        Intended to be called periodically from a background thread (see server.py) so
        _begin_word almost never has to embed synchronously on the simulation's own
        critical path."""
        with self._lock:
            upcoming = self._tokens[self._word_index : self._word_index + word_count]
        if upcoming:
            self._word_cache.precompute(upcoming)  # WordEmbeddingCache has its own internal lock

    def load_new_text(self, tokens: list[str]) -> None:
        """Swap in a new book's tokens and restart word progress from the first word,
        deliberately keeping the simulated brain's ongoing LIF/engagement state and the
        word embedding cache intact across books - only the text feed changes."""
        if not tokens:
            raise ValueError("tokens must not be empty")
        with self._lock:
            self._tokens = tokens
            self._reset_word_progress()

    def restart(self) -> None:
        """Restart the current book from its first word, keeping its tokens and the
        simulated brain's ongoing LIF/engagement state intact."""
        with self._lock:
            self._reset_word_progress()

    def _reset_word_progress(self) -> None:
        self._word_index = 0
        self._steps_into_word = 0
        self._words_simulated_total = 0
        self._channel_levels = dict.fromkeys(self._config.channel_configs, 0.0)
        self._odor_levels = torch.zeros_like(self._odor_levels)
        self._valence_injection_rate_pam = 0.0
        self._valence_injection_rate_ppl1 = 0.0

    def advance(self, step_count: int) -> FrameResult:
        """Advance the simulation by up to step_count raw dt_ms steps (fewer if the
        book finishes first) and return this frame's aggregated result."""
        with self._lock:
            frame_spike_counts = torch.zeros(self._neuron_count, device=self._device)
            steps_run = 0
            book_finished = self._word_index >= len(self._tokens)

            for _ in range(step_count):
                if book_finished:
                    break
                spikes = self._advance_single_step()
                frame_spike_counts = accumulate_spike_counts(frame_spike_counts, spikes)
                steps_run += 1
                book_finished = self._word_index >= len(self._tokens)

            self._last_frame_spike_counts = frame_spike_counts
            return self._build_frame_result(frame_spike_counts, steps_run, book_finished)

    def last_frame_fired_neuron_indices(self, max_count: int) -> torch.Tensor:
        """Return up to max_count neuron indices that fired at least once during the
        most recent advance() call - for the frontend's spike-cloud visualization
        (see server.py), sent as a separate binary WebSocket message rather than
        bloating the JSON frame result. Arbitrarily truncated (not sampled) if more
        neurons fired than max_count - which ones get dropped doesn't matter for a
        purely visual glow effect. Empty before the first advance() call."""
        fired_indices = torch.nonzero(self._last_frame_spike_counts, as_tuple=True)[0]
        return fired_indices[:max_count]

    def _advance_single_step(self) -> torch.Tensor:
        """Advance exactly one dt_ms step and return its spike tensor."""
        if self._steps_into_word == 0:
            self._begin_word(self._word_index)

        forced_spike_mask = self._build_forced_spike_mask()
        self._state, spikes = synaptic_step(self._state, self._config.lif_parameters, forced_spike_mask)
        contribution = propagate_spikes(self._synapses, spikes)
        self._state = replace(
            self._state,
            pending_conductance_increments=advance_delay_buffer(self._state.pending_conductance_increments, contribution),
        )
        self._decay_channel_levels()
        self._update_plasticity(spikes)

        self._steps_into_word += 1
        if self._steps_into_word >= self._steps_per_word:
            self._steps_into_word = 0
            self._word_index += 1
            self._words_simulated_total += 1

        return spikes

    def _begin_word(self, word_index: int) -> None:
        """Refresh every channel's drive level and the reward/punishment injection
        rate from the word about to be read, on top of whatever hasn't yet decayed
        from recent words (see _decay_channel_levels)."""
        word = self._tokens[word_index]
        word_embedding = self._word_cache.get(word)
        if word_embedding is None:
            self._word_cache.precompute([word])  # rare fallback: lookahead hasn't reached this word yet
            word_embedding = self._word_cache.get(word)

        drives = channel_drives(word_embedding, self._channel_calibrations)
        for name, drive in drives.items():
            self._channel_levels[name] = max(self._channel_levels[name], drive)

        odor = semantic_odor(word_embedding, self._odor_projection_np, self._config.odor_active_fraction)
        self._odor_levels = torch.maximum(self._odor_levels, torch.as_tensor(odor, dtype=torch.float32, device=self._device))

        context_words = self._tokens[max(0, word_index - self._config.context_window_word_count + 1) : word_index + 1]
        context_embedding = self._embed_fn([" ".join(context_words)])[0]
        valence = context_valence(context_embedding, self._positive_context_anchor, self._negative_context_anchor)
        self._valence_injection_rate_pam = max(0.0, valence) * self._config.valence_injection_max_rate_hz
        self._valence_injection_rate_ppl1 = max(0.0, -valence) * self._config.valence_injection_max_rate_hz

    def _decay_channel_levels(self) -> None:
        for name in self._channel_levels:
            self._channel_levels[name] *= self._channel_decay_factors[name]
        self._odor_levels = self._odor_levels * self._odor_decay_factor

    def _update_plasticity(self, spikes: torch.Tensor) -> None:
        """Decay every Kenyon cell's eligibility trace, mark any that just fired as
        freshly eligible, and every plasticity_update_interval_steps apply one
        dopamine-gated weight update to the real KC->MBON synapses (see plasticity.py).
        A no-op if this connectome cache has no Kenyon cells (kc_indices empty)."""
        if len(self._kc_indices) == 0:
            return
        dt_ms = self._config.lif_parameters.dt_ms
        self._kc_eligibility = decay_eligibility_trace(
            self._kc_eligibility, dt_ms, self._config.plasticity_eligibility_time_constant_ms
        )
        kc_spike_mask = spikes[self._kc_indices] > 0
        self._kc_eligibility = mark_eligible_kcs(self._kc_eligibility, kc_spike_mask)

        self._steps_since_plasticity_update += 1
        if self._steps_since_plasticity_update >= self._config.plasticity_update_interval_steps:
            self._apply_plasticity_update(self._steps_since_plasticity_update * dt_ms)
            self._steps_since_plasticity_update = 0

    def _apply_plasticity_update(self, elapsed_ms: float) -> None:
        max_rate = self._config.valence_injection_max_rate_hz
        reward_drive = self._valence_injection_rate_pam / max_rate if max_rate > 0 else 0.0
        punishment_drive = self._valence_injection_rate_ppl1 / max_rate if max_rate > 0 else 0.0
        for edges, dopamine_drive in ((self._avoidance_edges, reward_drive), (self._approach_edges, punishment_drive)):
            if edges.edge_positions.numel() == 0:
                continue
            current_weights = self._synapses.values[edges.edge_positions]
            edge_eligibility = self._kc_eligibility[edges.edge_kc_local_index]
            updated_weights = compute_updated_edge_weights(
                current_weights,
                edges.initial_weights,
                edge_eligibility,
                dopamine_drive,
                elapsed_ms,
                self._config.plasticity_learning_rate,
                self._config.plasticity_recovery_time_constant_ms,
            )
            self._synapses.values[edges.edge_positions] = updated_weights

    def memory_snapshot(self) -> dict[str, numpy.ndarray]:
        """The current real KC->MBON synapse weights, keyed for npz persistence (see
        brain_loader.save_memory/load_memory_into_session)."""
        with self._lock:
            return {
                "avoidance_edge_weights": self._synapses.values[self._avoidance_edges.edge_positions].cpu().numpy(),
                "approach_edge_weights": self._synapses.values[self._approach_edges.edge_positions].cpu().numpy(),
            }

    def load_memory(self, data: dict[str, numpy.ndarray]) -> bool:
        """Apply a previously saved memory_snapshot() if its edge counts match this
        session's real KC->MBON edges exactly (a different connectome cache would have
        different counts) - returns whether it was actually applied."""
        avoidance_weights = data.get("avoidance_edge_weights")
        approach_weights = data.get("approach_edge_weights")
        if avoidance_weights is None or approach_weights is None:
            return False
        if len(avoidance_weights) != len(self._avoidance_edges.edge_positions):
            return False
        if len(approach_weights) != len(self._approach_edges.edge_positions):
            return False
        with self._lock:
            self._synapses.values[self._avoidance_edges.edge_positions] = torch.as_tensor(
                avoidance_weights, dtype=self._synapses.values.dtype, device=self._device
            )
            self._synapses.values[self._approach_edges.edge_positions] = torch.as_tensor(
                approach_weights, dtype=self._synapses.values.dtype, device=self._device
            )
            return True

    def reset_memory(self) -> None:
        """Reset every real KC->MBON synapse back to its original, un-learned weight."""
        with self._lock:
            self._synapses.values[self._avoidance_edges.edge_positions] = self._avoidance_edges.initial_weights
            self._synapses.values[self._approach_edges.edge_positions] = self._approach_edges.initial_weights

    def _build_forced_spike_mask(self) -> torch.Tensor:
        dt_ms = self._config.lif_parameters.dt_ms
        full_mask = torch.zeros(self._neuron_count, dtype=torch.bool, device=self._device)

        for name, level in self._channel_levels.items():
            pool_indices = self._sensory_group_indices.get(name)
            if pool_indices is None or len(pool_indices) == 0 or level <= 0.0:
                continue
            drives = torch.full((len(pool_indices),), level, device=self._device)
            pool_spikes = poisson_forced_spikes(drives, self._config.channel_configs[name].max_rate_hz, dt_ms)
            full_mask[pool_indices] |= pool_spikes

        if len(self._olfactory_remaining_indices) > 0:
            odor_spikes = poisson_forced_spikes(self._odor_levels, self._config.odor_config.max_rate_hz, dt_ms)
            full_mask[self._olfactory_remaining_indices] |= odor_spikes

        for rate, pool_indices in (
            (self._valence_injection_rate_pam, self._reward_pam_indices),
            (self._valence_injection_rate_ppl1, self._punishment_ppl1_indices),
        ):
            if rate > 0.0 and len(pool_indices) > 0:
                drives = torch.ones(len(pool_indices), device=self._device)
                full_mask[pool_indices] |= poisson_forced_spikes(drives, rate, dt_ms)

        return full_mask

    def _build_frame_result(self, frame_spike_counts: torch.Tensor, steps_run: int, book_finished: bool) -> FrameResult:
        elapsed_ms = max(steps_run, 1) * self._config.lif_parameters.dt_ms
        raw_rates = readout_rates(self._readout_matrix, frame_spike_counts, max(steps_run, 1))
        smoothed_rates = self._update_display_emas(raw_rates, elapsed_ms)
        rates_by_name = dict(zip(self._readout_row_names, smoothed_rates.tolist()))

        neuropil_row_count = len(self._readout_row_names) - len(BEHAVIOR_NAMES) - len(REGION_ACTIVITY_GROUP_NAMES) - len(MBON_GROUP_NAMES)
        neuropil_activity = {name: rates_by_name[name] for name in self._readout_row_names[:neuropil_row_count]}
        behaviors = {
            name: normalize_rate(rates_by_name[f"behavior_{name}"], self._config.behavior_ceilings[name])
            for name in BEHAVIOR_NAMES
        }
        region_activity = {key: rates_by_name[key] for key in REGION_ACTIVITY_GROUP_NAMES}

        mbon_approach_raw = raw_rates[self._readout_row_names.index("approach")].item()
        mbon_avoidance_raw = raw_rates[self._readout_row_names.index("avoidance")].item()
        arousal_raw = raw_rates[self._readout_row_names.index("arousal")].item()
        emotions, rating, learned_valence = self._compute_emotions_and_rating(mbon_approach_raw, mbon_avoidance_raw, arousal_raw)
        wants_new_book = self._update_engagement(mbon_approach_raw, mbon_avoidance_raw, arousal_raw, elapsed_ms)

        current_word = None if book_finished else self._tokens[self._word_index]
        page_progress = 1.0 if book_finished else (self._word_index + 1) / len(self._tokens)
        words_read = len(self._tokens) if book_finished else self._word_index + 1
        spikes_per_second = (frame_spike_counts.sum().item() / max(steps_run, 1)) * (1000.0 / self._config.lif_parameters.dt_ms)

        return FrameResult(
            current_word=current_word,
            page_progress=page_progress,
            words_read=words_read,
            total_words=len(self._tokens),
            emotions=emotions,
            rating_0_10=rating,
            learned_valence=learned_valence,
            region_activity=region_activity,
            behaviors=behaviors,
            senses=dict(self._channel_levels),
            neuropil_activity=neuropil_activity,
            steps_simulated=steps_run,
            spikes_per_second=spikes_per_second,
            wants_new_book=wants_new_book,
            book_finished=book_finished,
        )

    def _update_display_emas(self, raw_rates: torch.Tensor, elapsed_ms: float) -> torch.Tensor:
        decay = math.exp(-elapsed_ms / self._config.display_time_constant_ms)
        if self._display_emas is None:
            self._display_emas = raw_rates.clone()
        else:
            self._display_emas = decay * self._display_emas + (1.0 - decay) * raw_rates
        return self._display_emas

    def _compute_emotions_and_rating(
        self, approach_rate: float, avoidance_rate: float, arousal_rate: float
    ) -> tuple[dict[str, float], float, float]:
        valence, arousal = pool_rates_to_valence_arousal(
            approach_rate, avoidance_rate, arousal_rate,
            self._config.positive_valence_ceiling, self._config.negative_valence_ceiling, self._config.arousal_ceiling,
        )
        return compute_emotions(valence, arousal), compute_rating(valence), valence

    def _update_engagement(self, approach_rate: float, avoidance_rate: float, arousal_rate: float, elapsed_ms: float) -> bool:
        """Track a long-time-constant engagement score (from the same real MBON
        approach/avoidance and OA arousal readouts emotions/rating use) and report
        whether the fly wants a new book."""
        smoothed_approach = self._engagement_approach_average.update(approach_rate, elapsed_ms)
        smoothed_avoidance = self._engagement_avoidance_average.update(avoidance_rate, elapsed_ms)
        smoothed_arousal = self._engagement_arousal_average.update(arousal_rate, elapsed_ms)
        valence, arousal = pool_rates_to_valence_arousal(
            smoothed_approach, smoothed_avoidance, smoothed_arousal,
            self._config.positive_valence_ceiling, self._config.negative_valence_ceiling, self._config.arousal_ceiling,
        )
        engagement_score = (compute_rating(valence) / 10.0 + arousal) / 2.0
        smoothed_engagement = self._engagement_average.update(engagement_score, elapsed_ms)
        has_enough_history = self._words_simulated_total >= self._config.min_words_before_boredom_check
        return has_enough_history and smoothed_engagement < self._config.engagement_threshold
