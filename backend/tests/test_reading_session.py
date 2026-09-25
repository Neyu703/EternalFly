import numpy
import pytest
import scipy.sparse
import torch

from eternalfly.lif import SHIU_2024_PARAMETERS
from eternalfly.reading_session import ReadingSession, ReadingSessionConfig, SensoryChannelConfig
from eternalfly.semantic_encoder import ChannelCalibration
from eternalfly.synapses import build_event_driven_synapses

NEURON_COUNT = 14
INDEX = {
    "sweet": 0, "bitter": 1, "odor_a": 2, "odor_b": 3,
    "escape": 4, "feeding": 5, "backing": 6, "turn_left": 7, "turn_right": 8,
    "reward_pam": 9, "punishment_ppl1": 10, "arousal_oa": 11, "mbon_approach": 12, "mbon_avoidance": 13,
}


def _cell_groups() -> dict[str, torch.Tensor]:
    return {
        "sensory_sweet": torch.tensor([INDEX["sweet"]]),
        "sensory_bitter": torch.tensor([INDEX["bitter"]]),
        "olfactory_remaining": torch.tensor([INDEX["odor_a"], INDEX["odor_b"]]),
        "behavior_escape": torch.tensor([INDEX["escape"]]),
        "behavior_feeding": torch.tensor([INDEX["feeding"]]),
        "behavior_backing": torch.tensor([INDEX["backing"]]),
        "behavior_turn_left": torch.tensor([INDEX["turn_left"]]),
        "behavior_turn_right": torch.tensor([INDEX["turn_right"]]),
        "reward_pam": torch.tensor([INDEX["reward_pam"]]),
        "punishment_ppl1": torch.tensor([INDEX["punishment_ppl1"]]),
        "arousal_oa": torch.tensor([INDEX["arousal_oa"]]),
        "mbon_approach": torch.tensor([INDEX["mbon_approach"]]),
        "mbon_avoidance": torch.tensor([INDEX["mbon_avoidance"]]),
    }


def _zero_synapses():
    return build_event_driven_synapses(scipy.sparse.csr_matrix((NEURON_COUNT, NEURON_COUNT)))


def _synapses_with_edge(pre: int, post: int, weight: float):
    matrix = scipy.sparse.csr_matrix((NEURON_COUNT, NEURON_COUNT))
    matrix = matrix.tolil()
    matrix[pre, post] = weight
    return build_event_driven_synapses(matrix.tocsr())


class FakeEmbedder:
    """Fake EmbedFn: fixed lookup table, records every batch it was called with."""

    def __init__(self, vectors: dict[str, tuple[float, float]], default: tuple[float, float] = (0.0, 0.0)):
        self.vectors = vectors
        self.default = default
        self.calls: list[list[str]] = []

    def __call__(self, texts: list[str]) -> numpy.ndarray:
        self.calls.append(list(texts))
        return numpy.array([self.vectors.get(text, self.default) for text in texts])


# 3 dimensions so "neutral" can sit exactly orthogonal to BOTH the sweet and bitter
# anchor directions (truly zero similarity), rather than leaking a nonzero drive onto
# either channel the way a 2D "midpoint" embedding would.
WORD_EMBEDDINGS = {"honig": (1.0, 0.0, 0.0), "sauer": (0.0, 1.0, 0.0), "neutral": (0.0, 0.0, 1.0)}


def _make_config(**overrides) -> ReadingSessionConfig:
    defaults = dict(
        lif_parameters=SHIU_2024_PARAMETERS,
        sim_ms_per_word=8.0,
        channel_configs={
            "sweet": SensoryChannelConfig(max_rate_hz=5000.0, decay_ms=1000.0),
            "bitter": SensoryChannelConfig(max_rate_hz=5000.0, decay_ms=1000.0),
        },
        odor_config=SensoryChannelConfig(max_rate_hz=5000.0, decay_ms=1000.0),
        odor_active_fraction=0.5,
        context_window_word_count=1,
        valence_injection_max_rate_hz=5000.0,
        display_time_constant_ms=1000.0,
        engagement_time_constant_ms=1000.0,
        engagement_threshold=0.3,
        min_words_before_boredom_check=3,
        positive_valence_ceiling=1.0,
        negative_valence_ceiling=1.0,
        arousal_ceiling=1.0,
        device="cpu",
    )
    defaults.update(overrides)
    return ReadingSessionConfig(**defaults)


def _make_session(tokens, synapses=None, embed_vectors=None, **config_overrides) -> tuple[ReadingSession, FakeEmbedder]:
    embedder = FakeEmbedder(embed_vectors if embed_vectors is not None else WORD_EMBEDDINGS)
    session = ReadingSession(
        neuron_count=NEURON_COUNT,
        synapses=synapses if synapses is not None else _zero_synapses(),
        cell_groups=_cell_groups(),
        neuropil_readout_matrix=torch.zeros(1, NEURON_COUNT),
        neuropil_names=["FAKE_REGION"],
        odor_projection=torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        # background_mean/std chosen so an exact anchor match (cosine 1.0) lands
        # exactly at Z_RAMP_END (drive 1.0) and an orthogonal embedding (cosine 0.0)
        # lands exactly at Z_RAMP_START (drive 0.0) - see semantic_encoder.py.
        channel_calibrations={
            "sweet": ChannelCalibration(numpy.array([1.0, 0.0, 0.0]), background_mean=-0.75, background_std=0.5),
            "bitter": ChannelCalibration(numpy.array([0.0, 1.0, 0.0]), background_mean=-0.75, background_std=0.5),
        },
        positive_context_anchor=numpy.array([1.0, 0.0, 0.0]),
        negative_context_anchor=numpy.array([0.0, 1.0, 0.0]),
        embed_fn=embedder,
        tokens=tokens,
        config=_make_config(**config_overrides),
    )
    return session, embedder


def test_advance_reports_first_word_current_word_and_progress():
    session, _ = _make_session(["neutral", "neutral"])

    result = session.advance(4)

    assert result.current_word == "neutral"
    assert result.words_read == 1
    assert result.total_words == 2
    assert result.page_progress == 0.5
    assert result.book_finished is False


def test_advance_moves_to_second_word_after_sim_ms_per_word_steps():
    session, _ = _make_session(["neutral", "neutral"], sim_ms_per_word=4.0)

    session.advance(4)
    result = session.advance(1)

    assert result.words_read == 2


def test_advance_marks_book_finished_and_stops_early():
    session, _ = _make_session(["neutral"], sim_ms_per_word=3.0)

    result = session.advance(100)

    assert result.book_finished is True
    assert result.current_word is None
    assert result.steps_simulated == 3  # stopped as soon as the single word finished
    assert result.page_progress == 1.0
    assert result.words_read == 1


def test_advance_with_zero_step_count_does_not_crash_and_advances_nothing():
    session, _ = _make_session(["neutral", "neutral"])

    result = session.advance(0)

    assert result.steps_simulated == 0
    assert result.words_read == 1


def test_senses_reflects_the_current_words_channel_drive():

    session, _ = _make_session(["honig", "sauer"], sim_ms_per_word=2.0)

    result = session.advance(1)

    # "honig" clears the sweet channel's z-ramp exactly at the top; slightly below 1.0
    # here since one decay step (decay_ms=1000) has already applied by the time this
    # frame's result is read.
    assert result.senses["sweet"] == pytest.approx(1.0, abs=0.01)
    assert result.senses["bitter"] == 0.0


def test_senses_decays_after_the_triggering_word_has_passed():
    session, _ = _make_session(
        ["honig", "neutral", "neutral", "neutral"],
        sim_ms_per_word=2.0,
        channel_configs={
            "sweet": SensoryChannelConfig(max_rate_hz=5000.0, decay_ms=2.0),
            "bitter": SensoryChannelConfig(max_rate_hz=5000.0, decay_ms=1000.0),
        },
    )

    first = session.advance(2)
    later = session.advance(6)

    assert first.senses["sweet"] > 0.0
    assert 0.0 < later.senses["sweet"] < first.senses["sweet"]


def test_forced_sensory_spikes_propagate_through_a_real_synapse_to_behavior_readout():
    # neuron 0 (sweet) -> neuron 4 (escape) with a huge weight: every step "honig" is
    # read, the sweet pool is guaranteed to fire (max_rate_hz*dt/1000 clamped to
    # probability 1.0), and that spike alone should push the escape neuron over threshold.
    synapses = _synapses_with_edge(INDEX["sweet"], INDEX["escape"], weight=50.0)
    session, _ = _make_session(["honig", "honig", "honig"], synapses=synapses, sim_ms_per_word=6.0)

    result = session.advance(6)

    assert result.behaviors["escape"] > 0.0


def test_neutral_word_never_drives_the_escape_behavior_through_the_same_synapse():
    synapses = _synapses_with_edge(INDEX["sweet"], INDEX["escape"], weight=50.0)
    session, _ = _make_session(["neutral", "neutral"], synapses=synapses, sim_ms_per_word=6.0)

    result = session.advance(6)

    assert result.behaviors["escape"] == 0.0


def test_positive_context_valence_injects_into_the_reward_pam_readout():
    session, _ = _make_session(["honig", "honig"], sim_ms_per_word=6.0)

    result = session.advance(6)

    assert result.region_activity["reward"] > 0.0
    assert result.region_activity["punishment"] == 0.0


def test_negative_context_valence_injects_into_the_punishment_ppl1_readout():
    session, _ = _make_session(["sauer", "sauer"], sim_ms_per_word=6.0)

    result = session.advance(6)

    assert result.region_activity["punishment"] > 0.0
    assert result.region_activity["reward"] == 0.0


def test_neutral_context_injects_into_neither_pam_nor_ppl1():
    session, _ = _make_session(["neutral", "neutral"], sim_ms_per_word=6.0)

    result = session.advance(6)

    assert result.region_activity["reward"] == 0.0
    assert result.region_activity["punishment"] == 0.0


def test_wants_new_book_becomes_true_after_enough_disengaged_words():
    tokens = ["neutral"] * 10
    session, _ = _make_session(tokens, sim_ms_per_word=2.0, min_words_before_boredom_check=3, engagement_threshold=0.3)

    results = []
    for _ in range(10):
        results.append(session.advance(2))

    assert results[0].wants_new_book is False
    assert any(result.wants_new_book for result in results)


def test_wants_new_book_stays_false_before_min_words_reached():
    tokens = ["neutral"] * 10
    session, _ = _make_session(tokens, sim_ms_per_word=2.0, min_words_before_boredom_check=1000, engagement_threshold=0.9)

    for _ in range(5):
        result = session.advance(2)

    assert result.wants_new_book is False


def test_restart_resets_word_progress_and_channel_levels_but_leaves_lif_state_untouched():
    session, _ = _make_session(["honig", "neutral", "neutral"], sim_ms_per_word=2.0)
    session.advance(6)
    state_before_restart = session._state

    session.restart()

    assert session._state is state_before_restart  # restart() itself never touches LIF state

    result = session.advance(1)

    assert result.current_word == "honig"
    assert result.words_read == 1
    assert result.senses["sweet"] > 0.9  # re-reads "honig" fresh from word 0, not mid-decay from before


def test_load_new_text_replaces_tokens_and_resets_progress():
    session, _ = _make_session(["honig", "neutral"], sim_ms_per_word=2.0)
    session.advance(2)

    session.load_new_text(["sauer", "neutral"])
    result = session.advance(1)

    assert result.current_word == "sauer"
    assert result.total_words == 2


def test_load_new_text_raises_value_error_on_empty_token_list():
    session, _ = _make_session(["honig"])

    with pytest.raises(ValueError, match="tokens must not be empty"):
        session.load_new_text([])


def test_precompute_upcoming_words_embeds_only_uncached_words_ahead_of_current_position():
    session, embedder = _make_session(["honig", "sauer", "neutral", "neutral"])

    session.precompute_upcoming_words(3)

    assert embedder.calls == [["honig", "sauer", "neutral"]]

    session.precompute_upcoming_words(3)  # all three already cached, no new call
    assert embedder.calls == [["honig", "sauer", "neutral"]]


def test_begin_word_falls_back_to_synchronous_embedding_when_not_precomputed():
    # No precompute_upcoming_words() call first - _begin_word must still work.
    session, embedder = _make_session(["honig", "neutral"], sim_ms_per_word=2.0)

    result = session.advance(1)

    assert result.senses["sweet"] > 0.9
    assert ["honig"] in embedder.calls


def test_frame_result_reports_neuropil_activity_and_spikes_per_second_fields():
    session, _ = _make_session(["neutral", "neutral"])

    result = session.advance(4)

    assert result.neuropil_activity == {"FAKE_REGION": 0.0}
    assert result.spikes_per_second >= 0.0
    assert set(result.behaviors) == {"escape", "feeding", "backing", "turn_left", "turn_right"}


def test_sim_ms_per_word_and_dt_ms_properties_expose_the_configured_values():
    session, _ = _make_session(["neutral"], sim_ms_per_word=42.0)

    assert session.sim_ms_per_word == 42.0
    assert session.dt_ms == SHIU_2024_PARAMETERS.dt_ms
