import math

import pytest
import torch

from eternalfly.emotion_decoder import EMOTION_NAMES, PoolCalibration, compute_emotions, compute_rating, emotions_to_valence
from eternalfly.lif import LIFParameters
from eternalfly.activity_readout import RegionSynapseWeights
from eternalfly.reading_session import ReadingSession, ReadingSessionConfig, TickResult

NEURON_COUNT = 8
ZERO_ADJACENCY = torch.zeros((NEURON_COUNT, NEURON_COUNT))
POOL_INDICES = {
    "sensory_input": torch.tensor([0, 1], dtype=torch.int64),
    "valence_positive": torch.tensor([5], dtype=torch.int64),
    "valence_negative": torch.tensor([6], dtype=torch.int64),
    "arousal_input": torch.tensor([7], dtype=torch.int64),
}
EMOTION_CALIBRATIONS = {name: PoolCalibration(resting_rate=0.0, peak_rate=1.0) for name in EMOTION_NAMES}
LIF_PARAMETERS = LIFParameters(
    membrane_time_constant_ms=20.0,
    spike_threshold=1.0,
    reset_potential=0.0,
    refractory_period_ms=2.0,
    dt_ms=1.0,
)


def _make_config(**overrides) -> ReadingSessionConfig:
    defaults = dict(
        lif_parameters=LIF_PARAMETERS,
        ticks_per_word=2,
        input_current_scale=0.5,
        valence_weight=1.0,
        arousal_weight=1.0,
        token_seed=0,
        engagement_window_size=3,
        engagement_threshold=0.3,
        min_ticks_before_boredom_check=3,
        display_window_size=3,
        emotion_calibrations=EMOTION_CALIBRATIONS,
        device="cpu",
    )
    defaults.update(overrides)
    return ReadingSessionConfig(**defaults)


def _make_session(
    tokens: list[str],
    pool_indices: dict[str, torch.Tensor] = POOL_INDICES,
    region_synapse_weights: RegionSynapseWeights | None = None,
    **config_overrides,
) -> ReadingSession:
    """A session reading tokens on the tiny unconnected test network, configured by
    _make_config(**config_overrides)."""
    return ReadingSession(
        NEURON_COUNT, ZERO_ADJACENCY, pool_indices, tokens, _make_config(**config_overrides), region_synapse_weights
    )


# Input strong enough to make every excited neuron spike on its first tick, one tick per
# word, and a display window of just that tick.
STRONG_INPUT_CONFIG = dict(ticks_per_word=1, input_current_scale=100.0, display_window_size=1)


def _make_strong_input_session(word: str, region_synapse_weights: RegionSynapseWeights | None = None) -> ReadingSession:
    """A one-word session with STRONG_INPUT_CONFIG."""
    return _make_session([word], region_synapse_weights=region_synapse_weights, **STRONG_INPUT_CONFIG)


def test_tick_reports_current_word_and_page_progress_advancing_with_ticks_per_word():
    session = _make_session(["hello", "world"])

    first_tick = session.tick()
    second_tick = session.tick()
    third_tick = session.tick()

    assert first_tick.current_word == "hello"
    assert first_tick.page_progress == pytest.approx(0.5)
    assert second_tick.current_word == "hello"
    assert second_tick.page_progress == pytest.approx(0.5)
    assert third_tick.current_word == "world"
    assert third_tick.page_progress == pytest.approx(1.0)


def test_tick_returns_none_word_and_full_progress_after_book_finished():
    session = _make_session(["hello"], ticks_per_word=1)

    session.tick()
    after_book_end = session.tick()

    assert after_book_end.current_word is None
    assert after_book_end.page_progress == pytest.approx(1.0)


def test_tick_reports_words_read_and_total_words_on_first_tick():
    session = _make_session(["hello", "world"])

    first_tick = session.tick()

    assert first_tick.total_words == 2
    assert first_tick.words_read == 1


def test_tick_words_read_reaches_and_stays_at_total_words_after_book_finished():
    session = _make_session(["hello"], ticks_per_word=1)

    session.tick()
    after_book_end = session.tick()

    assert after_book_end.words_read == 1
    assert after_book_end.total_words == 1


def test_tick_computes_emotions_and_rating_from_zero_pool_activity():
    session = _make_session(["hello", "world"])

    tick_result = session.tick()

    expected_emotions = compute_emotions(dict.fromkeys(EMOTION_NAMES, 0.0), EMOTION_CALIBRATIONS)
    expected_rating = compute_rating(emotions_to_valence(expected_emotions))

    assert tick_result.emotions == expected_emotions
    assert tick_result.rating_0_10 == pytest.approx(expected_rating)
    assert tick_result.region_activity == {"approach": 0.0, "avoidance": 0.0, "arousal": 0.0}


def test_tick_sets_wants_new_book_once_engagement_stays_low_long_enough():
    session = _make_session(["hello", "world", "again"], ticks_per_word=1, min_ticks_before_boredom_check=3)

    results = [session.tick() for _ in range(4)]

    assert results[0].wants_new_book is False
    assert results[1].wants_new_book is False
    assert results[3].wants_new_book is True


def test_tick_does_not_flag_wants_new_book_before_min_ticks_reached():
    session = _make_session(["hello"], ticks_per_word=1, min_ticks_before_boredom_check=1000)

    tick_result = session.tick()

    assert tick_result.wants_new_book is False


def test_tick_reports_empty_neuropil_activity_when_no_region_synapse_weights_given():
    session = _make_session(["hello", "world"])

    tick_result = session.tick()

    assert tick_result.neuropil_activity == {}


def test_tick_reports_zero_region_firing_rates_while_nothing_fires():
    region_synapse_weights = RegionSynapseWeights(names=["ME_L", "MB_CA_R"], weights=torch.zeros((2, NEURON_COUNT)))
    session = _make_session(["hello", "world"], region_synapse_weights=region_synapse_weights)

    tick_result = session.tick()

    assert tick_result.neuropil_activity == {"ME_L": 0.0, "MB_CA_R": 0.0}


def test_tick_reports_each_regions_synapse_weighted_firing_rate_in_hz():
    # Neuron 5 (a reward dopamine neuron) fires on "good"; region A holds a quarter of its
    # synapses on it, region B none.
    weights = torch.zeros((2, NEURON_COUNT))
    weights[0, 5] = 0.25
    weights[0, 0] = 0.75
    weights[1, 1] = 1.0
    session = _make_strong_input_session(
        "good", region_synapse_weights=RegionSynapseWeights(names=["A", "B"], weights=weights.to_sparse_csr())
    )

    tick_result = session.tick()

    assert tick_result.neuropil_activity == pytest.approx({"A": 250.0, "B": 0.0})


def test_tick_reports_the_whole_brains_mean_firing_rate_in_hz():
    session = _make_strong_input_session("good")

    tick_result = session.tick()

    spiking_neuron_count = int(session._previous_spikes.sum().item())
    assert spiking_neuron_count > 0
    assert tick_result.firing_rate_hz == pytest.approx(spiking_neuron_count / NEURON_COUNT * 1000.0)


def test_tick_reports_the_dopamine_neurons_firing_rates_in_hz():
    tick_result = _make_strong_input_session("good").tick()

    assert tick_result.region_activity["approach"] == pytest.approx(1000.0)
    assert tick_result.region_activity["avoidance"] == pytest.approx(0.0)


def test_load_new_text_replaces_tokens_and_resets_tick_number_but_keeps_current_word_from_new_book():
    session = _make_session(["hello", "world"], ticks_per_word=1)
    session.tick()
    session.tick()

    session.load_new_text(["new", "book"])
    tick_result = session.tick()

    assert tick_result.current_word == "new"
    assert tick_result.total_words == 2
    assert tick_result.words_read == 1


def test_load_new_text_preserves_lif_state_and_engagement_tracking_across_books():
    session = _make_session(["hello", "world", "again"], ticks_per_word=1, min_ticks_before_boredom_check=3)
    for _ in range(4):
        session.tick()
    state_before_reload = session._state
    previous_spikes_before_reload = session._previous_spikes
    readout_before_reload = session._readout

    session.load_new_text(["fresh", "start"])

    assert session._state is state_before_reload
    assert session._previous_spikes is previous_spikes_before_reload
    assert session._readout is readout_before_reload


def test_load_new_text_raises_value_error_on_empty_token_list():
    session = _make_session(["hello", "world"])

    with pytest.raises(ValueError, match="tokens must not be empty"):
        session.load_new_text([])


def test_tick_reports_book_finished_false_while_book_still_has_words():
    session = _make_session(["hello", "world"])

    tick_result = session.tick()

    assert tick_result.book_finished is False


def test_tick_reports_book_finished_true_once_book_is_finished():
    session = _make_session(["hello"], ticks_per_word=1)

    session.tick()
    after_book_end = session.tick()

    assert after_book_end.book_finished is True


def test_restart_resets_word_progress_to_first_word_keeping_same_tokens():
    session = _make_session(["hello", "world"], ticks_per_word=1)
    session.tick()
    session.tick()

    session.restart()
    tick_result = session.tick()

    assert tick_result.current_word == "hello"
    assert tick_result.total_words == 2


def test_restart_preserves_lif_state_and_engagement_tracking():
    session = _make_session(["hello", "world", "again"], ticks_per_word=1, min_ticks_before_boredom_check=3)
    for _ in range(4):
        session.tick()
    state_before_restart = session._state
    readout_before_restart = session._readout

    session.restart()

    assert session._state is state_before_restart
    assert session._readout is readout_before_restart


def test_set_paused_true_freezes_tick_result_instead_of_advancing():
    session = _make_session(["hello", "world"])
    first_tick = session.tick()

    session.set_paused(True)
    second_tick = session.tick()
    third_tick = session.tick()

    assert second_tick == first_tick
    assert third_tick == first_tick


def test_set_paused_false_after_pause_resumes_advancing():
    session = _make_session(["hello", "world"], ticks_per_word=1)
    session.tick()

    session.set_paused(True)
    session.tick()
    session.set_paused(False)
    resumed_tick = session.tick()

    assert resumed_tick.current_word == "world"


def test_set_speed_multiplier_above_one_advances_words_faster():
    session = _make_session(["hello", "world"], ticks_per_word=4)
    session.set_speed_multiplier(4.0)

    first_tick = session.tick()
    second_tick = session.tick()

    assert first_tick.current_word == "hello"
    assert second_tick.current_word == "world"


def test_set_speed_multiplier_beyond_the_tick_floor_advances_multiple_words_per_call():
    # ticks_per_word=2, so speed_multiplier=2 already hits the floor of 1 tick/word;
    # speed_multiplier=6 is 3x beyond that floor, so one tick() call should batch 3 raw
    # ticks internally (see ReadingSession._ticks_per_call) and land on the third word.
    session = _make_session(["a", "b", "c", "d", "e"], ticks_per_word=2)
    session.set_speed_multiplier(6.0)

    first_tick = session.tick()

    assert first_tick.current_word == "c"
    assert first_tick.words_read == 3


def test_set_speed_multiplier_beyond_the_tick_floor_stops_batching_at_book_end():
    session = _make_session(["a", "b"], ticks_per_word=2)
    session.set_speed_multiplier(6.0)

    tick_result = session.tick()

    assert tick_result.book_finished is True
    assert tick_result.current_word is None


def test_set_speed_multiplier_far_beyond_book_length_breaks_out_of_the_batch_loop_early():
    # ticks_per_call would be 10 raw ticks here, but the 2-word book finishes after only
    # 3 (word "a", word "b", then finished) - the batch loop must stop advancing once
    # book_finished flips True instead of wastefully looping through the rest.
    session = _make_session(["a", "b"], ticks_per_word=2)
    session.set_speed_multiplier(20.0)

    session.tick()

    assert session._tick_number == 3


def test_set_speed_multiplier_non_positive_raises_value_error():
    session = _make_session(["hello"])

    with pytest.raises(ValueError, match="speed multiplier must be positive"):
        session.set_speed_multiplier(0.0)


def test_pool_spike_rate_of_nan_is_never_produced_even_with_empty_pool():
    empty_pool_indices = dict(POOL_INDICES)
    empty_pool_indices["valence_positive"] = torch.tensor([], dtype=torch.int64)
    session = _make_session(["hello"], pool_indices=empty_pool_indices, ticks_per_word=1)

    tick_result = session.tick()

    assert not math.isnan(tick_result.region_activity["approach"])
    assert tick_result.region_activity["approach"] == 0.0


def test_tick_reads_reward_from_the_reward_dopamine_neurons_a_positive_word_excites():
    tick_result = _make_strong_input_session("good").tick()

    assert tick_result.emotions["reward"] == pytest.approx(1.0)
    assert tick_result.emotions["aversion"] == pytest.approx(0.0)
    assert tick_result.rating_0_10 == pytest.approx(10.0)


def test_tick_reads_aversion_from_the_punishment_dopamine_neurons_a_negative_word_excites():
    tick_result = _make_strong_input_session("bad").tick()

    assert tick_result.emotions["aversion"] == pytest.approx(1.0)
    assert tick_result.emotions["reward"] == pytest.approx(0.0)
    assert tick_result.rating_0_10 == pytest.approx(0.0)


def test_tick_reads_arousal_from_the_octopamine_neurons_any_charged_word_excites():
    tick_result = _make_strong_input_session("bad").tick()

    assert tick_result.emotions["arousal"] == pytest.approx(1.0)
    assert tick_result.region_activity["arousal"] == pytest.approx(1.0)


def test_tick_shows_no_emotion_for_a_neutral_word():
    tick_result = _make_strong_input_session("table").tick()

    assert tick_result.emotions == {"reward": 0.0, "aversion": 0.0, "arousal": 0.0}
    assert tick_result.rating_0_10 == pytest.approx(5.0)


def test_session_logs_book_loads_restarts_pauses_and_speed_changes(caplog):
    session = _make_session(["hello"])

    with caplog.at_level("INFO", logger="eternalfly.reading_session"):
        session.load_new_text(["new", "book"])
        session.restart()
        session.set_paused(True)
        session.set_paused(False)
        session.set_speed_multiplier(2.5)

    assert [record.getMessage() for record in caplog.records] == [
        "new book loaded: 2 words",
        "book restarted from its first word",
        "reading paused",
        "reading resumed",
        "speed multiplier set to 2.5",
    ]


def test_tick_logs_each_finished_word_at_debug(caplog):
    session = _make_session(["hello", "world"], ticks_per_word=1)

    with caplog.at_level("DEBUG", logger="eternalfly.word_activity_log"):
        session.tick()
        session.tick()

    assert len(caplog.records) == 1
    assert "'hello'" in caplog.records[0].getMessage()


def test_set_speed_multiplier_mid_book_keeps_the_word_being_read():
    session = _make_session(["one", "two", "three", "four"], ticks_per_word=4)
    for _ in range(5):
        session.tick()  # four ticks of "one", then the first tick of "two"

    session.set_speed_multiplier(4.0)  # one tick per word from now on

    assert session.tick().current_word == "two"
    assert session.tick().current_word == "three"


def test_paused_session_shows_the_new_books_first_word_after_load_new_text():
    session = _make_session(["old", "book"], ticks_per_word=1)
    session.tick()
    session.set_paused(True)

    session.load_new_text(["new", "story"])

    assert session.tick().current_word == "new"
    assert session.tick().current_word == "new"


def test_paused_session_shows_the_first_word_again_after_restart():
    session = _make_session(["first", "second"], ticks_per_word=1)
    session.tick()
    session.tick()
    session.set_paused(True)

    session.restart()

    assert session.tick().current_word == "first"
