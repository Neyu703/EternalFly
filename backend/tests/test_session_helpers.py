import math

import pytest
import torch

from eternalfly.session_helpers import (
    compute_pool_spike_rate,
    inject_currents_at_indices,
    is_new_word_tick,
    word_index_for_tick,
)


def test_compute_pool_spike_rate_partial_spiking_returns_correct_fraction():
    spikes = torch.tensor([1.0, 0.0, 1.0, 0.0, 0.0])
    pool_indices = torch.tensor([0, 1, 2, 3])

    spike_rate = compute_pool_spike_rate(spikes, pool_indices)

    assert spike_rate == pytest.approx(0.5)


def test_compute_pool_spike_rate_none_spiking_returns_zero():
    spikes = torch.tensor([0.0, 0.0, 0.0, 1.0])
    pool_indices = torch.tensor([0, 1, 2])

    spike_rate = compute_pool_spike_rate(spikes, pool_indices)

    assert spike_rate == pytest.approx(0.0)


def test_compute_pool_spike_rate_all_spiking_returns_one():
    spikes = torch.tensor([1.0, 1.0, 1.0, 0.0])
    pool_indices = torch.tensor([0, 1, 2])

    spike_rate = compute_pool_spike_rate(spikes, pool_indices)

    assert spike_rate == pytest.approx(1.0)


def test_compute_pool_spike_rate_empty_pool_indices_returns_zero_not_nan():
    spikes = torch.tensor([1.0, 0.0, 1.0])
    pool_indices = torch.tensor([], dtype=torch.int64)

    spike_rate = compute_pool_spike_rate(spikes, pool_indices)

    assert spike_rate == 0.0
    assert not math.isnan(spike_rate)


def test_inject_currents_at_indices_places_values_at_correct_positions():
    pool_indices = torch.tensor([1, 3])
    currents = torch.tensor([5.0, 7.0])

    injected = inject_currents_at_indices(
        neuron_count=5, pool_indices=pool_indices, currents=currents, device="cpu"
    )

    assert injected.tolist() == [0.0, 5.0, 0.0, 7.0, 0.0]


def test_inject_currents_at_indices_mismatched_lengths_raises_value_error():
    pool_indices = torch.tensor([0, 1])
    currents = torch.tensor([5.0])

    with pytest.raises(ValueError):
        inject_currents_at_indices(
            neuron_count=5, pool_indices=pool_indices, currents=currents, device="cpu"
        )


def test_inject_currents_at_indices_empty_pool_returns_all_zero_tensor():
    pool_indices = torch.tensor([], dtype=torch.int64)
    currents = torch.tensor([], dtype=torch.float32)

    injected = inject_currents_at_indices(
        neuron_count=4, pool_indices=pool_indices, currents=currents, device="cpu"
    )

    assert injected.shape == (4,)
    assert injected.tolist() == [0.0, 0.0, 0.0, 0.0]


def test_is_new_word_tick_tick_zero_is_always_a_new_word_tick():
    assert is_new_word_tick(tick_number=0, ticks_per_word=5) is True


@pytest.mark.parametrize("tick_number", [0, 5, 10])
def test_is_new_word_tick_multiples_of_ticks_per_word_are_new_word_ticks(tick_number):
    assert is_new_word_tick(tick_number=tick_number, ticks_per_word=5) is True


@pytest.mark.parametrize("tick_number", [1, 2, 3, 4, 6])
def test_is_new_word_tick_non_multiples_of_ticks_per_word_are_not_new_word_ticks(tick_number):
    assert is_new_word_tick(tick_number=tick_number, ticks_per_word=5) is False


@pytest.mark.parametrize("tick_number", [0, 1, 2, 3, 4])
def test_word_index_for_tick_returns_zero_for_first_word_ticks(tick_number):
    assert word_index_for_tick(tick_number=tick_number, ticks_per_word=5) == 0


@pytest.mark.parametrize("tick_number", [5, 6, 7, 8, 9])
def test_word_index_for_tick_returns_one_for_second_word_ticks(tick_number):
    assert word_index_for_tick(tick_number=tick_number, ticks_per_word=5) == 1


@pytest.mark.parametrize("ticks_per_word", [0, -1])
def test_is_new_word_tick_non_positive_ticks_per_word_raises_value_error(ticks_per_word):
    with pytest.raises(ValueError):
        is_new_word_tick(tick_number=0, ticks_per_word=ticks_per_word)


@pytest.mark.parametrize("ticks_per_word", [0, -1])
def test_word_index_for_tick_non_positive_ticks_per_word_raises_value_error(ticks_per_word):
    with pytest.raises(ValueError):
        word_index_for_tick(tick_number=0, ticks_per_word=ticks_per_word)
