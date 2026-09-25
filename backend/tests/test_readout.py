import numpy
import pytest
import torch

from eternalfly.readout import accumulate_spike_counts, build_group_readout_matrix, readout_rates


def test_accumulate_spike_counts_sums_across_calls():
    running_counts = torch.tensor([0.0, 0.0])
    running_counts = accumulate_spike_counts(running_counts, torch.tensor([1.0, 0.0]))
    running_counts = accumulate_spike_counts(running_counts, torch.tensor([1.0, 1.0]))

    assert running_counts.tolist() == [2.0, 1.0]


def test_readout_rates_divides_by_step_count():
    readout_matrix = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    spike_counts = torch.tensor([4.0, 2.0])

    rates = readout_rates(readout_matrix, spike_counts, step_count=4)

    assert rates.tolist() == pytest.approx([1.0, 0.5])


def test_readout_rates_raises_on_non_positive_step_count():
    readout_matrix = torch.tensor([[1.0]])
    spike_counts = torch.tensor([1.0])

    with pytest.raises(ValueError, match="step_count must be positive"):
        readout_rates(readout_matrix, spike_counts, step_count=0)


def test_build_group_readout_matrix_row_is_uniform_mean_over_members():
    group_indices = {"escape": numpy.array([0, 2])}

    row_names, matrix = build_group_readout_matrix(group_indices, neuron_count=4)

    assert row_names == ["escape"]
    assert matrix.tolist() == [[0.5, 0.0, 0.5, 0.0]]


def test_build_group_readout_matrix_multiple_groups_preserve_dict_order():
    group_indices = {"a": numpy.array([0]), "b": numpy.array([1, 2])}

    row_names, matrix = build_group_readout_matrix(group_indices, neuron_count=3)

    assert row_names == ["a", "b"]
    assert matrix.tolist() == [[1.0, 0.0, 0.0], [0.0, 0.5, 0.5]]


def test_build_group_readout_matrix_empty_group_gets_all_zero_row():
    group_indices = {"empty": numpy.array([], dtype=numpy.int64)}

    _row_names, matrix = build_group_readout_matrix(group_indices, neuron_count=3)

    assert matrix.tolist() == [[0.0, 0.0, 0.0]]


def test_readout_rates_end_to_end_with_group_readout_matrix():
    # 4 neurons, group "escape" = {0, 1}; both spike every step for 5 steps -> rate 1.0.
    group_indices = {"escape": numpy.array([0, 1])}
    _row_names, readout_matrix = build_group_readout_matrix(group_indices, neuron_count=4)
    spike_counts = torch.zeros(4)
    for _ in range(5):
        spike_counts = accumulate_spike_counts(spike_counts, torch.tensor([1.0, 1.0, 0.0, 0.0]))

    rates = readout_rates(readout_matrix, spike_counts, step_count=5)

    assert rates.tolist() == pytest.approx([1.0])
