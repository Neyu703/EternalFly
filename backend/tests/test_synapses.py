import numpy
import pytest
import scipy.sparse
import torch

from eternalfly.synapses import (
    EventDrivenSynapses,
    advance_delay_buffer,
    build_event_driven_synapses,
    build_signed_edge_weights,
    propagate_spikes,
)


def _synapses(pre_major_csr: scipy.sparse.csr_matrix) -> EventDrivenSynapses:
    return build_event_driven_synapses(pre_major_csr, device="cpu")


def test_propagate_spikes_no_firing_neurons_returns_all_zero():
    # neuron 0 -> neuron 1 (weight 5); nobody fires.
    matrix = scipy.sparse.csr_matrix(numpy.array([[0.0, 5.0], [0.0, 0.0]]))
    synapses = _synapses(matrix)
    spikes = torch.tensor([0.0, 0.0])

    increment = propagate_spikes(synapses, spikes)

    assert increment.tolist() == [0.0, 0.0]


def test_propagate_spikes_single_firing_neuron_scatters_its_row():
    # neuron 0 -> neuron 1 (weight 5) and neuron 2 (weight -3); neuron 0 fires.
    matrix = scipy.sparse.csr_matrix(numpy.array([[0.0, 5.0, -3.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]))
    synapses = _synapses(matrix)
    spikes = torch.tensor([1.0, 0.0, 0.0])

    increment = propagate_spikes(synapses, spikes)

    assert increment.tolist() == [0.0, 5.0, -3.0]


def test_propagate_spikes_multiple_firing_neurons_sums_into_shared_postsynaptic_targets():
    # neuron 0 -> neuron 2 (weight 4); neuron 1 -> neuron 2 (weight 6); both fire.
    matrix = scipy.sparse.csr_matrix(numpy.array([[0.0, 0.0, 4.0], [0.0, 0.0, 6.0], [0.0, 0.0, 0.0]]))
    synapses = _synapses(matrix)
    spikes = torch.tensor([1.0, 1.0, 0.0])

    increment = propagate_spikes(synapses, spikes)

    assert increment.tolist() == [0.0, 0.0, 10.0]


def test_propagate_spikes_firing_neuron_with_no_outgoing_synapses_contributes_nothing():
    matrix = scipy.sparse.csr_matrix(numpy.array([[0.0, 0.0], [0.0, 0.0]]))
    synapses = _synapses(matrix)
    spikes = torch.tensor([1.0, 0.0])

    increment = propagate_spikes(synapses, spikes)

    assert increment.tolist() == [0.0, 0.0]


def test_propagate_spikes_matches_dense_matrix_vector_product_on_a_larger_random_case():
    generator = numpy.random.default_rng(0)
    dense_weights = generator.normal(size=(20, 20)) * (generator.random((20, 20)) < 0.2)
    matrix = scipy.sparse.csr_matrix(dense_weights)
    synapses = _synapses(matrix)
    firing_mask = generator.random(20) < 0.3
    spikes = torch.as_tensor(firing_mask.astype(numpy.float32))

    increment = propagate_spikes(synapses, spikes)

    # weight_matrix[post, pre] convention would be dense_weights.T @ firing_mask; here
    # rows are presynaptic, so it's firing_mask @ dense_weights (post = column).
    expected = firing_mask.astype(numpy.float64) @ dense_weights
    assert numpy.allclose(increment.numpy(), expected, atol=1e-5)


def test_build_event_driven_synapses_wraps_scipy_csr_correctly():
    matrix = scipy.sparse.csr_matrix(numpy.array([[0.0, 2.0], [3.0, 0.0]]))

    synapses = build_event_driven_synapses(matrix, device="cpu")

    assert synapses.post_neuron_count == 2
    assert synapses.crow_indices.tolist() == matrix.indptr.tolist()
    assert synapses.col_indices.tolist() == matrix.indices.tolist()
    assert synapses.values.tolist() == pytest.approx(matrix.data.tolist())


def test_advance_delay_buffer_drops_head_and_appends_new_tail():
    tail = torch.tensor([9.0])
    head = torch.tensor([1.0])
    middle = torch.tensor([2.0])
    pending = (head, middle)

    new_pending = advance_delay_buffer(pending, tail)

    assert new_pending == (middle, tail)
    assert new_pending[0] is middle  # no copy - same tensor object moved to the front
    assert new_pending[1] is tail


def test_advance_delay_buffer_preserves_single_slot_buffer_length():
    pending = (torch.tensor([1.0]),)
    new_pending = advance_delay_buffer(pending, torch.tensor([2.0]))

    assert len(new_pending) == 1
    assert new_pending[0].tolist() == [2.0]


def test_build_signed_edge_weights_applies_sign_and_scale():
    syn_counts = numpy.array([10, 20])
    signs = numpy.array([1, -1])

    weights = build_signed_edge_weights(syn_counts, signs, per_synapse_weight=0.04)

    assert numpy.allclose(weights, [0.4, -0.8])
