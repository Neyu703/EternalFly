import numpy
import pytest

from eternalfly.connectome import (
    build_region_synapse_weights,
    build_signed_adjacency,
    keep_shared_neurons_in_majority_pool,
    select_pool_by_activity,
)


def test_build_signed_adjacency_single_excitatory_connection():
    pre_ids = numpy.array([100])
    post_ids = numpy.array([200])
    syn_counts = numpy.array([5])
    nt_types = numpy.array(["ach"])
    neuron_id_to_index = {100: 0, 200: 1}

    adjacency = build_signed_adjacency(pre_ids, post_ids, syn_counts, nt_types, neuron_id_to_index)

    assert adjacency.toarray().tolist() == [[0, 0], [5, 0]]


def test_build_signed_adjacency_inhibitory_connection_is_negative():
    pre_ids = numpy.array([100])
    post_ids = numpy.array([200])
    syn_counts = numpy.array([3])
    nt_types = numpy.array(["gaba"])
    neuron_id_to_index = {100: 0, 200: 1}

    adjacency = build_signed_adjacency(pre_ids, post_ids, syn_counts, nt_types, neuron_id_to_index)

    assert adjacency.toarray().tolist() == [[0, 0], [-3, 0]]


def test_build_signed_adjacency_sums_duplicate_pre_post_pairs():
    pre_ids = numpy.array([100, 100])
    post_ids = numpy.array([200, 200])
    syn_counts = numpy.array([5, 2])
    nt_types = numpy.array(["ach", "ach"])
    neuron_id_to_index = {100: 0, 200: 1}

    adjacency = build_signed_adjacency(pre_ids, post_ids, syn_counts, nt_types, neuron_id_to_index)

    assert adjacency.toarray().tolist() == [[0, 0], [7, 0]]


def test_build_signed_adjacency_raises_keyerror_for_unknown_pre_id():
    pre_ids = numpy.array([999])
    post_ids = numpy.array([200])
    syn_counts = numpy.array([5])
    nt_types = numpy.array(["ach"])
    neuron_id_to_index = {100: 0, 200: 1}

    with pytest.raises(KeyError):
        build_signed_adjacency(pre_ids, post_ids, syn_counts, nt_types, neuron_id_to_index)


def test_build_signed_adjacency_raises_keyerror_for_unknown_post_id():
    pre_ids = numpy.array([100])
    post_ids = numpy.array([999])
    syn_counts = numpy.array([5])
    nt_types = numpy.array(["ach"])
    neuron_id_to_index = {100: 0, 200: 1}

    with pytest.raises(KeyError):
        build_signed_adjacency(pre_ids, post_ids, syn_counts, nt_types, neuron_id_to_index)


def test_select_pool_by_activity_returns_top_fraction_sorted_descending():
    neuron_ids = numpy.array([10, 20, 30, 40])
    activity_counts = numpy.array([1, 4, 3, 2])

    selected_neuron_ids = select_pool_by_activity(neuron_ids, activity_counts, top_fraction=0.5)

    assert selected_neuron_ids.tolist() == [20, 30]


def test_select_pool_by_activity_raises_valueerror_for_zero_top_fraction():
    neuron_ids = numpy.array([10, 20])
    activity_counts = numpy.array([1, 2])

    with pytest.raises(ValueError):
        select_pool_by_activity(neuron_ids, activity_counts, top_fraction=0)


def test_select_pool_by_activity_raises_valueerror_for_top_fraction_above_one():
    neuron_ids = numpy.array([10, 20])
    activity_counts = numpy.array([1, 2])

    with pytest.raises(ValueError):
        select_pool_by_activity(neuron_ids, activity_counts, top_fraction=1.5)


def test_select_pool_by_activity_returns_at_least_one_when_fraction_rounds_to_zero():
    neuron_ids = numpy.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    activity_counts = numpy.array([5, 1, 2, 3, 4, 6, 7, 8, 9, 10])

    selected_neuron_ids = select_pool_by_activity(neuron_ids, activity_counts, top_fraction=0.01)

    assert selected_neuron_ids.tolist() == [100]


def test_select_pool_by_activity_breaks_ties_by_original_array_position():
    neuron_ids = numpy.array([10, 20, 30])
    activity_counts = numpy.array([5, 5, 1])

    selected_neuron_ids = select_pool_by_activity(neuron_ids, activity_counts, top_fraction=0.67)

    assert selected_neuron_ids.tolist() == [10, 20]


def test_keep_shared_neurons_in_majority_pool_keeps_a_shared_neuron_only_where_its_count_is_highest():
    pool_candidates = {
        "reward": (numpy.array([1, 2, 3]), numpy.array([10, 1, 5])),
        "punishment": (numpy.array([2, 3, 4]), numpy.array([9, 2, 7])),
    }

    disjoint_pools = keep_shared_neurons_in_majority_pool(pool_candidates)

    assert disjoint_pools["reward"][0].tolist() == [1, 3]
    assert disjoint_pools["reward"][1].tolist() == [10, 5]
    assert disjoint_pools["punishment"][0].tolist() == [2, 4]
    assert disjoint_pools["punishment"][1].tolist() == [9, 7]


def test_keep_shared_neurons_in_majority_pool_breaks_ties_toward_the_first_pool():
    pool_candidates = {
        "reward": (numpy.array([5]), numpy.array([3])),
        "punishment": (numpy.array([5]), numpy.array([3])),
    }

    disjoint_pools = keep_shared_neurons_in_majority_pool(pool_candidates)

    assert disjoint_pools["reward"][0].tolist() == [5]
    assert disjoint_pools["punishment"][0].tolist() == []


def test_keep_shared_neurons_in_majority_pool_handles_an_empty_pool():
    pool_candidates = {
        "reward": (numpy.array([], dtype=numpy.int64), numpy.array([], dtype=numpy.int64)),
        "punishment": (numpy.array([7]), numpy.array([1])),
    }

    disjoint_pools = keep_shared_neurons_in_majority_pool(pool_candidates)

    assert disjoint_pools["reward"][0].tolist() == []
    assert disjoint_pools["punishment"][0].tolist() == [7]


def test_build_region_synapse_weights_gives_each_neuron_its_share_of_the_regions_synapses():
    weights = build_region_synapse_weights(
        neuron_indices=numpy.array([0, 1, 1, 2]),
        region_indices=numpy.array([0, 0, 0, 1]),
        synapse_counts=numpy.array([2, 3, 3, 5]),
        region_count=3,
        neuron_count=3,
    )

    assert weights.toarray().tolist() == [[0.25, 0.75, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]]


def test_build_region_synapse_weights_times_spikes_is_the_synapse_weighted_mean_firing():
    weights = build_region_synapse_weights(
        numpy.array([0, 1]), numpy.array([0, 0]), numpy.array([1, 3]), region_count=1, neuron_count=2
    )

    assert (weights @ numpy.array([1.0, 0.0])).tolist() == [0.25]
