import numpy
import pytest

from eternalfly.connectome import (
    build_signed_adjacency,
    neuropil_readout_matrix,
    select_pool_by_activity,
    split_mbons_by_dopamine_input,
)

NO_INHIBITORY_NEURONS = numpy.array([], dtype=numpy.int64)


def test_build_signed_adjacency_single_excitatory_connection():
    pre_ids = numpy.array([100])
    post_ids = numpy.array([200])
    syn_counts = numpy.array([5])
    root_ids = numpy.array([100, 200])

    adjacency = build_signed_adjacency(pre_ids, post_ids, syn_counts, root_ids, NO_INHIBITORY_NEURONS)

    assert adjacency.toarray().tolist() == [[0, 0], [5, 0]]


def test_build_signed_adjacency_inhibitory_presynaptic_neuron_is_negative():
    pre_ids = numpy.array([100])
    post_ids = numpy.array([200])
    syn_counts = numpy.array([3])
    root_ids = numpy.array([100, 200])
    inhibitory_neuron_ids = numpy.array([100])

    adjacency = build_signed_adjacency(pre_ids, post_ids, syn_counts, root_ids, inhibitory_neuron_ids)

    assert adjacency.toarray().tolist() == [[0, 0], [-3, 0]]


def test_build_signed_adjacency_sign_is_fixed_per_presynaptic_neuron_across_all_its_edges():
    # Dale's principle: neuron 100 is inhibitory, so BOTH of its outgoing edges (to 200
    # and to 300) come out negative, not classified independently per edge.
    pre_ids = numpy.array([100, 100])
    post_ids = numpy.array([200, 300])
    syn_counts = numpy.array([4, 6])
    root_ids = numpy.array([100, 200, 300])
    inhibitory_neuron_ids = numpy.array([100])

    adjacency = build_signed_adjacency(pre_ids, post_ids, syn_counts, root_ids, inhibitory_neuron_ids)

    assert adjacency.toarray().tolist() == [[0, 0, 0], [-4, 0, 0], [-6, 0, 0]]


def test_build_signed_adjacency_sums_duplicate_pre_post_pairs():
    pre_ids = numpy.array([100, 100])
    post_ids = numpy.array([200, 200])
    syn_counts = numpy.array([5, 2])
    root_ids = numpy.array([100, 200])

    adjacency = build_signed_adjacency(pre_ids, post_ids, syn_counts, root_ids, NO_INHIBITORY_NEURONS)

    assert adjacency.toarray().tolist() == [[0, 0], [7, 0]]


def test_build_signed_adjacency_raises_keyerror_for_unknown_pre_id():
    pre_ids = numpy.array([999])
    post_ids = numpy.array([200])
    syn_counts = numpy.array([5])
    root_ids = numpy.array([100, 200])

    with pytest.raises(KeyError):
        build_signed_adjacency(pre_ids, post_ids, syn_counts, root_ids, NO_INHIBITORY_NEURONS)


def test_build_signed_adjacency_raises_keyerror_for_unknown_post_id():
    pre_ids = numpy.array([100])
    post_ids = numpy.array([999])
    syn_counts = numpy.array([5])
    root_ids = numpy.array([100, 200])

    with pytest.raises(KeyError):
        build_signed_adjacency(pre_ids, post_ids, syn_counts, root_ids, NO_INHIBITORY_NEURONS)


def test_build_signed_adjacency_handles_uint64_root_ids_with_flywire_scale_values():
    # Regression test: root_ids loads as uint64 (numpy.load of the real .npy file) while
    # pre_ids/post_ids/syn_counts come from a pyarrow int64 column. numpy.searchsorted
    # between a uint64 array and an int64 one silently promotes both to float64 for the
    # comparison, which loses precision for real FlyWire-scale ids (these two ids are
    # only 13 apart, well inside float64's ~2**53 precision floor for values this large)
    # and produced false "not found" KeyErrors even though the id genuinely was present.
    base_id = 720575940623350000  # a real FlyWire-scale id, far beyond 2**53
    pre_ids = numpy.array([base_id])
    post_ids = numpy.array([base_id + 13])
    syn_counts = numpy.array([5])
    root_ids = numpy.array([base_id, base_id + 13], dtype=numpy.uint64)

    adjacency = build_signed_adjacency(pre_ids, post_ids, syn_counts, root_ids, NO_INHIBITORY_NEURONS)

    assert adjacency.toarray().tolist() == [[0, 0], [5, 0]]


def test_build_signed_adjacency_root_ids_need_not_be_pre_sorted():
    pre_ids = numpy.array([200])
    post_ids = numpy.array([100])
    syn_counts = numpy.array([9])
    root_ids = numpy.array([300, 100, 200])  # deliberately unsorted, index 1/2 matter

    adjacency = build_signed_adjacency(pre_ids, post_ids, syn_counts, root_ids, NO_INHIBITORY_NEURONS)

    assert adjacency.toarray().tolist() == [[0, 0, 0], [0, 0, 9], [0, 0, 0]]


def test_split_mbons_by_dopamine_input_classifies_by_which_cluster_synapses_more():
    pre_ids = numpy.array([1, 1, 2, 2, 3])  # 1,2 = reward DANs, 3 = punishment DAN
    post_ids = numpy.array([10, 11, 10, 12, 11])  # 10,11,12 = MBONs
    syn_counts = numpy.array([5, 1, 2, 8, 10])
    mbon_ids = numpy.array([10, 11, 12])
    reward_dan_ids = numpy.array([1, 2])
    punishment_dan_ids = numpy.array([3])

    avoidance_ids, approach_ids = split_mbons_by_dopamine_input(
        pre_ids, post_ids, syn_counts, mbon_ids, reward_dan_ids, punishment_dan_ids
    )

    # mbon 10: reward=5+2=7, punishment=0 -> avoidance. mbon 11: reward=1, punishment=10 -> approach.
    # mbon 12: reward=8, punishment=0 -> avoidance.
    assert sorted(avoidance_ids.tolist()) == [10, 12]
    assert approach_ids.tolist() == [11]


def test_split_mbons_by_dopamine_input_handles_no_edges_from_one_cluster_at_all():
    pre_ids = numpy.array([1])
    post_ids = numpy.array([10])
    syn_counts = numpy.array([5])
    mbon_ids = numpy.array([10])
    reward_dan_ids = numpy.array([1])
    punishment_dan_ids = numpy.array([3])  # no edges anywhere are from neuron 3

    avoidance_ids, approach_ids = split_mbons_by_dopamine_input(
        pre_ids, post_ids, syn_counts, mbon_ids, reward_dan_ids, punishment_dan_ids
    )

    assert avoidance_ids.tolist() == [10]
    assert approach_ids.tolist() == []


def test_split_mbons_by_dopamine_input_excludes_mbon_with_equal_or_no_dopaminergic_input():
    pre_ids = numpy.array([1, 3])
    post_ids = numpy.array([10, 10])
    syn_counts = numpy.array([5, 5])  # tied
    mbon_ids = numpy.array([10, 20])  # 20 has no dopaminergic input at all
    reward_dan_ids = numpy.array([1])
    punishment_dan_ids = numpy.array([3])

    avoidance_ids, approach_ids = split_mbons_by_dopamine_input(
        pre_ids, post_ids, syn_counts, mbon_ids, reward_dan_ids, punishment_dan_ids
    )

    assert avoidance_ids.tolist() == []
    assert approach_ids.tolist() == []


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


def test_neuropil_readout_matrix_weights_by_share_of_neurons_own_total_output():
    # neuron 100: 3 synapses in ME_L, 1 in LO_L -> total 4, so ME_L gets 3/4, LO_L 1/4.
    # neuron 200: 5 synapses in ME_L only -> total 5, ME_L gets 5/5 = 1.0.
    pre_ids = numpy.array([100, 100, 200])
    neuropil_names_column = numpy.array(["ME_L", "LO_L", "ME_L"])
    counts = numpy.array([3, 1, 5])
    root_ids = numpy.array([100, 200])

    readout = neuropil_readout_matrix(pre_ids, neuropil_names_column, counts, ["ME_L", "LO_L"], root_ids)

    assert numpy.allclose(readout.toarray(), [[0.75, 1.0], [0.25, 0.0]])


def test_neuropil_readout_matrix_ignores_neuropils_not_in_root_ids():
    pre_ids = numpy.array([999])  # not a proofread neuron
    neuropil_names_column = numpy.array(["ME_L"])
    counts = numpy.array([10])
    root_ids = numpy.array([100])

    readout = neuropil_readout_matrix(pre_ids, neuropil_names_column, counts, ["ME_L"], root_ids)

    assert readout.toarray().tolist() == [[0.0]]


def test_neuropil_readout_matrix_returns_zero_row_for_a_neuropil_with_no_presynapses():
    pre_ids = numpy.array([100])
    neuropil_names_column = numpy.array(["ME_L"])
    counts = numpy.array([1])
    root_ids = numpy.array([100])

    readout = neuropil_readout_matrix(pre_ids, neuropil_names_column, counts, ["ME_L", "FB"], root_ids)

    assert readout.toarray().tolist() == [[1.0], [0.0]]


def test_neuropil_readout_matrix_handles_uint64_root_ids_with_flywire_scale_values():
    # Same uint64/int64 searchsorted precision regression as
    # test_build_signed_adjacency_handles_uint64_root_ids_with_flywire_scale_values.
    base_id = 720575940623350000
    pre_ids = numpy.array([base_id + 13])
    neuropil_names_column = numpy.array(["ME_L"])
    counts = numpy.array([7])
    root_ids = numpy.array([base_id, base_id + 13], dtype=numpy.uint64)

    readout = neuropil_readout_matrix(pre_ids, neuropil_names_column, counts, ["ME_L"], root_ids)

    assert readout.toarray().tolist() == [[0.0, 1.0]]
