"""Pure, file-I/O-free transforms from extracted connectome arrays to simulation-ready structures."""

import numpy
import scipy.sparse

from eternalfly.neurotransmitters import neurotransmitter_sign


def build_signed_adjacency(
    pre_ids: numpy.ndarray,
    post_ids: numpy.ndarray,
    syn_counts: numpy.ndarray,
    nt_types: numpy.ndarray,
    neuron_id_to_index: dict[int, int],
) -> scipy.sparse.csr_matrix:
    """Build an N x N signed sparse adjacency matrix, entry [post_index, pre_index] = syn_count * sign.

    Duplicate (pre_id, post_id) pairs have their weighted contributions summed. Raises KeyError
    if any pre/post id is not present in neuron_id_to_index.
    """
    neuron_count = len(neuron_id_to_index)
    row_indices = []
    column_indices = []
    weighted_values = []

    for pre_id, post_id, syn_count, nt_type in zip(pre_ids, post_ids, syn_counts, nt_types):
        column_indices.append(_index_of(pre_id, neuron_id_to_index, "pre"))
        row_indices.append(_index_of(post_id, neuron_id_to_index, "post"))
        weighted_values.append(syn_count * neurotransmitter_sign(nt_type))

    return scipy.sparse.coo_matrix(
        (weighted_values, (row_indices, column_indices)),
        shape=(neuron_count, neuron_count),
    ).tocsr()


def _index_of(neuron_id, neuron_id_to_index: dict[int, int], role: str) -> int:
    """neuron_id's index in neuron_id_to_index. Raises KeyError naming role ("pre" or
    "post") when it isn't there."""
    neuron_id = int(neuron_id)
    if neuron_id not in neuron_id_to_index:
        raise KeyError(f"{role}_id {neuron_id} not found in neuron_id_to_index")
    return neuron_id_to_index[neuron_id]


def select_pool_by_activity(
    neuron_ids: numpy.ndarray, activity_counts: numpy.ndarray, top_fraction: float
) -> numpy.ndarray:
    """Return the ids of the top top_fraction most active neurons, sorted descending by activity.

    top_fraction must be in (0, 1]. Always returns at least one neuron, even if
    top_fraction * len(neuron_ids) rounds to zero. Ties keep the original array order.
    """
    if not 0 < top_fraction <= 1:
        raise ValueError("top_fraction must be in (0, 1]")

    descending_order = numpy.argsort(-activity_counts, kind="stable")
    selected_neuron_count = max(1, round(top_fraction * len(neuron_ids)))

    return neuron_ids[descending_order[:selected_neuron_count]]


def keep_shared_neurons_in_majority_pool(
    pool_candidates: dict[str, tuple[numpy.ndarray, numpy.ndarray]],
) -> dict[str, tuple[numpy.ndarray, numpy.ndarray]]:
    """Make pools disjoint: pool_candidates maps each pool name to parallel (neuron_ids,
    counts) arrays, and a neuron listed in several pools stays only in the pool where its
    count is highest (ties go to the pool listed first). Returns the same shape, with each
    pool's arrays filtered in their original order.

    Used for the dopaminergic valence pools: a dopamine neuron with most of its output in
    the reward-coding medial lobe and a few stray synapses in the punishment-coding
    vertical lobe belongs to the reward pool only, so a negative word never excites it."""
    majority_pool_by_neuron: dict[int, tuple[str, float]] = {}
    for pool_name, (neuron_ids, counts) in pool_candidates.items():
        for neuron_id, count in zip(neuron_ids.tolist(), counts.tolist()):
            if neuron_id not in majority_pool_by_neuron or count > majority_pool_by_neuron[neuron_id][1]:
                majority_pool_by_neuron[neuron_id] = (pool_name, count)

    disjoint_pools = {}
    for pool_name, (neuron_ids, counts) in pool_candidates.items():
        is_kept = numpy.array(
            [majority_pool_by_neuron[neuron_id][0] == pool_name for neuron_id in neuron_ids.tolist()], dtype=bool
        )
        disjoint_pools[pool_name] = (neuron_ids[is_kept], counts[is_kept])
    return disjoint_pools


def build_region_synapse_weights(
    neuron_indices: numpy.ndarray,
    region_indices: numpy.ndarray,
    synapse_counts: numpy.ndarray,
    region_count: int,
    neuron_count: int,
) -> scipy.sparse.csr_matrix:
    """Build a region_count x neuron_count matrix whose row r holds each neuron's share of
    region r's synapses (parallel input arrays; duplicate (region, neuron) entries, such as
    a neuron's pre- and postsynapses there, are summed). Row r times a spike vector is then
    the synapse-weighted mean firing of every neuron with synapses in region r: each neuron
    counts as much as it is present there, the way a neuropil's recorded activity mixes
    all the neurites passing through it. Rows of regions without synapses stay all zero."""
    region_by_neuron_counts = scipy.sparse.coo_matrix(
        (synapse_counts.astype(numpy.float64), (region_indices, neuron_indices)), shape=(region_count, neuron_count)
    ).tocsr()
    region_totals = numpy.asarray(region_by_neuron_counts.sum(axis=1)).ravel()
    inverse_totals = numpy.divide(1.0, region_totals, out=numpy.zeros_like(region_totals), where=region_totals > 0)
    return (scipy.sparse.diags(inverse_totals) @ region_by_neuron_counts).tocsr()
