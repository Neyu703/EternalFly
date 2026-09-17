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
        pre_id = int(pre_id)
        post_id = int(post_id)
        if pre_id not in neuron_id_to_index:
            raise KeyError(f"pre_id {pre_id} not found in neuron_id_to_index")
        if post_id not in neuron_id_to_index:
            raise KeyError(f"post_id {post_id} not found in neuron_id_to_index")

        row_indices.append(neuron_id_to_index[post_id])
        column_indices.append(neuron_id_to_index[pre_id])
        weighted_values.append(syn_count * neurotransmitter_sign(nt_type))

    return scipy.sparse.coo_matrix(
        (weighted_values, (row_indices, column_indices)),
        shape=(neuron_count, neuron_count),
    ).tocsr()


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
