"""Pure, file-I/O-free transforms from extracted connectome arrays to simulation-ready structures."""

import numpy
import scipy.sparse


def _lookup_positions(ids: numpy.ndarray, sorted_root_ids: numpy.ndarray, sort_order: numpy.ndarray, id_label: str) -> numpy.ndarray:
    """Vectorized equivalent of `[neuron_id_to_index[i] for i in ids]` via binary search
    against sorted_root_ids (sort_order maps a sorted-array position back to that id's
    position in the original, unsorted root_ids array). Raises KeyError on the first id
    not present in sorted_root_ids."""
    ids = ids.astype(numpy.int64)
    positions_in_sorted = numpy.clip(numpy.searchsorted(sorted_root_ids, ids), 0, len(sorted_root_ids) - 1)
    found_mask = sorted_root_ids[positions_in_sorted] == ids
    if not found_mask.all():
        raise KeyError(f"{id_label} {int(ids[~found_mask][0])} not found in root_ids")
    return sort_order[positions_in_sorted]


def build_signed_adjacency(
    pre_ids: numpy.ndarray,
    post_ids: numpy.ndarray,
    syn_counts: numpy.ndarray,
    root_ids: numpy.ndarray,
    inhibitory_neuron_ids: numpy.ndarray,
) -> scipy.sparse.csr_matrix:
    """Build an N x N signed sparse adjacency matrix, entry [post_index, pre_index] = syn_count * sign.

    The sign is fixed per PRESYNAPTIC neuron (Dale's principle: a neuron releases the
    same neurotransmitter at every one of its synapses) - inhibitory_neuron_ids is the
    set of real FlyWire root ids whose own dominant neurotransmitter is GABAergic or
    glutamatergic (see neurotransmitters.neurotransmitter_sign and
    scripts/build_connectome_cache.py, which derives this set from each neuron's own
    top_nt annotation), not classified per edge/connection as before.

    Duplicate (pre_id, post_id) pairs have their weighted contributions summed. Raises
    KeyError if any pre/post id is not present in root_ids. Vectorized via
    numpy.searchsorted rather than a per-edge python dict lookup - the real connectome
    has ~10^7 edges, for which a python loop takes minutes.
    """
    neuron_count = len(root_ids)
    # root_ids loads as uint64 (numpy.load of proofread_root_ids_783.npy); searchsorted
    # between a uint64 array and an int64 one silently promotes both to float64 for the
    # comparison, which loses precision for real FlyWire ids (well beyond 2**53) and
    # produces false "not found" results - both sides must be the same integer dtype
    # (same pitfall documented on data_prep.filter_ids_to_known_set for numpy.isin).
    sort_order = numpy.argsort(root_ids.astype(numpy.int64))
    sorted_root_ids = root_ids.astype(numpy.int64)[sort_order]

    row_indices = _lookup_positions(post_ids, sorted_root_ids, sort_order, "post_id")
    column_indices = _lookup_positions(pre_ids, sorted_root_ids, sort_order, "pre_id")

    is_inhibitory = numpy.isin(pre_ids.astype(numpy.int64), inhibitory_neuron_ids.astype(numpy.int64))
    signs = numpy.where(is_inhibitory, -1, 1)
    weighted_values = numpy.asarray(syn_counts) * signs

    return scipy.sparse.coo_matrix(
        (weighted_values, (row_indices, column_indices)),
        shape=(neuron_count, neuron_count),
    ).tocsr()


def _sum_syn_count_by_post(pre_ids: numpy.ndarray, post_ids: numpy.ndarray, syn_counts: numpy.ndarray, source_ids: numpy.ndarray) -> dict[int, int]:
    """Sum syn_counts of every (pre, post) edge whose pre_id is in source_ids, grouped by
    post_id. Returns a plain {post_id: total_syn_count} dict (missing post_id = 0)."""
    is_from_source = numpy.isin(pre_ids.astype(numpy.int64), source_ids.astype(numpy.int64))
    matching_post_ids = post_ids[is_from_source].astype(numpy.int64)
    matching_syn_counts = numpy.asarray(syn_counts)[is_from_source]
    if len(matching_post_ids) == 0:
        return {}
    unique_post_ids, inverse = numpy.unique(matching_post_ids, return_inverse=True)
    totals = numpy.bincount(inverse, weights=matching_syn_counts.astype(numpy.float64))
    return dict(zip(unique_post_ids.tolist(), totals.tolist()))


def split_mbons_by_dopamine_input(
    pre_ids: numpy.ndarray,
    post_ids: numpy.ndarray,
    syn_counts: numpy.ndarray,
    mbon_ids: numpy.ndarray,
    reward_dan_ids: numpy.ndarray,
    punishment_dan_ids: numpy.ndarray,
) -> tuple[numpy.ndarray, numpy.ndarray]:
    """Split mbon_ids into (avoidance_mbon_ids, approach_mbon_ids) by which dopaminergic
    cluster synapses more heavily onto each MBON.

    Real mushroom-body logic (Aso et al. 2014 eLife; Cohn et al. 2015 Cell): a
    compartment's baseline MBON output drives approach if its dopaminergic input is
    from the PPL1 (punishment) cluster, or avoidance if from the PAM (reward) cluster -
    dopamine release there *depresses* the KC->MBON synapse (see plasticity.py),
    weakening that baseline drive during learning. MBONs with equal (including zero)
    measured input from both clusters are excluded from both groups rather than
    arbitrarily assigned a side.
    """
    reward_totals = _sum_syn_count_by_post(pre_ids, post_ids, syn_counts, reward_dan_ids)
    punishment_totals = _sum_syn_count_by_post(pre_ids, post_ids, syn_counts, punishment_dan_ids)

    avoidance_ids = []
    approach_ids = []
    for mbon_id in mbon_ids.astype(numpy.int64).tolist():
        reward_total = reward_totals.get(mbon_id, 0.0)
        punishment_total = punishment_totals.get(mbon_id, 0.0)
        if reward_total > punishment_total:
            avoidance_ids.append(mbon_id)
        elif punishment_total > reward_total:
            approach_ids.append(mbon_id)
    return numpy.array(avoidance_ids, dtype=numpy.int64), numpy.array(approach_ids, dtype=numpy.int64)


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


def neuropil_readout_matrix(
    pre_neuropil_ids: numpy.ndarray,
    pre_neuropil_names: numpy.ndarray,
    pre_neuropil_counts: numpy.ndarray,
    neuropil_names: list[str],
    root_ids: numpy.ndarray,
) -> scipy.sparse.csr_matrix:
    """Build a (len(neuropil_names) x len(root_ids)) sparse matrix where row r, column c
    is neuron c's SHARE of its own total real presynaptic output that lands in
    neuropil_names[r] (a neuron's weights across every region it projects into sum to
    at most 1). Multiplying this matrix by a per-neuron spike-count vector gives a
    presynapse-weighted "how active is this neuropil's real output right now" readout,
    replacing the old per-region "top 5% most active neurons" heuristic pool.

    pre_neuropil_ids/pre_neuropil_names/pre_neuropil_counts are the parallel columns of
    per_neuron_neuropil_count_pre_783.feather (pre_pt_root_id, neuropil, count) - every
    neuropil a neuron projects into, not just the ones in neuropil_names, so the
    per-neuron total (the share's denominator) reflects its real complete output.
    """
    pre_neuropil_ids = pre_neuropil_ids.astype(numpy.int64)
    pre_neuropil_counts = pre_neuropil_counts.astype(numpy.float64)

    unique_pre_ids, pre_inverse = numpy.unique(pre_neuropil_ids, return_inverse=True)
    total_output_by_unique_id = numpy.bincount(pre_inverse, weights=pre_neuropil_counts)

    # Same uint64/int64 searchsorted precision pitfall as build_signed_adjacency above.
    sort_order = numpy.argsort(root_ids.astype(numpy.int64))
    sorted_root_ids = root_ids.astype(numpy.int64)[sort_order]

    row_indices = []
    column_indices = []
    weighted_values = []
    for neuropil_row, neuropil_name in enumerate(neuropil_names):
        is_target = pre_neuropil_names == neuropil_name
        target_pre_ids = pre_neuropil_ids[is_target]
        target_counts = pre_neuropil_counts[is_target]
        if len(target_pre_ids) == 0:
            continue

        column_positions = numpy.clip(numpy.searchsorted(sorted_root_ids, target_pre_ids), 0, len(sorted_root_ids) - 1)
        is_proofread = sorted_root_ids[column_positions] == target_pre_ids
        target_pre_ids = target_pre_ids[is_proofread]
        target_counts = target_counts[is_proofread]
        column_positions = sort_order[column_positions[is_proofread]]

        totals = total_output_by_unique_id[numpy.searchsorted(unique_pre_ids, target_pre_ids)]
        shares = numpy.divide(target_counts, totals, out=numpy.zeros_like(target_counts), where=totals > 0)

        row_indices.extend([neuropil_row] * len(column_positions))
        column_indices.extend(column_positions.tolist())
        weighted_values.extend(shares.tolist())

    return scipy.sparse.coo_matrix(
        (weighted_values, (row_indices, column_indices)), shape=(len(neuropil_names), len(root_ids))
    ).tocsr()
