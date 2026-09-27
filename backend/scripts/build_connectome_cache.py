"""One-off script: build the simulation-ready connectome cache from the downloaded
FlyWire files (Milestone 0's final step). Composes already-tested eternalfly functions;
not unit-tested itself, same convention as download_connectome.py."""

from pathlib import Path

import numpy
import scipy.sparse

from eternalfly.connectome import (
    build_region_synapse_weights,
    build_signed_adjacency,
    keep_shared_neurons_in_majority_pool,
    select_pool_by_activity,
)
from eternalfly.data_prep import (
    aggregate_connections_by_neuron_pair,
    aggregate_neuron_activity_by_neuropil,
    aggregate_neurotransmitter_by_neuron,
    build_neuron_index,
    dominant_neurotransmitter_labels,
    filter_ids_to_known_set,
    load_feather_table,
    load_root_ids,
    neuropil_synapse_count_rows,
)
from eternalfly.neuropils import ALL_NEUROPIL_NAMES

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CACHE_DIR = DATA_DIR / "cache"

# Neuropil groups per activity-selected pool: the optic lobe (visual) for sensory input.
POOL_NEUROPILS = {
    "sensory_input": ["ME_L", "ME_R", "LO_L", "LO_R"],
}
POOL_TOP_FRACTION = 0.05

# The medial mushroom body lobe is real Drosophila's reward-learning compartment
# (PAM cluster dopaminergic input); the vertical lobe is the punishment-learning
# compartment (PPL1 cluster). Picks out the real dopaminergic presynaptic neurons of
# each lobe, which words excite by their sentiment and which are read back out as the
# fly's reward and aversion (see reading_session.py and
# text_encoder.project_valence_to_currents).
VALENCE_DOPAMINERGIC_NEUROPILS = {
    "valence_positive": ["MB_ML_L", "MB_ML_R"],
    "valence_negative": ["MB_VL_L", "MB_VL_R"],
}

# Octopamine is Drosophila's real arousal/stress neuromodulator (the functional
# analog of noradrenaline): broadly excitatory, promotes wakefulness/alertness, and
# is a well-established real input to the central complex — the same role dopamine
# plays for reward/punishment, but for arousal. Read back out as the fly's arousal.
AROUSAL_OCTOPAMINERGIC_NEUROPILS = {
    "arousal_input": ["EB", "FB"],
}


def build_pool_indices(
    post_neuropil_table,
    root_ids: numpy.ndarray,
    neuron_id_to_index: dict[int, int],
    pool_neuropils: dict[str, list[str]],
) -> dict[str, numpy.ndarray]:
    """Select each pool's neuron indices: top POOL_TOP_FRACTION most postsynaptically
    active proofread neurons within that pool's target neuropils."""
    pool_indices = {}
    for pool_name, target_neuropils in pool_neuropils.items():
        candidate_ids, candidate_counts = aggregate_neuron_activity_by_neuropil(
            post_neuropil_table, target_neuropils, "post_pt_root_id"
        )
        known_ids, known_counts = filter_ids_to_known_set(candidate_ids, candidate_counts, root_ids)
        selected_ids = select_pool_by_activity(known_ids, known_counts, POOL_TOP_FRACTION)
        pool_indices[pool_name] = numpy.array(
            [neuron_id_to_index[int(neuron_id)] for neuron_id in selected_ids]
        )
    return pool_indices


def build_neurotransmitter_filtered_pool_indices(
    connections_table,
    pre_neuropil_table,
    root_ids: numpy.ndarray,
    neuron_id_to_index: dict[int, int],
    neurotransmitter_label: str,
    pool_neuropils: dict[str, list[str]],
) -> dict[str, numpy.ndarray]:
    """Select each pool's neuron indices: proofread presynaptic neurons whose own
    dominant neurotransmitter is neurotransmitter_label (e.g. "da" for dopaminergic
    reward/punishment, "oct" for octopaminergic arousal), restricted to those
    synapsing into that pool's target neuropils. Unlike build_pool_indices, keeps
    every matching neuron rather than a top-activity fraction — being a real,
    specific-neurotransmitter presynaptic partner of that compartment is already a
    strong, biologically-motivated filter (a few hundred neurons here, not the
    thousands build_pool_indices narrows down from). A neuron synapsing into several
    pools' neuropils is kept only in the pool holding most of those synapses (see
    connectome.keep_shared_neurons_in_majority_pool), so the pools never overlap."""
    per_neuron_nt_table = aggregate_neurotransmitter_by_neuron(connections_table, "pre_pt_root_id")
    neuron_nt_labels = dominant_neurotransmitter_labels(per_neuron_nt_table)
    matching_neuron_ids = per_neuron_nt_table["pre_pt_root_id"].to_numpy()[neuron_nt_labels == neurotransmitter_label]

    pool_candidates = {}
    for pool_name, target_neuropils in pool_neuropils.items():
        candidate_ids, candidate_counts = aggregate_neuron_activity_by_neuropil(
            pre_neuropil_table, target_neuropils, "pre_pt_root_id"
        )
        matches_neurotransmitter = numpy.isin(
            candidate_ids.astype(numpy.int64), matching_neuron_ids.astype(numpy.int64)
        )
        pool_candidates[pool_name] = (candidate_ids[matches_neurotransmitter], candidate_counts[matches_neurotransmitter])

    pool_indices = {}
    for pool_name, (candidate_ids, candidate_counts) in keep_shared_neurons_in_majority_pool(pool_candidates).items():
        known_ids, _known_counts = filter_ids_to_known_set(candidate_ids, candidate_counts, root_ids)
        pool_indices[pool_name] = numpy.array([neuron_id_to_index[int(neuron_id)] for neuron_id in known_ids])
    return pool_indices


def main() -> None:
    """Build and cache the signed adjacency matrix and the neuron pools."""
    CACHE_DIR.mkdir(exist_ok=True)

    root_ids = load_root_ids(DATA_DIR / "proofread_root_ids_783.npy")
    neuron_id_to_index = build_neuron_index(root_ids)
    print("neuron count:", len(root_ids))

    connections_table = load_feather_table(DATA_DIR / "proofread_connections_783.feather")
    aggregated_table = aggregate_connections_by_neuron_pair(connections_table)
    neurotransmitter_labels = dominant_neurotransmitter_labels(aggregated_table)
    print("aggregated edges:", aggregated_table.num_rows)

    adjacency_matrix = build_signed_adjacency(
        aggregated_table["pre_pt_root_id"].to_numpy(),
        aggregated_table["post_pt_root_id"].to_numpy(),
        aggregated_table["syn_count"].to_numpy(),
        neurotransmitter_labels,
        neuron_id_to_index,
    )
    scipy.sparse.save_npz(CACHE_DIR / "adjacency.npz", adjacency_matrix)
    print("saved adjacency:", adjacency_matrix.shape, "nnz:", adjacency_matrix.nnz)

    post_neuropil_table = load_feather_table(DATA_DIR / "per_neuron_neuropil_count_post_783.feather")
    pool_indices = build_pool_indices(post_neuropil_table, root_ids, neuron_id_to_index, POOL_NEUROPILS)

    pre_neuropil_table = load_feather_table(DATA_DIR / "per_neuron_neuropil_count_pre_783.feather")
    pool_indices.update(
        build_neurotransmitter_filtered_pool_indices(
            connections_table, pre_neuropil_table, root_ids, neuron_id_to_index, "da", VALENCE_DOPAMINERGIC_NEUROPILS
        )
    )
    pool_indices.update(
        build_neurotransmitter_filtered_pool_indices(
            connections_table, pre_neuropil_table, root_ids, neuron_id_to_index, "oct", AROUSAL_OCTOPAMINERGIC_NEUROPILS
        )
    )

    numpy.savez(CACHE_DIR / "pool_indices.npz", **pool_indices)
    for pool_name, indices in pool_indices.items():
        print(f"pool {pool_name}: {len(indices)} neurons")

    region_rows = [
        neuropil_synapse_count_rows(table, id_column, neuron_id_to_index, ALL_NEUROPIL_NAMES)
        for table, id_column in ((pre_neuropil_table, "pre_pt_root_id"), (post_neuropil_table, "post_pt_root_id"))
    ]
    region_synapse_weights = build_region_synapse_weights(
        numpy.concatenate([rows[0] for rows in region_rows]),
        numpy.concatenate([rows[1] for rows in region_rows]),
        numpy.concatenate([rows[2] for rows in region_rows]),
        region_count=len(ALL_NEUROPIL_NAMES),
        neuron_count=len(root_ids),
    )
    # Rows follow ALL_NEUROPIL_NAMES; see ReadingSession for how they become region firing rates.
    scipy.sparse.save_npz(CACHE_DIR / "region_synapse_weights.npz", region_synapse_weights)
    print("region synapse weights:", region_synapse_weights.shape, "nnz:", region_synapse_weights.nnz)


if __name__ == "__main__":
    main()
