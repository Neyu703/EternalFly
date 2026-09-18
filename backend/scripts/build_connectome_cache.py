"""One-off script: build the simulation-ready connectome cache from the downloaded
FlyWire files (Milestone 0's final step). Composes already-tested eternalfly functions;
not unit-tested itself, same convention as download_connectome.py."""

from pathlib import Path

import numpy
import scipy.sparse

from eternalfly.connectome import build_signed_adjacency, select_pool_by_activity
from eternalfly.data_prep import (
    aggregate_connections_by_neuron_pair,
    aggregate_neuron_activity_by_neuropil,
    build_neuron_index,
    dominant_neurotransmitter_labels,
    filter_ids_to_known_set,
    load_feather_table,
    load_root_ids,
)
from eternalfly.neuropils import ALL_NEUROPIL_NAMES

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CACHE_DIR = DATA_DIR / "cache"

# Neuropil groups per pool, chosen for biological plausibility (see plan docs):
# optic lobe (visual) for sensory input, mushroom body medial/vertical lobes for
# reward/punishment valence, central complex for arousal.
POOL_NEUROPILS = {
    "sensory_input": ["ME_L", "ME_R", "LO_L", "LO_R"],
    "approach": ["MB_ML_L", "MB_ML_R"],
    "avoidance": ["MB_VL_L", "MB_VL_R"],
    "arousal": ["EB", "FB"],
}
POOL_TOP_FRACTION = 0.05


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


def main() -> None:
    """Build and cache the signed adjacency matrix and the four neuron pools."""
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
    numpy.savez(CACHE_DIR / "pool_indices.npz", **pool_indices)
    for pool_name, indices in pool_indices.items():
        print(f"pool {pool_name}: {len(indices)} neurons")

    neuropil_single_name_groups = {name: [name] for name in ALL_NEUROPIL_NAMES}
    neuropil_pool_indices = build_pool_indices(
        post_neuropil_table, root_ids, neuron_id_to_index, neuropil_single_name_groups
    )
    numpy.savez(CACHE_DIR / "neuropil_pool_indices.npz", **neuropil_pool_indices)
    print(f"neuropil pools: {len(neuropil_pool_indices)} regions")


if __name__ == "__main__":
    main()
