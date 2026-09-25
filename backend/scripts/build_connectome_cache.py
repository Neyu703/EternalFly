"""One-off script: build the simulation-ready connectome cache (real Dale-signed
synapses, named cell groups, neuropil readout, neuron scene positions) from the
downloaded FlyWire files and cell-type annotations. Composes already-tested eternalfly
functions; not unit-tested itself, same convention as download_connectome.py."""

from pathlib import Path

import numpy
import scipy.sparse

from eternalfly.cell_groups import (
    BEHAVIOR_SELECTORS,
    DOPAMINE_PUNISHMENT_SELECTOR,
    DOPAMINE_REWARD_SELECTOR,
    KENYON_CELL_SELECTOR,
    MBON_SELECTOR,
    OCTOPAMINE_AROUSAL_SELECTOR,
    SENSORY_CHANNEL_SELECTORS,
    select_cell_group,
    select_remaining_olfactory_receptor_neurons,
)
from eternalfly.connectome import build_signed_adjacency, neuropil_readout_matrix, split_mbons_by_dopamine_input
from eternalfly.data_prep import (
    aggregate_connections_by_neuron_pair,
    build_neuron_index,
    filter_edges_to_known_neurons,
    load_feather_table,
    load_neuron_annotations,
    load_root_ids,
)
from eternalfly.geometry import neuron_scene_positions
from eternalfly.neuropils import ALL_NEUROPIL_NAMES
from eternalfly.neurotransmitters import neurotransmitter_sign

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CACHE_DIR = DATA_DIR / "cache"
ANNOTATIONS_PATH = DATA_DIR / "Supplemental_file1_neuron_annotations.tsv"
FRONTEND_MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "frontend" / "public" / "models"


def build_inhibitory_neuron_ids(annotations) -> numpy.ndarray:
    """Return the real root ids whose own dominant neurotransmitter (the annotations'
    top_nt column) is inhibitory (GABAergic/glutamatergic) - see
    neurotransmitters.neurotransmitter_sign. Used for Dale's-principle signed weights:
    one sign per presynaptic neuron, not per connection (see connectome.build_signed_adjacency)."""
    annotation_ids = numpy.asarray(annotations["root_id"]).astype(numpy.int64)
    top_nt_values = numpy.asarray(annotations["top_nt"])
    is_inhibitory = numpy.array([neurotransmitter_sign(nt) == -1 for nt in top_nt_values])
    return annotation_ids[is_inhibitory]


def build_cell_groups(
    annotations,
    edge_pre_ids: numpy.ndarray,
    edge_post_ids: numpy.ndarray,
    edge_syn_counts: numpy.ndarray,
    root_ids: numpy.ndarray,
    neuron_id_to_index: dict[int, int],
) -> dict[str, numpy.ndarray]:
    """Select every real named neuron group (sensory input channels, behavior readouts,
    reward/punishment/arousal pools, MBONs split by which dopaminergic cluster feeds
    them, Kenyon cells for KC->MBON plasticity, and the remaining olfactory glomeruli
    for the semantic-odor fallback - see cell_groups.py), converted from root ids to
    matrix indices."""
    root_id_groups: dict[str, numpy.ndarray] = {}
    for channel_name, selector in SENSORY_CHANNEL_SELECTORS.items():
        root_id_groups[f"sensory_{channel_name}"] = select_cell_group(annotations, root_ids, selector)
    for behavior_name, selector in BEHAVIOR_SELECTORS.items():
        root_id_groups[f"behavior_{behavior_name}"] = select_cell_group(annotations, root_ids, selector)
    root_id_groups["reward_pam"] = select_cell_group(annotations, root_ids, DOPAMINE_REWARD_SELECTOR)
    root_id_groups["punishment_ppl1"] = select_cell_group(annotations, root_ids, DOPAMINE_PUNISHMENT_SELECTOR)
    root_id_groups["arousal_oa"] = select_cell_group(annotations, root_ids, OCTOPAMINE_AROUSAL_SELECTOR)
    root_id_groups["olfactory_remaining"] = select_remaining_olfactory_receptor_neurons(
        annotations, root_ids, list(SENSORY_CHANNEL_SELECTORS.values())
    )

    mbon_ids = select_cell_group(annotations, root_ids, MBON_SELECTOR)
    avoidance_mbon_ids, approach_mbon_ids = split_mbons_by_dopamine_input(
        edge_pre_ids,
        edge_post_ids,
        edge_syn_counts,
        mbon_ids,
        root_id_groups["reward_pam"],
        root_id_groups["punishment_ppl1"],
    )
    root_id_groups["mbon_avoidance"] = avoidance_mbon_ids
    root_id_groups["mbon_approach"] = approach_mbon_ids
    root_id_groups["kenyon_cells"] = select_cell_group(annotations, root_ids, KENYON_CELL_SELECTOR)

    return {name: numpy.array([neuron_id_to_index[int(i)] for i in ids]) for name, ids in root_id_groups.items()}


def main() -> None:
    """Build and cache the signed adjacency matrix and the neuron pools."""
    CACHE_DIR.mkdir(exist_ok=True)

    root_ids = load_root_ids(DATA_DIR / "proofread_root_ids_783.npy")
    neuron_id_to_index = build_neuron_index(root_ids)
    print("neuron count:", len(root_ids))

    annotations = load_neuron_annotations(ANNOTATIONS_PATH)
    print("annotated neurons:", annotations.num_rows)

    connections_table = load_feather_table(DATA_DIR / "proofread_connections_783.feather")
    aggregated_table = aggregate_connections_by_neuron_pair(connections_table)
    print("aggregated edges:", aggregated_table.num_rows)

    # The real connections file includes synapses onto postsynaptic segments that are
    # NOT themselves in the strictly proofread root id set (e.g. still-unproofread
    # partners of an otherwise proofread neuron) - drop those before building anything
    # that indexes neurons by position in root_ids.
    edge_pre_ids, edge_post_ids, edge_syn_counts = filter_edges_to_known_neurons(
        aggregated_table["pre_pt_root_id"].to_numpy(),
        aggregated_table["post_pt_root_id"].to_numpy(),
        aggregated_table["syn_count"].to_numpy(),
        root_ids,
    )
    print("edges with both endpoints proofread:", len(edge_pre_ids), "of", aggregated_table.num_rows)

    inhibitory_neuron_ids = build_inhibitory_neuron_ids(annotations)
    print("inhibitory neurons (Dale, per-neuron):", len(inhibitory_neuron_ids))

    adjacency_matrix = build_signed_adjacency(edge_pre_ids, edge_post_ids, edge_syn_counts, root_ids, inhibitory_neuron_ids)
    scipy.sparse.save_npz(CACHE_DIR / "adjacency.npz", adjacency_matrix)
    print("saved adjacency:", adjacency_matrix.shape, "nnz:", adjacency_matrix.nnz)

    # synapses.py's event-driven propagation gathers by PRESYNAPTIC row (which neurons
    # just fired) - the transpose of adjacency_matrix's post-major convention
    # (entry[post, pre]). Raw signed syn_counts, not yet scaled by
    # lif.SHIU_2024_PARAMETERS.per_synapse_weight - see
    # synapses.build_signed_edge_weights, applied at load time so retuning that
    # constant never requires rebuilding this cache.
    synapses_matrix = adjacency_matrix.T.tocsr()
    scipy.sparse.save_npz(CACHE_DIR / "synapses.npz", synapses_matrix)
    print("saved synapses (pre-major):", synapses_matrix.shape, "nnz:", synapses_matrix.nnz)

    cell_groups = build_cell_groups(annotations, edge_pre_ids, edge_post_ids, edge_syn_counts, root_ids, neuron_id_to_index)
    numpy.savez(CACHE_DIR / "cell_groups.npz", **cell_groups)
    for group_name, indices in sorted(cell_groups.items()):
        print(f"cell group {group_name}: {len(indices)} neurons")

    pre_neuropil_table = load_feather_table(DATA_DIR / "per_neuron_neuropil_count_pre_783.feather")
    readout_matrix = neuropil_readout_matrix(
        pre_neuropil_table["pre_pt_root_id"].to_numpy(),
        pre_neuropil_table["neuropil"].to_numpy(),
        pre_neuropil_table["count"].to_numpy(),
        ALL_NEUROPIL_NAMES,
        root_ids,
    )
    scipy.sparse.save_npz(CACHE_DIR / "neuropil_readout.npz", readout_matrix)
    print("saved neuropil readout:", readout_matrix.shape, "nnz:", readout_matrix.nnz)

    scene_positions = neuron_scene_positions(
        root_ids,
        numpy.asarray(annotations["root_id"]),
        numpy.asarray(annotations["pos_x"]),
        numpy.asarray(annotations["pos_y"]),
        numpy.asarray(annotations["pos_z"]),
    )
    FRONTEND_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    scene_positions.astype(numpy.float32).tofile(FRONTEND_MODELS_DIR / "neuron-positions.bin")
    print("saved neuron positions:", scene_positions.shape, "->", FRONTEND_MODELS_DIR / "neuron-positions.bin")


if __name__ == "__main__":
    main()
