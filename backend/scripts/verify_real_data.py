"""One-off script: run the real Milestone 0 pipeline against the downloaded connectome
files and report sizes/timings. Not part of the tested package - throwaway verification."""

import time
from pathlib import Path

import numpy

from eternalfly.data_prep import (
    aggregate_connections_by_neuron_pair,
    aggregate_neuron_activity_by_neuropil,
    build_neuron_index,
    dominant_neurotransmitter_labels,
    filter_ids_to_known_set,
    load_feather_table,
    load_root_ids,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

SENSORY_INPUT_NEUROPILS = ["ME_L", "ME_R", "LO_L", "LO_R"]
APPROACH_POOL_NEUROPILS = ["MB_ML_L", "MB_ML_R"]
AVOIDANCE_POOL_NEUROPILS = ["MB_VL_L", "MB_VL_R"]
AROUSAL_POOL_NEUROPILS = ["EB", "FB"]


def timed(label: str, function, *args):
    """Run function(*args), print label + elapsed seconds, return its result."""
    start_time = time.perf_counter()
    result = function(*args)
    elapsed_seconds = time.perf_counter() - start_time
    print(f"{label}: {elapsed_seconds:.1f}s")
    return result


def main() -> None:
    root_ids = timed("load_root_ids", load_root_ids, DATA_DIR / "proofread_root_ids_783.npy")
    print("neuron count:", len(root_ids))

    connections_table = timed(
        "load connections feather", load_feather_table, DATA_DIR / "proofread_connections_783.feather"
    )
    print("raw connection rows:", connections_table.num_rows)

    aggregated_table = timed(
        "aggregate_connections_by_neuron_pair", aggregate_connections_by_neuron_pair, connections_table
    )
    print("aggregated neuron-pair edges:", aggregated_table.num_rows)

    nt_labels = timed(
        "dominant_neurotransmitter_labels", dominant_neurotransmitter_labels, aggregated_table
    )
    print("nt label counts:", {label: int((nt_labels == label).sum()) for label in set(nt_labels.tolist())})

    neuron_index = timed("build_neuron_index", build_neuron_index, root_ids)

    pre_ids = aggregated_table["pre_pt_root_id"].to_numpy()
    post_ids = aggregated_table["post_pt_root_id"].to_numpy()
    syn_counts = aggregated_table["syn_count"].to_numpy()

    unknown_pre = (~numpy.isin(pre_ids, root_ids)).sum()
    unknown_post = (~numpy.isin(post_ids, root_ids)).sum()
    print("edges with unknown pre id:", unknown_pre, " unknown post id:", unknown_post)

    post_neuropil_table = timed(
        "load post-neuropil feather", load_feather_table, DATA_DIR / "per_neuron_neuropil_count_post_783.feather"
    )
    for pool_name, neuropils in [
        ("sensory_input", SENSORY_INPUT_NEUROPILS),
        ("approach", APPROACH_POOL_NEUROPILS),
        ("avoidance", AVOIDANCE_POOL_NEUROPILS),
        ("arousal", AROUSAL_POOL_NEUROPILS),
    ]:
        pool_ids, pool_counts = aggregate_neuron_activity_by_neuropil(
            post_neuropil_table, neuropils, "post_pt_root_id"
        )
        pool_ids, pool_counts = filter_ids_to_known_set(pool_ids, pool_counts, root_ids)
        print(f"pool candidates for {pool_name}: {len(pool_ids)} neurons (of {len(root_ids)} total)")

    print("skipping build_signed_adjacency full run for now (timing only planned) -- see follow-up")


if __name__ == "__main__":
    main()
