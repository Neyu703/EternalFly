"""One-off script: run the real Milestone 0 pipeline against the downloaded connectome
files and report sizes/timings. Not part of the tested package - throwaway verification."""

import time

import numpy

from eternalfly.data_prep import (
    aggregate_connections_by_neuron_pair,
    build_neuron_index,
    dominant_neurotransmitter_labels,
    load_feather_table,
    load_root_ids,
)
from scripts.build_connectome_cache import build_all_pool_indices
from scripts.data_paths import CONNECTIONS_PATH, POST_NEUROPIL_COUNTS_PATH, PRE_NEUROPIL_COUNTS_PATH, ROOT_IDS_PATH


def timed(label: str, function, *args):
    """Run function(*args), print label + elapsed seconds, return its result."""
    start_time = time.perf_counter()
    result = function(*args)
    elapsed_seconds = time.perf_counter() - start_time
    print(f"{label}: {elapsed_seconds:.1f}s")
    return result


def count_unknown_ids(neuron_ids: numpy.ndarray, root_ids: numpy.ndarray) -> int:
    """How many of neuron_ids are not proofread neurons. Both are cast to int64 first:
    numpy.isin on mixed int64/uint64 ids compares them as float64, which loses precision
    for FlyWire ids (see data_prep.filter_ids_to_known_set)."""
    return int((~numpy.isin(neuron_ids.astype(numpy.int64), root_ids.astype(numpy.int64))).sum())


def main() -> None:
    """Load the connectome, aggregate it and build the session's neuron pools, reporting
    sizes and how long each stage takes."""
    root_ids = timed("load_root_ids", load_root_ids, ROOT_IDS_PATH)
    print("neuron count:", len(root_ids))

    connections_table = timed("load connections feather", load_feather_table, CONNECTIONS_PATH)
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

    unknown_pre = count_unknown_ids(aggregated_table["pre_pt_root_id"].to_numpy(), root_ids)
    unknown_post = count_unknown_ids(aggregated_table["post_pt_root_id"].to_numpy(), root_ids)
    print("edges with unknown pre id:", unknown_pre, " unknown post id:", unknown_post)

    pre_neuropil_table = timed("load pre-neuropil feather", load_feather_table, PRE_NEUROPIL_COUNTS_PATH)
    post_neuropil_table = timed("load post-neuropil feather", load_feather_table, POST_NEUROPIL_COUNTS_PATH)
    pool_indices = timed(
        "build_all_pool_indices",
        build_all_pool_indices,
        connections_table,
        pre_neuropil_table,
        post_neuropil_table,
        root_ids,
        neuron_index,
    )
    for pool_name, indices in pool_indices.items():
        print(f"pool {pool_name}: {len(indices)} neurons (of {len(root_ids)} total)")


if __name__ == "__main__":
    main()
