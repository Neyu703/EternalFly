"""Loading and aggregation helpers for the real FlyWire connectome data files."""

import pathlib

import numpy
import pyarrow
import pyarrow.compute
import pyarrow.feather

NEUROTRANSMITTER_COLUMN_ORDER = ["ach", "gaba", "glut", "da", "oct", "ser"]


def load_root_ids(path: pathlib.Path) -> numpy.ndarray:
    """Load the proofread neuron root id array from a .npy file at path."""
    return numpy.load(path)


def load_feather_table(path: pathlib.Path) -> pyarrow.Table:
    """Load a pyarrow Table from a .feather file at path."""
    return pyarrow.feather.read_table(path)


def aggregate_connections_by_neuron_pair(connections_table: pyarrow.Table) -> pyarrow.Table:
    """Collapse per-(pre, post, neuropil) rows into per-(pre, post) neuron-pair rows.

    Sums syn_count across neuropils for the same pair, and computes each neurotransmitter
    probability column as the syn_count-weighted average across that pair's neuropil rows.
    """
    neurotransmitter_column_names = ["gaba_avg", "ach_avg", "glut_avg", "oct_avg", "ser_avg", "da_avg"]

    weighted_columns_table = connections_table
    for column_name in neurotransmitter_column_names:
        weighted_columns_table = weighted_columns_table.append_column(
            f"{column_name}_weighted",
            pyarrow.compute.multiply(
                connections_table[column_name], connections_table["syn_count"]
            ),
        )

    aggregations = [("syn_count", "sum")] + [
        (f"{column_name}_weighted", "sum") for column_name in neurotransmitter_column_names
    ]
    grouped_table = weighted_columns_table.group_by(
        ["pre_pt_root_id", "post_pt_root_id"]
    ).aggregate(aggregations)

    summed_syn_count = grouped_table["syn_count_sum"]
    result_columns = {
        "pre_pt_root_id": grouped_table["pre_pt_root_id"],
        "post_pt_root_id": grouped_table["post_pt_root_id"],
        "syn_count": summed_syn_count,
    }
    for column_name in neurotransmitter_column_names:
        result_columns[column_name] = pyarrow.compute.divide(
            grouped_table[f"{column_name}_weighted_sum"], summed_syn_count
        )

    return pyarrow.table(result_columns)


def dominant_neurotransmitter_labels(aggregated_table: pyarrow.Table) -> numpy.ndarray:
    """Return, per row, the short label of the neurotransmitter with the highest probability.

    Compares the six `{label}_avg` columns in NEUROTRANSMITTER_COLUMN_ORDER; ties go to the
    first label in that fixed order (numpy argmax default behavior).
    """
    probability_columns = numpy.column_stack([
        aggregated_table[f"{label}_avg"].to_numpy() for label in NEUROTRANSMITTER_COLUMN_ORDER
    ])
    dominant_column_indices = numpy.argmax(probability_columns, axis=1)
    label_lookup = numpy.array(NEUROTRANSMITTER_COLUMN_ORDER)
    return label_lookup[dominant_column_indices]


def aggregate_neuron_activity_by_neuropil(
    neuropil_count_table: pyarrow.Table, target_neuropils: list[str], id_column_name: str
) -> tuple[numpy.ndarray, numpy.ndarray]:
    """Sum per-neuron activity counts restricted to a set of target neuropils.

    Filters neuropil_count_table to rows whose neuropil is in target_neuropils, then groups by
    id_column_name (e.g. "pre_pt_root_id" or "post_pt_root_id") and sums count.
    Returns (neuron_ids, summed_counts) as parallel numpy arrays.
    """
    is_target_neuropil = pyarrow.compute.is_in(
        neuropil_count_table["neuropil"], value_set=pyarrow.array(target_neuropils)
    )
    filtered_table = neuropil_count_table.filter(is_target_neuropil)
    grouped_table = filtered_table.group_by(id_column_name).aggregate([("count", "sum")])

    neuron_ids = grouped_table[id_column_name].to_numpy()
    summed_counts = grouped_table["count_sum"].to_numpy()
    return neuron_ids, summed_counts


def filter_ids_to_known_set(
    ids: numpy.ndarray, counts: numpy.ndarray, known_ids: numpy.ndarray
) -> tuple[numpy.ndarray, numpy.ndarray]:
    """Keep only the (id, count) pairs whose id is present in known_ids.

    Used to drop raw/unproofread segment ids that appear in the neuropil-count files
    but aren't part of the simulated proofread neuron set.

    Both id arrays are cast to a common integer dtype (int64) before comparison:
    numpy.isin silently promotes mixed int64/uint64 arrays to float64 for its
    sort-based algorithm on large inputs, which loses precision for FlyWire ids
    (beyond 2**53) and produces false positives.
    """
    is_known = numpy.isin(ids.astype(numpy.int64), known_ids.astype(numpy.int64))
    return ids[is_known], counts[is_known]


def build_neuron_index(root_ids: numpy.ndarray) -> dict[int, int]:
    """Map each neuron root id to its position within root_ids, as plain Python ints."""
    return {int(neuron_id): position for position, neuron_id in enumerate(root_ids)}
