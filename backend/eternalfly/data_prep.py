"""Loading and aggregation helpers for the real FlyWire connectome data files."""

import pathlib

import numpy
import pyarrow
import pyarrow.compute
import pyarrow.csv
import pyarrow.feather

NEUROTRANSMITTER_COLUMN_ORDER = ["ach", "gaba", "glut", "da", "oct", "ser"]
NEUROTRANSMITTER_PROBABILITY_COLUMNS = ["gaba_avg", "ach_avg", "glut_avg", "oct_avg", "ser_avg", "da_avg"]

# The columns cell_groups.py's selectors and geometry.neuron_scene_positions actually
# read from the real FlyWire cell-type annotations
# (Supplemental_file1_neuron_annotations.tsv) - the file has 31 columns in total, most
# of it per-neuron metadata (soma position, hemilineage, VirtualFlyBrain ids, ...)
# irrelevant here.
ANNOTATION_COLUMNS = [
    "root_id", "super_class", "cell_class", "cell_sub_class", "cell_type", "side", "top_nt", "pos_x", "pos_y", "pos_z",
]


def load_root_ids(path: pathlib.Path) -> numpy.ndarray:
    """Load the proofread neuron root id array from a .npy file at path."""
    return numpy.load(path)


def load_neuron_annotations(path: pathlib.Path) -> pyarrow.Table:
    """Load ANNOTATION_COLUMNS from the real FlyWire cell-type annotations TSV (tab-
    delimited, see scripts/download_connectome.py). root_id is parsed as int64 - FlyWire
    ids exceed 2**53, so this must never go through float64 (see filter_ids_to_known_set's
    note on the same numpy.isin pitfall). Every other column stays a plain string,
    including where a row has no annotation for it (kept as "", never null, so
    cell_groups.py's selectors can compare it directly against a fixed set of values)."""
    parse_options = pyarrow.csv.ParseOptions(delimiter="\t")
    convert_options = pyarrow.csv.ConvertOptions(
        column_types={"root_id": pyarrow.int64()},
        include_columns=ANNOTATION_COLUMNS,
        null_values=[],  # an empty cell is a real "no annotation" value here, not null
    )
    return pyarrow.csv.read_csv(path, parse_options=parse_options, convert_options=convert_options)


def load_feather_table(path: pathlib.Path) -> pyarrow.Table:
    """Load a pyarrow Table from a .feather file at path."""
    return pyarrow.feather.read_table(path)


def _aggregate_weighted_neurotransmitters(connections_table: pyarrow.Table, group_by_columns: list[str]) -> pyarrow.Table:
    """Collapse connections_table rows into one row per unique combination of
    group_by_columns, summing syn_count and computing each neurotransmitter
    probability column as the syn_count-weighted average across the collapsed rows."""
    weighted_columns_table = connections_table
    for column_name in NEUROTRANSMITTER_PROBABILITY_COLUMNS:
        weighted_columns_table = weighted_columns_table.append_column(
            f"{column_name}_weighted",
            pyarrow.compute.multiply(
                connections_table[column_name], connections_table["syn_count"]
            ),
        )

    aggregations = [("syn_count", "sum")] + [
        (f"{column_name}_weighted", "sum") for column_name in NEUROTRANSMITTER_PROBABILITY_COLUMNS
    ]
    grouped_table = weighted_columns_table.group_by(group_by_columns).aggregate(aggregations)

    summed_syn_count = grouped_table["syn_count_sum"]
    result_columns = {column_name: grouped_table[column_name] for column_name in group_by_columns}
    result_columns["syn_count"] = summed_syn_count
    for column_name in NEUROTRANSMITTER_PROBABILITY_COLUMNS:
        result_columns[column_name] = pyarrow.compute.divide(
            grouped_table[f"{column_name}_weighted_sum"], summed_syn_count
        )

    return pyarrow.table(result_columns)


def aggregate_connections_by_neuron_pair(connections_table: pyarrow.Table) -> pyarrow.Table:
    """Collapse per-(pre, post, neuropil) rows into per-(pre, post) neuron-pair rows.

    Sums syn_count across neuropils for the same pair, and computes each neurotransmitter
    probability column as the syn_count-weighted average across that pair's neuropil rows.
    """
    return _aggregate_weighted_neurotransmitters(connections_table, ["pre_pt_root_id", "post_pt_root_id"])


def aggregate_neurotransmitter_by_neuron(connections_table: pyarrow.Table, id_column_name: str) -> pyarrow.Table:
    """Collapse connections_table rows into one row per neuron (identified by
    id_column_name, e.g. "pre_pt_root_id" to aggregate by presynaptic neuron), with
    each neurotransmitter probability column as the syn_count-weighted average across
    all of that neuron's connections (across every post-partner and neuropil).

    Used to classify a neuron's own dominant neurotransmitter (via
    dominant_neurotransmitter_labels on the result), as opposed to
    aggregate_connections_by_neuron_pair's per-connection classification."""
    return _aggregate_weighted_neurotransmitters(connections_table, [id_column_name])


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


def filter_edges_to_known_neurons(
    pre_ids: numpy.ndarray, post_ids: numpy.ndarray, syn_counts: numpy.ndarray, known_ids: numpy.ndarray
) -> tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    """Keep only the (pre_id, post_id, syn_count) edges where BOTH endpoints are in
    known_ids (the proofread/simulated neuron set).

    The real connections file includes synapses onto postsynaptic segments that are
    not themselves part of the strictly proofread root id set (e.g. still-unproofread
    partners of an otherwise proofread neuron) - build_signed_adjacency has no index
    for those, so they must be dropped here rather than raising. Same int64-cast
    reasoning as filter_ids_to_known_set (FlyWire ids exceed 2**53, so numpy.isin must
    never fall back to a float64 comparison).
    """
    pre_ids_int64 = pre_ids.astype(numpy.int64)
    post_ids_int64 = post_ids.astype(numpy.int64)
    known_ids_int64 = known_ids.astype(numpy.int64)
    is_known = numpy.isin(pre_ids_int64, known_ids_int64) & numpy.isin(post_ids_int64, known_ids_int64)
    return pre_ids_int64[is_known], post_ids_int64[is_known], syn_counts[is_known]
