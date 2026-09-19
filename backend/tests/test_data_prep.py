import numpy
import pyarrow
import pyarrow.feather
import pytest

from eternalfly.data_prep import (
    aggregate_connections_by_neuron_pair,
    aggregate_neuron_activity_by_neuropil,
    aggregate_neurotransmitter_by_neuron,
    build_neuron_index,
    dominant_neurotransmitter_labels,
    filter_ids_to_known_set,
    load_feather_table,
    load_root_ids,
)


def test_load_root_ids_reads_npy_array(tmp_path):
    original_ids = numpy.array([10, 20, 30], dtype=numpy.uint64)
    npy_path = tmp_path / "root_ids.npy"
    numpy.save(npy_path, original_ids)

    loaded_ids = load_root_ids(npy_path)

    assert loaded_ids.tolist() == [10, 20, 30]


def test_load_feather_table_round_trips_synthetic_table(tmp_path):
    original_table = pyarrow.table({
        "pre_pt_root_id": [100, 200],
        "syn_count": [3, 4],
    })
    feather_path = tmp_path / "connections.feather"
    pyarrow.feather.write_feather(original_table, feather_path)

    loaded_table = load_feather_table(feather_path)

    assert loaded_table.to_pydict() == original_table.to_pydict()


def _make_connections_table(rows):
    """Build a synthetic per-(pre, post, neuropil) connections table from a list of row dicts."""
    columns = {
        "pre_pt_root_id": [row["pre_pt_root_id"] for row in rows],
        "post_pt_root_id": [row["post_pt_root_id"] for row in rows],
        "neuropil": [row["neuropil"] for row in rows],
        "syn_count": [row["syn_count"] for row in rows],
        "gaba_avg": [row["gaba_avg"] for row in rows],
        "ach_avg": [row["ach_avg"] for row in rows],
        "glut_avg": [row["glut_avg"] for row in rows],
        "oct_avg": [row["oct_avg"] for row in rows],
        "ser_avg": [row["ser_avg"] for row in rows],
        "da_avg": [row["da_avg"] for row in rows],
    }
    return pyarrow.table(columns)


def test_aggregate_connections_by_neuron_pair_weights_by_syn_count_across_neuropils():
    connections_table = _make_connections_table([
        {
            "pre_pt_root_id": 100, "post_pt_root_id": 200, "neuropil": "ME_L",
            "syn_count": 3, "gaba_avg": 0.1, "ach_avg": 0.8, "glut_avg": 0.05,
            "da_avg": 0.03, "oct_avg": 0.01, "ser_avg": 0.01,
        },
        {
            "pre_pt_root_id": 100, "post_pt_root_id": 200, "neuropil": "MB_CA_L",
            "syn_count": 1, "gaba_avg": 0.9, "ach_avg": 0.05, "glut_avg": 0.02,
            "da_avg": 0.01, "oct_avg": 0.01, "ser_avg": 0.01,
        },
    ])

    aggregated_table = aggregate_connections_by_neuron_pair(connections_table)
    aggregated_row = aggregated_table.to_pylist()[0]

    assert aggregated_row["syn_count"] == 4
    assert aggregated_row["gaba_avg"] == pytest.approx(0.3)
    assert aggregated_row["ach_avg"] == pytest.approx(0.6125)


def test_aggregate_connections_by_neuron_pair_keeps_pair_present_in_single_neuropil_unchanged():
    connections_table = _make_connections_table([
        {
            "pre_pt_root_id": 300, "post_pt_root_id": 400, "neuropil": "EB",
            "syn_count": 5, "gaba_avg": 0.2, "ach_avg": 0.5, "glut_avg": 0.1,
            "da_avg": 0.1, "oct_avg": 0.05, "ser_avg": 0.05,
        },
    ])

    aggregated_table = aggregate_connections_by_neuron_pair(connections_table)
    aggregated_row = aggregated_table.to_pylist()[0]

    assert aggregated_row["syn_count"] == 5
    assert aggregated_row["gaba_avg"] == pytest.approx(0.2)
    assert aggregated_row["ach_avg"] == pytest.approx(0.5)


def test_aggregate_connections_by_neuron_pair_returns_expected_columns_only():
    connections_table = _make_connections_table([
        {
            "pre_pt_root_id": 300, "post_pt_root_id": 400, "neuropil": "EB",
            "syn_count": 5, "gaba_avg": 0.2, "ach_avg": 0.5, "glut_avg": 0.1,
            "da_avg": 0.1, "oct_avg": 0.05, "ser_avg": 0.05,
        },
    ])

    aggregated_table = aggregate_connections_by_neuron_pair(connections_table)

    assert set(aggregated_table.column_names) == {
        "pre_pt_root_id", "post_pt_root_id", "syn_count",
        "gaba_avg", "ach_avg", "glut_avg", "oct_avg", "ser_avg", "da_avg",
    }


def test_aggregate_neurotransmitter_by_neuron_weights_by_syn_count_across_all_connections():
    connections_table = _make_connections_table([
        {
            "pre_pt_root_id": 100, "post_pt_root_id": 200, "neuropil": "ME_L",
            "syn_count": 3, "gaba_avg": 0.1, "ach_avg": 0.8, "glut_avg": 0.05,
            "da_avg": 0.03, "oct_avg": 0.01, "ser_avg": 0.01,
        },
        {
            "pre_pt_root_id": 100, "post_pt_root_id": 300, "neuropil": "MB_CA_L",
            "syn_count": 1, "gaba_avg": 0.9, "ach_avg": 0.05, "glut_avg": 0.02,
            "da_avg": 0.01, "oct_avg": 0.01, "ser_avg": 0.01,
        },
    ])

    aggregated_table = aggregate_neurotransmitter_by_neuron(connections_table, "pre_pt_root_id")
    aggregated_row = aggregated_table.to_pylist()[0]

    # Same weighted-average math as aggregate_connections_by_neuron_pair, but grouped
    # by the single pre-neuron across BOTH of its post-partners (200 and 300).
    assert aggregated_row["pre_pt_root_id"] == 100
    assert aggregated_row["syn_count"] == 4
    assert aggregated_row["gaba_avg"] == pytest.approx(0.3)
    assert aggregated_row["ach_avg"] == pytest.approx(0.6125)


def test_aggregate_neurotransmitter_by_neuron_keeps_neurons_separate():
    connections_table = _make_connections_table([
        {
            "pre_pt_root_id": 100, "post_pt_root_id": 200, "neuropil": "ME_L",
            "syn_count": 2, "gaba_avg": 0.9, "ach_avg": 0.05, "glut_avg": 0.02,
            "da_avg": 0.01, "oct_avg": 0.01, "ser_avg": 0.01,
        },
        {
            "pre_pt_root_id": 500, "post_pt_root_id": 200, "neuropil": "ME_L",
            "syn_count": 2, "gaba_avg": 0.05, "ach_avg": 0.05, "glut_avg": 0.02,
            "da_avg": 0.85, "oct_avg": 0.02, "ser_avg": 0.01,
        },
    ])

    aggregated_table = aggregate_neurotransmitter_by_neuron(connections_table, "pre_pt_root_id")
    labels = dominant_neurotransmitter_labels(aggregated_table)

    rows_by_pre_id = dict(zip(aggregated_table["pre_pt_root_id"].to_pylist(), labels.tolist()))
    assert rows_by_pre_id == {100: "gaba", 500: "da"}


def test_dominant_neurotransmitter_labels_picks_max_column_per_row():
    aggregated_table = pyarrow.table({
        "ach_avg": [0.1, 0.7],
        "gaba_avg": [0.8, 0.1],
        "glut_avg": [0.05, 0.1],
        "da_avg": [0.02, 0.05],
        "oct_avg": [0.02, 0.02],
        "ser_avg": [0.01, 0.03],
    })

    labels = dominant_neurotransmitter_labels(aggregated_table)

    assert labels.tolist() == ["gaba", "ach"]


def test_dominant_neurotransmitter_labels_breaks_ties_towards_first_in_fixed_order():
    aggregated_table = pyarrow.table({
        "ach_avg": [0.5],
        "gaba_avg": [0.5],
        "glut_avg": [0.0],
        "da_avg": [0.0],
        "oct_avg": [0.0],
        "ser_avg": [0.0],
    })

    labels = dominant_neurotransmitter_labels(aggregated_table)

    assert labels.tolist() == ["ach"]


def test_aggregate_neuron_activity_by_neuropil_filters_and_sums_matching_rows():
    neuropil_count_table = pyarrow.table({
        "pre_pt_root_id": [100, 100, 200, 300],
        "neuropil": ["ME_L", "EB", "ME_L", "MB_CA_L"],
        "count": [5, 7, 2, 9],
    })

    neuron_ids, summed_counts = aggregate_neuron_activity_by_neuropil(
        neuropil_count_table, target_neuropils=["ME_L", "EB"], id_column_name="pre_pt_root_id"
    )

    activity_by_id = dict(zip(neuron_ids.tolist(), summed_counts.tolist()))
    assert activity_by_id == {100: 12, 200: 2}


def test_aggregate_neuron_activity_by_neuropil_works_with_post_id_column():
    neuropil_count_table = pyarrow.table({
        "post_pt_root_id": [400, 500],
        "neuropil": ["EB", "MB_CA_L"],
        "count": [3, 6],
    })

    neuron_ids, summed_counts = aggregate_neuron_activity_by_neuropil(
        neuropil_count_table, target_neuropils=["EB"], id_column_name="post_pt_root_id"
    )

    assert neuron_ids.tolist() == [400]
    assert summed_counts.tolist() == [3]


def test_filter_ids_to_known_set_drops_ids_not_in_known_set():
    ids = numpy.array([100, 200, 300, 400], dtype=numpy.uint64)
    counts = numpy.array([5, 6, 7, 8], dtype=numpy.int64)
    known_ids = numpy.array([200, 400, 999], dtype=numpy.uint64)

    filtered_ids, filtered_counts = filter_ids_to_known_set(ids, counts, known_ids)

    assert filtered_ids.tolist() == [200, 400]
    assert filtered_counts.tolist() == [6, 8]


def test_filter_ids_to_known_set_handles_large_ids_across_mixed_int_dtypes():
    # Regression test: for large known_ids arrays, numpy.isin switches to a
    # sort-based algorithm that silently promotes mixed int64/uint64 arrays to
    # float64, losing precision for ids beyond 2**53 and causing false positives.
    # A small known_ids array does NOT trigger this path, so this test deliberately
    # uses a large one (matching the real ~139k-neuron scale) to catch a regression.
    real_flywire_id = 720575940631109888
    a_different_id_that_rounds_to_the_same_float64 = real_flywire_id + 1
    padding_ids = numpy.arange(720575940600000000, 720575940600000000 + 10_000, dtype=numpy.uint64)
    known_ids = numpy.concatenate([padding_ids, numpy.array([real_flywire_id], dtype=numpy.uint64)])

    ids = numpy.array([a_different_id_that_rounds_to_the_same_float64], dtype=numpy.int64)
    counts = numpy.array([42], dtype=numpy.int64)

    filtered_ids, filtered_counts = filter_ids_to_known_set(ids, counts, known_ids)

    assert filtered_ids.tolist() == []
    assert filtered_counts.tolist() == []


def test_filter_ids_to_known_set_keeps_all_when_all_known():
    ids = numpy.array([1, 2], dtype=numpy.uint64)
    counts = numpy.array([10, 20], dtype=numpy.int64)
    known_ids = numpy.array([1, 2], dtype=numpy.uint64)

    filtered_ids, filtered_counts = filter_ids_to_known_set(ids, counts, known_ids)

    assert filtered_ids.tolist() == [1, 2]
    assert filtered_counts.tolist() == [10, 20]


def test_build_neuron_index_maps_ids_to_positions_as_plain_ints():
    root_ids = numpy.array([700, 800, 900], dtype=numpy.uint64)

    neuron_id_to_index = build_neuron_index(root_ids)

    assert neuron_id_to_index == {700: 0, 800: 1, 900: 2}
    assert all(type(neuron_id) is int for neuron_id in neuron_id_to_index)
