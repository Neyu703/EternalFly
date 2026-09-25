import math

import pytest
import torch

from eternalfly.plasticity import (
    compute_updated_edge_weights,
    decay_eligibility_trace,
    find_kc_to_mbon_edge_positions,
    mark_eligible_kcs,
)


def test_find_kc_to_mbon_edge_positions_returns_empty_for_no_kc_indices():
    crow = torch.tensor([0, 0])
    col = torch.tensor([], dtype=torch.int64)
    values = torch.tensor([])

    edges = find_kc_to_mbon_edge_positions(crow, col, values, torch.tensor([], dtype=torch.int64), torch.tensor([1]))

    assert edges.edge_positions.numel() == 0
    assert edges.edge_kc_local_index.numel() == 0
    assert edges.edge_mbon_local_index.numel() == 0
    assert edges.initial_weights.numel() == 0


def test_find_kc_to_mbon_edge_positions_returns_empty_for_no_mbon_indices():
    crow = torch.tensor([0, 1, 1])
    col = torch.tensor([1])
    values = torch.tensor([5.0])

    edges = find_kc_to_mbon_edge_positions(crow, col, values, torch.tensor([0]), torch.tensor([], dtype=torch.int64))

    assert edges.edge_positions.numel() == 0


def test_find_kc_to_mbon_edge_positions_returns_empty_when_kc_rows_have_no_outgoing_synapses():
    # KC (neuron 0) has an empty row (crow[0] == crow[1]).
    crow = torch.tensor([0, 0, 1])
    col = torch.tensor([0])
    values = torch.tensor([1.0])

    edges = find_kc_to_mbon_edge_positions(crow, col, values, torch.tensor([0]), torch.tensor([1]))

    assert edges.edge_positions.numel() == 0


def test_find_kc_to_mbon_edge_positions_excludes_edges_to_non_mbon_neurons():
    # Neuron 0 (the only KC) -> neuron 1 (not an MBON), weight 7.
    crow = torch.tensor([0, 1, 1, 1])
    col = torch.tensor([1])
    values = torch.tensor([7.0])

    edges = find_kc_to_mbon_edge_positions(crow, col, values, torch.tensor([0]), torch.tensor([2]))

    assert edges.edge_positions.numel() == 0


def test_find_kc_to_mbon_edge_positions_finds_a_single_real_edge():
    # Neuron 0 = KC, neuron 1 = MBON. KC -> MBON, weight 3.
    crow = torch.tensor([0, 1, 1])
    col = torch.tensor([1])
    values = torch.tensor([3.0])

    edges = find_kc_to_mbon_edge_positions(crow, col, values, torch.tensor([0]), torch.tensor([1]))

    assert edges.edge_positions.tolist() == [0]
    assert edges.edge_kc_local_index.tolist() == [0]
    assert edges.edge_mbon_local_index.tolist() == [0]
    assert edges.initial_weights.tolist() == [3.0]


def test_find_kc_to_mbon_edge_positions_maps_multiple_kcs_and_mbons_correctly():
    # Neuron 0 = KC "A" -> neuron 3 (MBON local index 1), weight 1.0
    # Neuron 1 = KC "B" -> neuron 2 (MBON local index 0), weight 2.0; also -> neuron 4 (not MBON)
    # Neuron 2..4 targets; MBON set = {2, 3}. kc_indices = [0, 1] (A local 0, B local 1).
    # col_indices layout: row0 -> [3], row1 -> [2, 4]
    crow = torch.tensor([0, 1, 3, 3, 3, 3])
    col = torch.tensor([3, 2, 4])
    values = torch.tensor([1.0, 2.0, 5.0])
    kc_indices = torch.tensor([0, 1])
    mbon_indices = torch.tensor([2, 3])

    edges = find_kc_to_mbon_edge_positions(crow, col, values, kc_indices, mbon_indices)

    # Order follows the KC-row scan: KC0's edge to neuron 3 (mbon-local 1) first, then
    # KC1's edge to neuron 2 (mbon-local 0); KC1's edge to neuron 4 is excluded.
    assert edges.edge_positions.tolist() == [0, 1]
    assert edges.edge_kc_local_index.tolist() == [0, 1]
    assert edges.edge_mbon_local_index.tolist() == [1, 0]
    assert edges.initial_weights.tolist() == [1.0, 2.0]


def test_decay_eligibility_trace_raises_on_non_positive_elapsed_ms():
    with pytest.raises(ValueError, match="elapsed_ms"):
        decay_eligibility_trace(torch.tensor([1.0]), elapsed_ms=0.0, time_constant_ms=1000.0)


def test_decay_eligibility_trace_decays_by_expected_factor_after_one_time_constant():
    eligibility = torch.tensor([1.0, 0.5])

    decayed = decay_eligibility_trace(eligibility, elapsed_ms=1000.0, time_constant_ms=1000.0)

    assert decayed.tolist() == pytest.approx([1.0 / math.e, 0.5 / math.e], rel=1e-5)


def test_mark_eligible_kcs_snaps_firing_kcs_to_one_and_leaves_others_untouched():
    eligibility = torch.tensor([0.2, 0.6, 0.0])
    kc_spike_mask = torch.tensor([True, False, True])

    updated = mark_eligible_kcs(eligibility, kc_spike_mask)

    assert updated.tolist() == pytest.approx([1.0, 0.6, 1.0])


def test_compute_updated_edge_weights_raises_on_non_positive_elapsed_ms():
    with pytest.raises(ValueError, match="elapsed_ms"):
        compute_updated_edge_weights(
            current_edge_weights=torch.tensor([1.0]),
            initial_edge_weights=torch.tensor([1.0]),
            edge_eligibility=torch.tensor([1.0]),
            edge_dopamine_drive=torch.tensor([1.0]),
            elapsed_ms=0.0,
            learning_rate=0.1,
            recovery_time_constant_ms=1000.0,
        )


def test_compute_updated_edge_weights_depresses_when_eligible_and_dopamine_present():
    updated = compute_updated_edge_weights(
        current_edge_weights=torch.tensor([1.0]),
        initial_edge_weights=torch.tensor([1.0]),
        edge_eligibility=torch.tensor([1.0]),
        edge_dopamine_drive=torch.tensor([1.0]),
        elapsed_ms=1.0,
        learning_rate=0.3,
        recovery_time_constant_ms=1_000_000.0,  # effectively no recovery over 1ms
    )

    # depressed = 1.0 - 0.3*1.0*1.0 = 0.7, negligible recovery over 1ms at this time constant
    assert updated.item() == pytest.approx(0.7, abs=1e-4)


def test_compute_updated_edge_weights_clamps_depression_at_zero():
    updated = compute_updated_edge_weights(
        current_edge_weights=torch.tensor([0.1]),
        initial_edge_weights=torch.tensor([1.0]),
        edge_eligibility=torch.tensor([1.0]),
        edge_dopamine_drive=torch.tensor([1.0]),
        elapsed_ms=1.0,
        learning_rate=5.0,  # would drive it deeply negative unclamped
        recovery_time_constant_ms=1_000_000.0,
    )

    assert updated.item() >= 0.0


def test_compute_updated_edge_weights_never_exceeds_initial_weight():
    # No depression (zero eligibility) and a short recovery time constant relative to
    # elapsed_ms - full recovery should still land exactly at w0, never above it.
    updated = compute_updated_edge_weights(
        current_edge_weights=torch.tensor([0.5]),
        initial_edge_weights=torch.tensor([1.0]),
        edge_eligibility=torch.tensor([0.0]),
        edge_dopamine_drive=torch.tensor([0.0]),
        elapsed_ms=1_000_000.0,
        learning_rate=0.3,
        recovery_time_constant_ms=1.0,
    )

    assert updated.item() == pytest.approx(1.0, abs=1e-4)


def test_compute_updated_edge_weights_recovers_partially_toward_initial_weight_without_new_depression():
    updated = compute_updated_edge_weights(
        current_edge_weights=torch.tensor([0.5]),
        initial_edge_weights=torch.tensor([1.0]),
        edge_eligibility=torch.tensor([0.0]),
        edge_dopamine_drive=torch.tensor([0.0]),
        elapsed_ms=1000.0,
        learning_rate=0.3,
        recovery_time_constant_ms=1000.0,
    )

    # recovered = 1.0 - (1.0 - 0.5) * exp(-1) = 1.0 - 0.5/e
    assert updated.item() == pytest.approx(1.0 - 0.5 / math.e, rel=1e-5)
