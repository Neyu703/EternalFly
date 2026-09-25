"""KC->MBON dopamine-gated plasticity: real Kenyon-cell -> MBON synapses get
depressed by how strongly they were recently active (an eligibility trace) at the same
time their target MBON's compartment receives dopamine, and slowly recover back toward
their original strength otherwise (forgetting).

"Compartment-proxy" dopamine drive (see connectome.split_mbons_by_dopamine_input's own
docstring): rather than modeling every individual DAN->MBON synapse, each MBON already
carries a real majority dopaminergic input (PAM for avoidance-MBONs, PPL1 for
approach-MBONs - that's exactly what split_mbons_by_dopamine_input measures), so that
MBON's own compartment dopamine level for a given step is just the real PAM or PPL1
pool's current injected drive (see reading_session.py). Reward (PAM) then depresses
avoidance-MBONs' incoming KC synapses, weakening the fly's baseline avoidance and
producing learned approach; punishment (PPL1) does the mirror image (Aso et al. 2014
eLife; Cohn et al. 2015 Cell).
"""

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class KCToMBONEdges:
    """Real KC->MBON synapses, located once at session build time (the connectome
    itself never changes at runtime - only these edges' weights do)."""

    edge_positions: torch.Tensor  # index into EventDrivenSynapses.values/col_indices
    edge_kc_local_index: torch.Tensor  # this edge's presynaptic KC's position within kc_indices
    edge_mbon_local_index: torch.Tensor  # this edge's postsynaptic MBON's position within mbon_indices
    initial_weights: torch.Tensor  # w0 per edge (== synapse_values[edge_positions] at build time)


def find_kc_to_mbon_edge_positions(
    crow_indices: torch.Tensor,
    col_indices: torch.Tensor,
    synapse_values: torch.Tensor,
    kc_indices: torch.Tensor,
    mbon_indices: torch.Tensor,
) -> KCToMBONEdges:
    """Scan every Kenyon cell's presynaptic row (crow_indices/col_indices - the same
    pre-major CSR layout synapses.propagate_spikes gathers by) and keep only the edges
    landing on a real MBON. Vectorized the same way propagate_spikes gathers firing
    rows: build every KC row's absolute (position, source KC) pairs in one shot, then
    filter down to MBON targets with a single torch.isin - a python loop over ~5000 KC
    rows would work too but this composes with code already proven correct."""
    if len(kc_indices) == 0 or len(mbon_indices) == 0:
        empty = torch.empty(0, dtype=torch.int64)
        return KCToMBONEdges(empty, empty, empty, torch.empty(0, dtype=synapse_values.dtype))

    row_starts = crow_indices[kc_indices]
    row_ends = crow_indices[kc_indices + 1]
    row_lengths = row_ends - row_starts
    total_entries = int(row_lengths.sum().item())
    if total_entries == 0:
        empty = torch.empty(0, dtype=torch.int64)
        return KCToMBONEdges(empty, empty, empty, torch.empty(0, dtype=synapse_values.dtype))

    row_start_offsets = torch.repeat_interleave(torch.cumsum(row_lengths, 0) - row_lengths, row_lengths)
    offsets_within_row = torch.arange(total_entries, device=col_indices.device) - row_start_offsets
    absolute_positions = torch.repeat_interleave(row_starts, row_lengths) + offsets_within_row
    source_kc_local_index = torch.repeat_interleave(
        torch.arange(len(kc_indices), device=col_indices.device), row_lengths
    )

    targets = col_indices[absolute_positions]
    is_to_mbon = torch.isin(targets, mbon_indices)

    edge_positions = absolute_positions[is_to_mbon]
    edge_kc_local_index = source_kc_local_index[is_to_mbon]

    mbon_local_index_by_neuron = torch.full(
        (int(mbon_indices.max().item()) + 1,), -1, dtype=torch.int64, device=col_indices.device
    )
    mbon_local_index_by_neuron[mbon_indices] = torch.arange(len(mbon_indices), device=col_indices.device)
    edge_mbon_local_index = mbon_local_index_by_neuron[targets[is_to_mbon]]

    return KCToMBONEdges(
        edge_positions=edge_positions,
        edge_kc_local_index=edge_kc_local_index,
        edge_mbon_local_index=edge_mbon_local_index,
        initial_weights=synapse_values[edge_positions].clone(),
    )


def decay_eligibility_trace(eligibility: torch.Tensor, elapsed_ms: float, time_constant_ms: float) -> torch.Tensor:
    """Exponentially decay the per-KC eligibility trace toward 0 over elapsed_ms of
    simulated time. Raises ValueError if elapsed_ms is not positive (same contract as
    emotion_decoder.ExponentialMovingAverage.update)."""
    if elapsed_ms <= 0:
        raise ValueError("elapsed_ms must be positive")
    return eligibility * math.exp(-elapsed_ms / time_constant_ms)


def mark_eligible_kcs(eligibility: torch.Tensor, kc_spike_mask: torch.Tensor) -> torch.Tensor:
    """Snap any just-fired KC's eligibility to 1.0 (a fresh tag), leaving every other
    KC's (already-decayed) trace untouched - kc_spike_mask is a bool tensor, one entry
    per KC in the same order as eligibility."""
    return torch.where(kc_spike_mask, torch.ones_like(eligibility), eligibility)


def compute_updated_edge_weights(
    current_edge_weights: torch.Tensor,
    initial_edge_weights: torch.Tensor,
    edge_eligibility: torch.Tensor,
    edge_dopamine_drive: torch.Tensor,
    elapsed_ms: float,
    learning_rate: float,
    recovery_time_constant_ms: float,
) -> torch.Tensor:
    """One dopamine-gated plasticity update: depress every edge by
    learning_rate * its KC's eligibility * its MBON's dopamine drive (clamped to never
    go negative or above that edge's own real w0), then let the depressed weight relax
    back toward w0 by the fraction elapsed_ms/recovery_time_constant_ms would recover
    (slow forgetting - real synapses don't stay depressed forever). Depression and
    recovery apply in the same call since both are driven by the same elapsed_ms of
    simulated time; recovery is calculated from the POST-depression weight, so a still-
    being-depressed synapse doesn't get an artificial recovery credit for ground it's
    losing in this same step.
    """
    if elapsed_ms <= 0:
        raise ValueError("elapsed_ms must be positive")
    depressed = torch.clamp(current_edge_weights - learning_rate * edge_eligibility * edge_dopamine_drive, min=0.0)
    depressed = torch.minimum(depressed, initial_edge_weights)
    recovery_decay = math.exp(-elapsed_ms / recovery_time_constant_ms)
    recovered = initial_edge_weights - (initial_edge_weights - depressed) * recovery_decay
    return torch.minimum(recovered, initial_edge_weights)
