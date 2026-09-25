"""Event-driven synaptic propagation: only neurons that actually fired this step
contribute to the next conductance change, gathered and scatter-added rather than a
dense/sparse matrix-vector product against a spike vector that is almost entirely zero
every step (the real network fires well under 1% of its ~139k neurons per tick)."""

from dataclasses import dataclass

import numpy
import scipy.sparse
import torch


@dataclass(frozen=True)
class EventDrivenSynapses:
    """Real signed, per-synapse-count-weighted connectivity in PRESYNAPTIC-row CSR
    form: crow_indices[i]:crow_indices[i+1] indexes into col_indices/values for every
    real synaptic partner neuron i projects onto. This is the transpose of this
    codebase's older post-major adjacency convention (connectome.build_signed_adjacency,
    entry[post, pre]) - event-driven propagation needs to gather by PRESYNAPTIC row
    (which neurons just fired), not postsynaptic row."""

    crow_indices: torch.Tensor
    col_indices: torch.Tensor
    values: torch.Tensor
    post_neuron_count: int


def build_event_driven_synapses(pre_major_csr: scipy.sparse.csr_matrix, device: str = "cpu") -> EventDrivenSynapses:
    """Wrap a scipy CSR matrix (row = presynaptic neuron) as the torch tensors
    propagate_spikes needs. pre_major_csr is the TRANSPOSE of
    connectome.build_signed_adjacency's own post-major convention - see
    scripts/build_connectome_cache.py, which saves it already transposed as synapses.npz."""
    return EventDrivenSynapses(
        crow_indices=torch.as_tensor(pre_major_csr.indptr, dtype=torch.int64, device=device),
        col_indices=torch.as_tensor(pre_major_csr.indices, dtype=torch.int64, device=device),
        values=torch.as_tensor(pre_major_csr.data, dtype=torch.float32, device=device),
        post_neuron_count=pre_major_csr.shape[1],
    )


def propagate_spikes(synapses: EventDrivenSynapses, spikes: torch.Tensor) -> torch.Tensor:
    """Return this step's conductance increment for every postsynaptic neuron, from
    only the presynaptic neurons that fired in `spikes`.

    Vectorized CSR row-gather: for the k firing rows, builds every one of their nonzero
    entries' absolute position in col_indices/values in one shot (no python loop over
    firing neurons - torch.repeat_interleave + an arange offset trick), then a single
    index_add_ scatters them into the result. Cost scales with the firing rows' total
    nonzero entries, not with the full synapse count - the actual "event-driven" saving
    over a full sparse matrix-vector product against a near-all-zero spike vector.
    """
    firing_indices = torch.nonzero(spikes, as_tuple=True)[0]
    conductance_increment = torch.zeros(synapses.post_neuron_count, dtype=synapses.values.dtype, device=spikes.device)
    if firing_indices.numel() == 0:
        return conductance_increment

    row_starts = synapses.crow_indices[firing_indices]
    row_ends = synapses.crow_indices[firing_indices + 1]
    row_lengths = row_ends - row_starts
    total_entries = int(row_lengths.sum().item())
    if total_entries == 0:
        return conductance_increment

    row_start_offsets = torch.repeat_interleave(torch.cumsum(row_lengths, 0) - row_lengths, row_lengths)
    offsets_within_row = torch.arange(total_entries, device=spikes.device) - row_start_offsets
    absolute_positions = torch.repeat_interleave(row_starts, row_lengths) + offsets_within_row

    gathered_post_indices = synapses.col_indices[absolute_positions]
    gathered_values = synapses.values[absolute_positions]
    conductance_increment.index_add_(0, gathered_post_indices, gathered_values)
    return conductance_increment


def advance_delay_buffer(pending: tuple[torch.Tensor, ...], new_contribution: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Drop the just-consumed head (this step's arrived contribution, already read by
    lif.synaptic_step) and append new_contribution as the new tail (arrives
    synaptic_delay_ms from now, once it works its way back to index 0). A tuple
    rebuild via slicing, not an in-place array shift, so no tensor data is copied - only
    tensor references move position."""
    return pending[1:] + (new_contribution,)


def build_signed_edge_weights(syn_counts: numpy.ndarray, signs: numpy.ndarray, per_synapse_weight: float) -> numpy.ndarray:
    """Convert raw (unsigned) synapse counts into Shiu-calibrated conductance weights:
    syn_count * sign * per_synapse_weight (see lif.SHIU_2024_PARAMETERS.per_synapse_weight)."""
    return syn_counts.astype(numpy.float64) * signs * per_synapse_weight
