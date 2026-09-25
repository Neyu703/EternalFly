"""Batches an entire frame's worth of steps' spike counts on-device before reading
anything back to the CPU. The old per-tick loop called .item() once per tracked pool
(up to 81 GPU<->CPU syncs per tick, its single biggest cost) - accumulating counts
across a whole frame and reading back once removes that entirely."""

import numpy
import torch


def accumulate_spike_counts(running_counts: torch.Tensor, spikes: torch.Tensor) -> torch.Tensor:
    """Add spikes (this step's binary spike tensor) into running_counts (per-neuron
    spike count accumulated across a frame's worth of steps so far) - stays on
    whichever device the tensors are already on, no .item()/.cpu() call here."""
    return running_counts + spikes


def readout_rates(readout_matrix: torch.Tensor, spike_counts: torch.Tensor, step_count: int) -> torch.Tensor:
    """Convert accumulated spike_counts (summed over step_count steps) into a firing
    RATE (spikes per step, 0..1) per readout row, via one matrix-vector product against
    readout_matrix - e.g. connectome.neuropil_readout_matrix's real presynapse-weighted
    neuropil rows, or build_group_readout_matrix's named-cell-group rows. Raises
    ValueError if step_count is not positive."""
    if step_count <= 0:
        raise ValueError(f"step_count must be positive, got {step_count}")
    return (readout_matrix @ spike_counts) / step_count


def build_group_readout_matrix(
    group_indices: dict[str, numpy.ndarray], neuron_count: int, device: str = "cpu"
) -> tuple[list[str], torch.Tensor]:
    """Build a (len(group_indices) x neuron_count) matrix where row r is the uniform
    mean-firing-rate readout for group_indices' r-th named group (1/group_size at each
    member index, 0 elsewhere) - readout_rates(this_matrix, spike_counts, step_count)
    then gives every named group's mean firing rate in the SAME single batched
    matrix-vector product as neuropil_readout_matrix's rows, letting a frame's entire
    readout (every sensory channel, behavior, reward/punishment/arousal pool, and
    neuropil) come from one matmul and one .cpu() transfer.

    A group with zero members gets an all-zero row (rate always reads 0, never NaN).
    Returns (row_names_in_order, matrix).
    """
    row_names = list(group_indices)
    matrix = torch.zeros(len(row_names), neuron_count, device=device)
    for row, name in enumerate(row_names):
        indices = group_indices[name]
        if len(indices) > 0:
            matrix[row, torch.as_tensor(indices, dtype=torch.int64, device=device)] = 1.0 / len(indices)
    return row_names, matrix
