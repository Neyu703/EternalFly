"""Small pure, deterministic helpers composed by the reading-session orchestration layer."""

import torch

from eternalfly.validation import require_positive_int


def compute_pool_spike_rate(spikes: torch.Tensor, pool_indices: torch.Tensor) -> float:
    """Return the fraction of a neuron pool that spiked this tick, as a plain Python float.

    spikes: full-network binary spike tensor, shape (neuron_count,), values 0.0/1.0.
    pool_indices: 1-D integer tensor of neuron indices belonging to the pool.
    Returns 0.0 (not NaN) when pool_indices is empty, since an empty pool carries no signal.
    """
    if pool_indices.shape[0] == 0:
        return 0.0
    return spikes[pool_indices].mean().item()


def inject_currents_at_indices(
    neuron_count: int, pool_indices: torch.Tensor, currents: torch.Tensor, device: str = "cpu"
) -> torch.Tensor:
    """Build a full-length (neuron_count,) current tensor, zero except at pool_indices.

    currents[i] is placed at position pool_indices[i] (parallel arrays). Raises ValueError
    if pool_indices and currents have different lengths. Returns an all-zero tensor when
    pool_indices is empty.
    """
    if pool_indices.shape[0] != currents.shape[0]:
        raise ValueError(
            f"pool_indices and currents must have the same length, "
            f"got {pool_indices.shape[0]} and {currents.shape[0]}"
        )

    full_currents = torch.zeros(neuron_count, device=device)
    if pool_indices.shape[0] == 0:
        return full_currents

    full_currents[pool_indices] = currents.to(device)
    return full_currents


def is_new_word_tick(tick_number: int, ticks_per_word: int) -> bool:
    """Return True when tick_number starts a new word (including tick 0, the first word).

    A new word begins every ticks_per_word ticks. Raises ValueError if ticks_per_word
    is not a positive integer.
    """
    require_positive_int("ticks_per_word", ticks_per_word)
    return tick_number % ticks_per_word == 0


def spike_fraction_to_hz(spike_fraction: float, dt_ms: float) -> float:
    """Convert the fraction of neurons spiking per simulation step of dt_ms milliseconds
    into their mean firing rate in spikes per second (Hz)."""
    return spike_fraction * 1000.0 / dt_ms
