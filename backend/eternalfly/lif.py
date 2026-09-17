"""Discrete-time Leaky-Integrate-and-Fire (LIF) spiking network simulation engine."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class LIFParameters:
    """Parameters governing one Euler-discretized LIF update step.

    membrane_time_constant_ms: decay time constant tau of the membrane potential.
    spike_threshold: potential at/above which a non-refractory neuron spikes.
    reset_potential: potential a neuron is held at while refractory or reset to after spiking.
    refractory_period_ms: duration a neuron stays refractory after spiking.
    dt_ms: simulation timestep size.
    """

    membrane_time_constant_ms: float
    spike_threshold: float
    reset_potential: float
    refractory_period_ms: float
    dt_ms: float


@dataclass(frozen=True)
class LIFState:
    """Per-neuron simulation state carried between timesteps."""

    membrane_potential: torch.Tensor
    refractory_time_remaining: torch.Tensor


def create_initial_state(neuron_count: int, device: str = "cpu") -> LIFState:
    """Create an all-zero LIFState for neuron_count neurons on the given device."""
    return LIFState(
        membrane_potential=torch.zeros(neuron_count, device=device),
        refractory_time_remaining=torch.zeros(neuron_count, device=device),
    )


def _integrate_membrane_potential(
    state: LIFState, total_input: torch.Tensor, parameters: LIFParameters
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Advance membrane potential and refractory countdown by one Euler step, before spike reset.

    Neurons currently refractory (refractory_time_remaining > 0) are held at reset_potential
    and skip integration; their refractory_time_remaining is decremented by dt_ms, clamped to
    a minimum of 0. Non-refractory neurons integrate the standard leaky-integrate Euler update.
    Returns (new_potential, decremented_refractory_time_remaining, was_refractory_mask).
    """
    was_refractory_mask = state.refractory_time_remaining > 0

    integrated_potential = state.membrane_potential + (
        parameters.dt_ms / parameters.membrane_time_constant_ms
    ) * (-state.membrane_potential + total_input)
    held_potential = torch.full_like(state.membrane_potential, parameters.reset_potential)
    new_potential = torch.where(was_refractory_mask, held_potential, integrated_potential)

    decremented_refractory_time_remaining = torch.clamp(
        state.refractory_time_remaining - parameters.dt_ms, min=0.0
    )

    return new_potential, decremented_refractory_time_remaining, was_refractory_mask


def _detect_spikes(
    new_potential: torch.Tensor, was_refractory_mask: torch.Tensor, parameters: LIFParameters
) -> torch.Tensor:
    """Return a binary spike tensor (same dtype as new_potential).

    A neuron spikes if it was not refractory this step and its new potential reached or
    exceeded spike_threshold.
    """
    spiked_mask = (~was_refractory_mask) & (new_potential >= parameters.spike_threshold)
    return spiked_mask.to(new_potential.dtype)


def _apply_spike_reset(
    new_potential: torch.Tensor,
    decremented_refractory_time_remaining: torch.Tensor,
    spikes: torch.Tensor,
    parameters: LIFParameters,
) -> LIFState:
    """Reset spiking neurons to reset_potential and start their refractory period."""
    spiked_mask = spikes.bool()
    final_potential = torch.where(
        spiked_mask, torch.full_like(new_potential, parameters.reset_potential), new_potential
    )
    final_refractory_time_remaining = torch.where(
        spiked_mask,
        torch.full_like(decremented_refractory_time_remaining, parameters.refractory_period_ms),
        decremented_refractory_time_remaining,
    )
    return LIFState(
        membrane_potential=final_potential,
        refractory_time_remaining=final_refractory_time_remaining,
    )


def step(
    state: LIFState,
    weight_matrix: torch.Tensor,
    external_input: torch.Tensor,
    previous_spikes: torch.Tensor,
    parameters: LIFParameters,
) -> tuple[LIFState, torch.Tensor]:
    """Advance the LIF network state by one Euler timestep of size parameters.dt_ms.

    weight_matrix[post, pre] is the signed synaptic weight from presynaptic neuron pre
    to postsynaptic neuron post; it may be dense or a torch.sparse_csr_tensor.
    Returns the new LIFState and the new binary spike tensor (1.0 spiked / 0.0 did not),
    which becomes the next step's previous_spikes.
    """
    synaptic_input = weight_matrix @ previous_spikes
    total_input = synaptic_input + external_input

    new_potential, decremented_refractory_time_remaining, was_refractory_mask = _integrate_membrane_potential(
        state, total_input, parameters
    )
    spikes = _detect_spikes(new_potential, was_refractory_mask, parameters)
    new_state = _apply_spike_reset(new_potential, decremented_refractory_time_remaining, spikes, parameters)

    return new_state, spikes
