"""Discrete-time Leaky-Integrate-and-Fire (LIF) spiking network simulation engine."""

import math
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


@dataclass(frozen=True)
class SynapticLIFParameters:
    """Parameters for the LIF-with-decaying-synaptic-conductance model, EXACTLY
    integrated (no discretization error - see _propagator_coefficients) rather than
    Euler-stepped:

        dv/dt = (reset_potential - v + g) / membrane_time_constant_ms   (unless refractory)
        dg/dt = -g / synaptic_time_constant_ms
        spikes when v >= spike_threshold; reset: v = reset_potential, g = 0

    Validated against the real FlyWire connectome by Shiu et al. 2024 (Nature 634:210) -
    values taken from the paper's own companion code
    (github.com/philshiu/Drosophila_brain_model/blob/master/model.py), normalized here
    so spike_threshold=1.0/reset_potential=0.0 (the paper's real membrane-potential
    units: reset/rest -52 mV, threshold -45 mV, a 7 mV gap) rather than millivolts, to
    match this codebase's existing LIFParameters convention. per_synapse_weight scales
    by that same 7 mV gap (their 0.275 mV per synapse / 7 mV), so the network's actual
    spiking dynamics are identical, just in unitless coordinates.
    """

    membrane_time_constant_ms: float
    synaptic_time_constant_ms: float
    spike_threshold: float
    reset_potential: float
    refractory_period_ms: float
    synaptic_delay_ms: float
    dt_ms: float
    per_synapse_weight: float


SHIU_2024_PARAMETERS = SynapticLIFParameters(
    membrane_time_constant_ms=20.0,
    synaptic_time_constant_ms=5.0,
    spike_threshold=1.0,
    reset_potential=0.0,
    refractory_period_ms=2.2,
    synaptic_delay_ms=1.8,
    dt_ms=1.0,
    per_synapse_weight=0.275 / 7.0,
)


@dataclass(frozen=True)
class SynapticLIFState:
    """Per-neuron simulation state for the synaptic-conductance model.

    pending_conductance_increments is a fixed-length tuple (one tensor per delay step,
    index 0 = arrives THIS step) acting as a ring buffer of not-yet-arrived synaptic
    input - see synapses.advance_delay_buffer, which rebuilds this tuple by slicing
    (dropping the consumed head, appending a new tail) rather than shifting an array's
    contents in place, so no tensor data is ever copied, only tensor references.
    """

    membrane_potential: torch.Tensor
    synaptic_conductance: torch.Tensor
    refractory_time_remaining: torch.Tensor
    pending_conductance_increments: tuple[torch.Tensor, ...]


def delay_steps_for(parameters: SynapticLIFParameters) -> int:
    """Return how many dt_ms-sized steps synaptic_delay_ms spans, at least 1."""
    return max(1, round(parameters.synaptic_delay_ms / parameters.dt_ms))


def create_initial_synaptic_state(neuron_count: int, parameters: SynapticLIFParameters, device: str = "cpu") -> SynapticLIFState:
    """Create an all-zero SynapticLIFState, with an empty (all-zero) delay ring buffer
    sized for parameters' own synaptic_delay_ms/dt_ms."""
    return SynapticLIFState(
        membrane_potential=torch.zeros(neuron_count, device=device),
        synaptic_conductance=torch.zeros(neuron_count, device=device),
        refractory_time_remaining=torch.zeros(neuron_count, device=device),
        pending_conductance_increments=tuple(
            torch.zeros(neuron_count, device=device) for _ in range(delay_steps_for(parameters))
        ),
    )


def _propagator_coefficients(parameters: SynapticLIFParameters) -> tuple[float, float, float]:
    """Return the exact linear propagator (a, b, k) for one dt_ms step of the coupled
    (v, g) system (Rotter & Diesmann 1999's exact synaptic integration for LIF with an
    exponentially decaying synaptic current on a separate time constant from the
    membrane):

        v(t+dt) = reset_potential + (v(t) - reset_potential) * a + g(t) * k
        g(t+dt) = g(t) * b

    This is exact for any dt_ms (not a discretization approximation like Euler), at the
    same per-step cost (a few precomputed scalar multiply-adds). Raises ValueError if
    membrane_time_constant_ms equals synaptic_time_constant_ms (the closed form has a
    removable singularity there that SHIU_2024_PARAMETERS' fixed 20ms/5ms never hits).
    """
    tau_m = parameters.membrane_time_constant_ms
    tau_s = parameters.synaptic_time_constant_ms
    if tau_m == tau_s:
        raise ValueError("membrane_time_constant_ms must differ from synaptic_time_constant_ms")
    dt = parameters.dt_ms
    a = math.exp(-dt / tau_m)
    b = math.exp(-dt / tau_s)
    k = (tau_s / (tau_m - tau_s)) * (a - b)
    return a, b, k


def synaptic_step(
    state: SynapticLIFState,
    parameters: SynapticLIFParameters,
    forced_spike_mask: torch.Tensor | None = None,
) -> tuple[SynapticLIFState, torch.Tensor]:
    """Advance the synaptic-conductance LIF model by one dt_ms step.

    Consumes state.pending_conductance_increments[0] (this step's already-delayed,
    already synapse-weighted input - see synapses.propagate_spikes) as an instantaneous
    jump added to synaptic_conductance BEFORE this step's continuous integration (Shiu
    et al.'s on_pre: g += w), then integrates exactly (_propagator_coefficients).
    Returned state's pending_conductance_increments is UNCHANGED (still holds the just-
    consumed head) - advancing the ring buffer itself is synapses.advance_delay_buffer's
    job, called with this step's returned spikes once they've been propagated forward.

    forced_spike_mask: neurons an external Poisson sensory-input process is forcing to
    fire this step (real transduction current from real sensory receptors - see
    semantic_encoder.py - rather than network synaptic input). A forced neuron only
    actually spikes if it is not currently refractory: a refractory neuron cannot be
    forced to fire again, same as it cannot cross threshold again.

    Returns the new state and this step's binary spike tensor (natural threshold
    crossings union successfully-forced spikes).
    """
    a, b, k = _propagator_coefficients(parameters)

    was_refractory = state.refractory_time_remaining > 0
    g_with_input = state.synaptic_conductance + state.pending_conductance_increments[0]

    integrated_potential = parameters.reset_potential + (state.membrane_potential - parameters.reset_potential) * a + g_with_input * k
    integrated_conductance = g_with_input * b

    held_potential = torch.full_like(integrated_potential, parameters.reset_potential)
    new_potential = torch.where(was_refractory, held_potential, integrated_potential)
    new_conductance = torch.where(was_refractory, torch.zeros_like(integrated_conductance), integrated_conductance)

    decremented_refractory = torch.clamp(state.refractory_time_remaining - parameters.dt_ms, min=0.0)

    threshold_crossed = (~was_refractory) & (new_potential >= parameters.spike_threshold)
    if forced_spike_mask is not None:
        forced_and_eligible = forced_spike_mask.bool() & (~was_refractory)
        spiked_mask = threshold_crossed | forced_and_eligible
    else:
        spiked_mask = threshold_crossed

    final_potential = torch.where(spiked_mask, torch.full_like(new_potential, parameters.reset_potential), new_potential)
    final_conductance = torch.where(spiked_mask, torch.zeros_like(new_conductance), new_conductance)
    final_refractory = torch.where(
        spiked_mask, torch.full_like(decremented_refractory, parameters.refractory_period_ms), decremented_refractory
    )

    new_state = SynapticLIFState(
        membrane_potential=final_potential,
        synaptic_conductance=final_conductance,
        refractory_time_remaining=final_refractory,
        pending_conductance_increments=state.pending_conductance_increments,
    )
    return new_state, spiked_mask.to(final_potential.dtype)
