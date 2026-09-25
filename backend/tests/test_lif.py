import pytest
import torch

from eternalfly.lif import (
    SHIU_2024_PARAMETERS,
    LIFParameters,
    LIFState,
    SynapticLIFParameters,
    SynapticLIFState,
    create_initial_state,
    create_initial_synaptic_state,
    delay_steps_for,
    step,
    synaptic_step,
)


def test_create_initial_state_returns_zero_tensors_with_correct_shape_and_device():
    initial_state = create_initial_state(neuron_count=4, device="cpu")

    assert initial_state.membrane_potential.shape == (4,)
    assert initial_state.refractory_time_remaining.shape == (4,)
    assert initial_state.membrane_potential.tolist() == [0.0, 0.0, 0.0, 0.0]
    assert initial_state.refractory_time_remaining.tolist() == [0.0, 0.0, 0.0, 0.0]
    assert initial_state.membrane_potential.device.type == "cpu"
    assert initial_state.refractory_time_remaining.device.type == "cpu"


def test_step_non_refractory_neuron_with_zero_input_decays_toward_zero():
    parameters = LIFParameters(
        membrane_time_constant_ms=20.0,
        spike_threshold=-50.0,
        reset_potential=-70.0,
        refractory_period_ms=5.0,
        dt_ms=1.0,
    )
    state = LIFState(
        membrane_potential=torch.tensor([-60.0]),
        refractory_time_remaining=torch.tensor([0.0]),
    )
    weight_matrix = torch.tensor([[0.0]])
    external_input = torch.tensor([0.0])
    previous_spikes = torch.tensor([0.0])

    new_state, spikes = step(state, weight_matrix, external_input, previous_spikes, parameters)

    assert new_state.membrane_potential.tolist() == pytest.approx([-57.0])
    assert spikes.tolist() == [0.0]


def test_step_neuron_reaching_threshold_spikes_and_resets():
    parameters = LIFParameters(
        membrane_time_constant_ms=20.0,
        spike_threshold=-50.0,
        reset_potential=-70.0,
        refractory_period_ms=5.0,
        dt_ms=1.0,
    )
    state = LIFState(
        membrane_potential=torch.tensor([-51.0]),
        refractory_time_remaining=torch.tensor([0.0]),
    )
    weight_matrix = torch.tensor([[0.0]])
    external_input = torch.tensor([100.0])
    previous_spikes = torch.tensor([0.0])

    new_state, spikes = step(state, weight_matrix, external_input, previous_spikes, parameters)

    assert spikes.tolist() == [1.0]
    assert new_state.membrane_potential.tolist() == pytest.approx([-70.0])
    assert new_state.refractory_time_remaining.tolist() == pytest.approx([5.0])


def test_step_neuron_below_threshold_does_not_spike_or_reset():
    parameters = LIFParameters(
        membrane_time_constant_ms=20.0,
        spike_threshold=-50.0,
        reset_potential=-70.0,
        refractory_period_ms=5.0,
        dt_ms=1.0,
    )
    state = LIFState(
        membrane_potential=torch.tensor([-60.0]),
        refractory_time_remaining=torch.tensor([0.0]),
    )
    weight_matrix = torch.tensor([[0.0]])
    external_input = torch.tensor([0.0])
    previous_spikes = torch.tensor([0.0])

    new_state, spikes = step(state, weight_matrix, external_input, previous_spikes, parameters)

    assert spikes.tolist() == [0.0]
    assert new_state.membrane_potential.tolist() == pytest.approx([-57.0])
    assert new_state.refractory_time_remaining.tolist() == pytest.approx([0.0])


def test_step_refractory_neuron_does_not_integrate_or_spike_and_decrements_refractory_time():
    parameters = LIFParameters(
        membrane_time_constant_ms=20.0,
        spike_threshold=-50.0,
        reset_potential=-70.0,
        refractory_period_ms=5.0,
        dt_ms=1.0,
    )
    state = LIFState(
        membrane_potential=torch.tensor([-70.0]),
        refractory_time_remaining=torch.tensor([3.0]),
    )
    weight_matrix = torch.tensor([[0.0]])
    external_input = torch.tensor([1000.0])
    previous_spikes = torch.tensor([0.0])

    new_state, spikes = step(state, weight_matrix, external_input, previous_spikes, parameters)

    assert spikes.tolist() == [0.0]
    assert new_state.membrane_potential.tolist() == pytest.approx([-70.0])
    assert new_state.refractory_time_remaining.tolist() == pytest.approx([2.0])


def test_step_refractory_time_remaining_clamps_at_zero_not_negative():
    parameters = LIFParameters(
        membrane_time_constant_ms=20.0,
        spike_threshold=-50.0,
        reset_potential=-70.0,
        refractory_period_ms=5.0,
        dt_ms=1.0,
    )
    state = LIFState(
        membrane_potential=torch.tensor([-70.0]),
        refractory_time_remaining=torch.tensor([0.5]),
    )
    weight_matrix = torch.tensor([[0.0]])
    external_input = torch.tensor([0.0])
    previous_spikes = torch.tensor([0.0])

    new_state, spikes = step(state, weight_matrix, external_input, previous_spikes, parameters)

    assert new_state.refractory_time_remaining.tolist() == pytest.approx([0.0])


def test_step_previous_spikes_drive_synaptic_input_via_weight_matrix():
    parameters = LIFParameters(
        membrane_time_constant_ms=20.0,
        spike_threshold=-50.0,
        reset_potential=-70.0,
        refractory_period_ms=5.0,
        dt_ms=1.0,
    )
    state = LIFState(
        membrane_potential=torch.tensor([-70.0, -70.0]),
        refractory_time_remaining=torch.tensor([0.0, 0.0]),
    )
    # weight_matrix[post, pre]: neuron 0 -> neuron 1 with weight 4.0, no other connections.
    weight_matrix = torch.tensor([[0.0, 0.0], [4.0, 0.0]])
    external_input = torch.tensor([0.0, 0.0])
    previous_spikes = torch.tensor([1.0, 0.0])

    new_state, spikes = step(state, weight_matrix, external_input, previous_spikes, parameters)

    # neuron 0: no synaptic/external input; new_potential = -70 + 0.05 * 70 = -66.5
    # neuron 1: synaptic_input = 4.0 * 1.0 = 4.0; new_potential = -70 + 0.05 * (70 + 4.0) = -66.3
    assert new_state.membrane_potential.tolist() == pytest.approx([-66.5, -66.3])
    assert spikes.tolist() == [0.0, 0.0]


def test_step_dense_and_sparse_csr_weight_matrix_produce_same_result():
    parameters = LIFParameters(
        membrane_time_constant_ms=20.0,
        spike_threshold=-50.0,
        reset_potential=-70.0,
        refractory_period_ms=5.0,
        dt_ms=1.0,
    )
    dense_weight_matrix = torch.tensor(
        [[0.0, 0.0, 2.0], [3.0, 0.0, 0.0], [0.0, -1.0, 0.0]]
    )
    sparse_weight_matrix = dense_weight_matrix.to_sparse_csr()
    external_input = torch.tensor([0.0, 0.0, 0.0])
    previous_spikes = torch.tensor([1.0, 1.0, 0.0])

    dense_state = LIFState(
        membrane_potential=torch.tensor([-70.0, -70.0, -70.0]),
        refractory_time_remaining=torch.tensor([0.0, 0.0, 0.0]),
    )
    sparse_state = LIFState(
        membrane_potential=torch.tensor([-70.0, -70.0, -70.0]),
        refractory_time_remaining=torch.tensor([0.0, 0.0, 0.0]),
    )

    dense_new_state, dense_spikes = step(
        dense_state, dense_weight_matrix, external_input, previous_spikes, parameters
    )
    sparse_new_state, sparse_spikes = step(
        sparse_state, sparse_weight_matrix, external_input, previous_spikes, parameters
    )

    assert dense_new_state.membrane_potential.tolist() == pytest.approx(
        sparse_new_state.membrane_potential.tolist()
    )
    assert dense_new_state.refractory_time_remaining.tolist() == pytest.approx(
        sparse_new_state.refractory_time_remaining.tolist()
    )
    assert dense_spikes.tolist() == pytest.approx(sparse_spikes.tolist())


def test_delay_steps_for_rounds_to_nearest_and_floors_at_one():
    parameters = SynapticLIFParameters(
        membrane_time_constant_ms=20.0, synaptic_time_constant_ms=5.0, spike_threshold=1.0,
        reset_potential=0.0, refractory_period_ms=2.2, synaptic_delay_ms=1.8, dt_ms=1.0, per_synapse_weight=0.04,
    )
    assert delay_steps_for(parameters) == 2  # round(1.8/1.0)

    tiny_delay_parameters = SynapticLIFParameters(
        membrane_time_constant_ms=20.0, synaptic_time_constant_ms=5.0, spike_threshold=1.0,
        reset_potential=0.0, refractory_period_ms=2.2, synaptic_delay_ms=0.1, dt_ms=1.0, per_synapse_weight=0.04,
    )
    assert delay_steps_for(tiny_delay_parameters) == 1  # round(0.1/1.0) = 0, floored to 1


def test_create_initial_synaptic_state_returns_zero_tensors_with_delay_buffer_sized_by_delay_steps():
    state = create_initial_synaptic_state(neuron_count=4, parameters=SHIU_2024_PARAMETERS, device="cpu")

    assert state.membrane_potential.tolist() == [0.0, 0.0, 0.0, 0.0]
    assert state.synaptic_conductance.tolist() == [0.0, 0.0, 0.0, 0.0]
    assert state.refractory_time_remaining.tolist() == [0.0, 0.0, 0.0, 0.0]
    assert len(state.pending_conductance_increments) == delay_steps_for(SHIU_2024_PARAMETERS)
    for increment in state.pending_conductance_increments:
        assert increment.tolist() == [0.0, 0.0, 0.0, 0.0]


def test_synaptic_step_exact_integration_matches_fine_grained_euler_reference():
    # Cross-validates the closed-form propagator against a 10000-substep Euler
    # integration of the SAME ODE (dv/dt=(v0-v+g)/tau_m, dg/dt=-g/tau_s) over one dt -
    # this catches any algebra error in the exact solution, not just internal
    # self-consistency. Inputs stay well under spike_threshold so reset logic never
    # fires and doesn't interfere with the comparison.
    parameters = SynapticLIFParameters(
        membrane_time_constant_ms=20.0, synaptic_time_constant_ms=5.0, spike_threshold=100.0,
        reset_potential=-1.0, refractory_period_ms=2.2, synaptic_delay_ms=1.8, dt_ms=1.0, per_synapse_weight=0.04,
    )
    v_start, g_start, incoming = 0.3, 0.0, 0.6
    state = SynapticLIFState(
        membrane_potential=torch.tensor([v_start]),
        synaptic_conductance=torch.tensor([g_start]),
        refractory_time_remaining=torch.tensor([0.0]),
        pending_conductance_increments=(torch.tensor([incoming]),),
    )

    new_state, spikes = synaptic_step(state, parameters)

    substep_count = 10000
    sub_dt = parameters.dt_ms / substep_count
    v_ref, g_ref = v_start, g_start + incoming
    for _ in range(substep_count):
        v_next = v_ref + sub_dt * ((parameters.reset_potential - v_ref + g_ref) / parameters.membrane_time_constant_ms)
        g_next = g_ref + sub_dt * (-g_ref / parameters.synaptic_time_constant_ms)
        v_ref, g_ref = v_next, g_next

    assert new_state.membrane_potential.item() == pytest.approx(v_ref, abs=1e-5)
    assert new_state.synaptic_conductance.item() == pytest.approx(g_ref, abs=1e-5)
    assert spikes.tolist() == [0.0]


def test_synaptic_step_raises_on_equal_membrane_and_synaptic_time_constants():
    parameters = SynapticLIFParameters(
        membrane_time_constant_ms=5.0, synaptic_time_constant_ms=5.0, spike_threshold=1.0,
        reset_potential=0.0, refractory_period_ms=2.2, synaptic_delay_ms=1.8, dt_ms=1.0, per_synapse_weight=0.04,
    )
    state = create_initial_synaptic_state(1, SHIU_2024_PARAMETERS, device="cpu")
    state = SynapticLIFState(
        state.membrane_potential, state.synaptic_conductance, state.refractory_time_remaining,
        (torch.tensor([0.0]),),
    )

    with pytest.raises(ValueError, match="must differ"):
        synaptic_step(state, parameters)


def test_synaptic_step_neuron_crossing_threshold_spikes_and_resets():
    parameters = SHIU_2024_PARAMETERS
    state = SynapticLIFState(
        membrane_potential=torch.tensor([0.99]),
        synaptic_conductance=torch.tensor([0.0]),
        refractory_time_remaining=torch.tensor([0.0]),
        pending_conductance_increments=(torch.tensor([10.0]),),  # large jump, guarantees crossing 1.0
    )

    new_state, spikes = synaptic_step(state, parameters)

    assert spikes.tolist() == [1.0]
    assert new_state.membrane_potential.item() == pytest.approx(parameters.reset_potential)
    assert new_state.synaptic_conductance.item() == pytest.approx(0.0)
    assert new_state.refractory_time_remaining.item() == pytest.approx(parameters.refractory_period_ms)


def test_synaptic_step_refractory_neuron_is_held_and_never_spikes_even_with_large_input():
    parameters = SHIU_2024_PARAMETERS
    state = SynapticLIFState(
        membrane_potential=torch.tensor([0.99]),
        synaptic_conductance=torch.tensor([0.5]),
        refractory_time_remaining=torch.tensor([1.5]),
        pending_conductance_increments=(torch.tensor([10.0]),),
    )

    new_state, spikes = synaptic_step(state, parameters)

    assert spikes.tolist() == [0.0]
    assert new_state.membrane_potential.item() == pytest.approx(parameters.reset_potential)
    assert new_state.synaptic_conductance.item() == pytest.approx(0.0)
    assert new_state.refractory_time_remaining.item() == pytest.approx(0.5)  # 1.5 - dt_ms(1.0)


def test_synaptic_step_forced_spike_fires_an_eligible_neuron_below_threshold():
    parameters = SHIU_2024_PARAMETERS
    state = SynapticLIFState(
        membrane_potential=torch.tensor([0.1, 0.1]),
        synaptic_conductance=torch.tensor([0.0, 0.0]),
        refractory_time_remaining=torch.tensor([0.0, 0.0]),
        pending_conductance_increments=(torch.tensor([0.0, 0.0]),),
    )
    forced_spike_mask = torch.tensor([True, False])

    new_state, spikes = synaptic_step(state, parameters, forced_spike_mask=forced_spike_mask)

    assert spikes.tolist() == [1.0, 0.0]
    assert new_state.membrane_potential[0].item() == pytest.approx(parameters.reset_potential)
    assert new_state.refractory_time_remaining[0].item() == pytest.approx(parameters.refractory_period_ms)


def test_synaptic_step_forced_spike_is_suppressed_for_a_refractory_neuron():
    parameters = SHIU_2024_PARAMETERS
    state = SynapticLIFState(
        membrane_potential=torch.tensor([0.1]),
        synaptic_conductance=torch.tensor([0.0]),
        refractory_time_remaining=torch.tensor([1.0]),
        pending_conductance_increments=(torch.tensor([0.0]),),
    )
    forced_spike_mask = torch.tensor([True])

    new_state, spikes = synaptic_step(state, parameters, forced_spike_mask=forced_spike_mask)

    assert spikes.tolist() == [0.0]


def test_synaptic_step_returned_state_keeps_pending_conductance_increments_unchanged():
    parameters = SHIU_2024_PARAMETERS
    original_pending = (torch.tensor([0.2]), torch.tensor([0.3]))
    state = SynapticLIFState(
        membrane_potential=torch.tensor([0.0]),
        synaptic_conductance=torch.tensor([0.0]),
        refractory_time_remaining=torch.tensor([0.0]),
        pending_conductance_increments=original_pending,
    )

    new_state, _ = synaptic_step(state, parameters)

    assert new_state.pending_conductance_increments is original_pending
