import pytest
import torch

from eternalfly.lif import LIFParameters, LIFState, accepting_sparse_csr_beta, create_initial_state, step

PARAMETERS = LIFParameters(
    membrane_time_constant_ms=20.0,
    spike_threshold=-50.0,
    reset_potential=-70.0,
    refractory_period_ms=5.0,
    dt_ms=1.0,
)


def _step_single_neuron(
    membrane_potential: float, refractory_time_remaining: float, external_input: float
) -> tuple[LIFState, torch.Tensor]:
    """Step a lone, unconnected neuron that didn't spike last tick once with PARAMETERS,
    returning its new state and spikes."""
    state = LIFState(
        membrane_potential=torch.tensor([membrane_potential]),
        refractory_time_remaining=torch.tensor([refractory_time_remaining]),
    )
    return step(state, torch.tensor([[0.0]]), torch.tensor([external_input]), torch.tensor([0.0]), PARAMETERS)


def test_create_initial_state_returns_zero_tensors_with_correct_shape_and_device():
    initial_state = create_initial_state(neuron_count=4, device="cpu")

    assert initial_state.membrane_potential.shape == (4,)
    assert initial_state.refractory_time_remaining.shape == (4,)
    assert initial_state.membrane_potential.tolist() == [0.0, 0.0, 0.0, 0.0]
    assert initial_state.refractory_time_remaining.tolist() == [0.0, 0.0, 0.0, 0.0]
    assert initial_state.membrane_potential.device.type == "cpu"
    assert initial_state.refractory_time_remaining.device.type == "cpu"


def test_step_non_refractory_neuron_with_zero_input_decays_toward_zero():
    new_state, spikes = _step_single_neuron(-60.0, 0.0, 0.0)

    assert new_state.membrane_potential.tolist() == pytest.approx([-57.0])
    assert spikes.tolist() == [0.0]


def test_step_neuron_reaching_threshold_spikes_and_resets():
    new_state, spikes = _step_single_neuron(-51.0, 0.0, 100.0)

    assert spikes.tolist() == [1.0]
    assert new_state.membrane_potential.tolist() == pytest.approx([-70.0])
    assert new_state.refractory_time_remaining.tolist() == pytest.approx([5.0])


def test_step_neuron_below_threshold_does_not_spike_or_reset():
    new_state, spikes = _step_single_neuron(-60.0, 0.0, 0.0)

    assert spikes.tolist() == [0.0]
    assert new_state.membrane_potential.tolist() == pytest.approx([-57.0])
    assert new_state.refractory_time_remaining.tolist() == pytest.approx([0.0])


def test_step_refractory_neuron_does_not_integrate_or_spike_and_decrements_refractory_time():
    new_state, spikes = _step_single_neuron(-70.0, 3.0, 1000.0)

    assert spikes.tolist() == [0.0]
    assert new_state.membrane_potential.tolist() == pytest.approx([-70.0])
    assert new_state.refractory_time_remaining.tolist() == pytest.approx([2.0])


def test_step_refractory_time_remaining_clamps_at_zero_not_negative():
    new_state, spikes = _step_single_neuron(-70.0, 0.5, 0.0)

    assert new_state.refractory_time_remaining.tolist() == pytest.approx([0.0])


def test_step_previous_spikes_drive_synaptic_input_via_weight_matrix():
    state = LIFState(
        membrane_potential=torch.tensor([-70.0, -70.0]),
        refractory_time_remaining=torch.tensor([0.0, 0.0]),
    )
    # weight_matrix[post, pre]: neuron 0 -> neuron 1 with weight 4.0, no other connections.
    weight_matrix = torch.tensor([[0.0, 0.0], [4.0, 0.0]])
    external_input = torch.tensor([0.0, 0.0])
    previous_spikes = torch.tensor([1.0, 0.0])

    new_state, spikes = step(state, weight_matrix, external_input, previous_spikes, PARAMETERS)

    # neuron 0: no synaptic/external input; new_potential = -70 + 0.05 * 70 = -66.5
    # neuron 1: synaptic_input = 4.0 * 1.0 = 4.0; new_potential = -70 + 0.05 * (70 + 4.0) = -66.3
    assert new_state.membrane_potential.tolist() == pytest.approx([-66.5, -66.3])
    assert spikes.tolist() == [0.0, 0.0]


def test_step_dense_and_sparse_csr_weight_matrix_produce_same_result():
    dense_weight_matrix = torch.tensor(
        [[0.0, 0.0, 2.0], [3.0, 0.0, 0.0], [0.0, -1.0, 0.0]]
    )
    with accepting_sparse_csr_beta():
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
        dense_state, dense_weight_matrix, external_input, previous_spikes, PARAMETERS
    )
    sparse_new_state, sparse_spikes = step(
        sparse_state, sparse_weight_matrix, external_input, previous_spikes, PARAMETERS
    )

    assert dense_new_state.membrane_potential.tolist() == pytest.approx(
        sparse_new_state.membrane_potential.tolist()
    )
    assert dense_new_state.refractory_time_remaining.tolist() == pytest.approx(
        sparse_new_state.refractory_time_remaining.tolist()
    )
    assert dense_spikes.tolist() == pytest.approx(sparse_spikes.tolist())
