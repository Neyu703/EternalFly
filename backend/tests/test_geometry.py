import numpy

from eternalfly.geometry import NANOMETERS_PER_SCENE_UNIT, VOXEL_SIZE_NANOMETERS, neuron_scene_positions


def test_neuron_scene_positions_converts_voxel_coordinates_to_scene_units():
    root_ids = numpy.array([100])
    annotation_root_ids = numpy.array([100])
    pos_x = numpy.array([1000.0])
    pos_y = numpy.array([2000.0])
    pos_z = numpy.array([3000.0])

    positions = neuron_scene_positions(root_ids, annotation_root_ids, pos_x, pos_y, pos_z)

    expected_x = 1000.0 * VOXEL_SIZE_NANOMETERS[0] / NANOMETERS_PER_SCENE_UNIT
    expected_y = 2000.0 * VOXEL_SIZE_NANOMETERS[1] / NANOMETERS_PER_SCENE_UNIT
    expected_z = 3000.0 * VOXEL_SIZE_NANOMETERS[2] / NANOMETERS_PER_SCENE_UNIT
    assert positions.shape == (1, 3)
    assert numpy.allclose(positions[0], [expected_x, expected_y, expected_z])


def test_neuron_scene_positions_preserves_root_ids_order_not_annotation_order():
    root_ids = numpy.array([200, 100])
    annotation_root_ids = numpy.array([100, 200])
    pos_x = numpy.array([1.0, 2.0])
    pos_y = numpy.array([0.0, 0.0])
    pos_z = numpy.array([0.0, 0.0])

    positions = neuron_scene_positions(root_ids, annotation_root_ids, pos_x, pos_y, pos_z)

    scale_x = VOXEL_SIZE_NANOMETERS[0] / NANOMETERS_PER_SCENE_UNIT
    assert numpy.allclose(positions[:, 0], [2.0 * scale_x, 1.0 * scale_x])


def test_neuron_scene_positions_defaults_to_origin_for_unannotated_neuron():
    root_ids = numpy.array([100, 999])
    annotation_root_ids = numpy.array([100])
    pos_x = numpy.array([1000.0])
    pos_y = numpy.array([1000.0])
    pos_z = numpy.array([1000.0])

    positions = neuron_scene_positions(root_ids, annotation_root_ids, pos_x, pos_y, pos_z)

    assert numpy.allclose(positions[1], [0.0, 0.0, 0.0])
    assert not numpy.allclose(positions[0], [0.0, 0.0, 0.0])


def test_neuron_scene_positions_returns_float32_array():
    root_ids = numpy.array([100])
    annotation_root_ids = numpy.array([100])
    positions = neuron_scene_positions(root_ids, annotation_root_ids, numpy.array([1.0]), numpy.array([1.0]), numpy.array([1.0]))

    assert positions.dtype == numpy.float32
