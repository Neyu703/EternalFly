"""Pure transforms from FlyWire neuron positions to the frontend's 3D scene coordinates."""

import numpy

# FlyWire's segmentation voxel resolution is 4x4x40 nanometers (x,y,z) - the annotation
# table's pos_x/pos_y/pos_z columns are in this voxel space, not nanometers or scene
# units. NANOMETERS_PER_SCENE_UNIT matches scripts/extract_brain_geometry.py's own
# scale factor for the brain-outline/neuropil-region meshes, so neuron positions land
# in the same frame as those meshes (verified: converted real annotation positions land
# well inside brain-outline.glb's own vertex bounding box).
VOXEL_SIZE_NANOMETERS = (4.0, 4.0, 40.0)
NANOMETERS_PER_SCENE_UNIT = 100_000.0


def neuron_scene_positions(
    root_ids: numpy.ndarray,
    annotation_root_ids: numpy.ndarray,
    pos_x: numpy.ndarray,
    pos_y: numpy.ndarray,
    pos_z: numpy.ndarray,
) -> numpy.ndarray:
    """Return an (len(root_ids), 3) float32 array of scene-unit xyz positions, in
    root_ids' order (the same order the adjacency/synapse matrices index neurons by).

    annotation_root_ids/pos_x/pos_y/pos_z are the annotation table's parallel columns
    (voxel-space positions); a root_id with no matching annotation row (a small number
    of real proofread neurons lack one) gets the scene origin (0, 0, 0) rather than
    raising, since a missing position only affects where one point in the frontend's
    spike-cloud visualization renders, not the simulation itself.
    """
    annotation_root_ids = annotation_root_ids.astype(numpy.int64)
    sort_order = numpy.argsort(annotation_root_ids)
    sorted_annotation_ids = annotation_root_ids[sort_order]

    lookup_root_ids = root_ids.astype(numpy.int64)
    positions_in_sorted = numpy.clip(numpy.searchsorted(sorted_annotation_ids, lookup_root_ids), 0, len(sorted_annotation_ids) - 1)
    has_annotation = sorted_annotation_ids[positions_in_sorted] == lookup_root_ids
    annotation_positions = sort_order[positions_in_sorted]

    scene_positions = numpy.zeros((len(root_ids), 3), dtype=numpy.float32)
    voxel_scale = numpy.array(VOXEL_SIZE_NANOMETERS, dtype=numpy.float64) / NANOMETERS_PER_SCENE_UNIT

    matched_annotation_positions = annotation_positions[has_annotation]
    scene_positions[has_annotation, 0] = pos_x[matched_annotation_positions] * voxel_scale[0]
    scene_positions[has_annotation, 1] = pos_y[matched_annotation_positions] * voxel_scale[1]
    scene_positions[has_annotation, 2] = pos_z[matched_annotation_positions] * voxel_scale[2]

    return scene_positions
