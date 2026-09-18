"""One-off script: extract the FlyWire brain outline mesh (GLB) and per-neuropil centroid
positions (JSON) for the frontend Brain-Glow visualization. Requires temporary packages
`flybrains` and `fafbseg` (not project dependencies - installed ad hoc, not in pyproject.toml).
Not unit-tested - throwaway asset-generation tooling, same convention as other scripts/."""

import json
from pathlib import Path

import flybrains
import numpy
import trimesh
from fafbseg import flywire

OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "frontend" / "public" / "models"
NEUROPIL_NAMES = [
    "AL_L", "AL_R", "AME_L", "AME_R", "AMMC_L", "AMMC_R", "AOTU_L", "AOTU_R", "ATL_L", "ATL_R",
    "AVLP_L", "AVLP_R", "BU_L", "BU_R", "CAN_L", "CAN_R", "CRE_L", "CRE_R", "EB", "EPA_L", "EPA_R",
    "FB", "FLA_L", "FLA_R", "GA_L", "GA_R", "GNG", "GOR_L", "GOR_R", "IB_L", "IB_R", "ICL_L", "ICL_R",
    "IPS_L", "IPS_R", "LAL_L", "LAL_R", "LA_L", "LA_R", "LH_L", "LH_R", "LOP_L", "LOP_R", "LO_L",
    "LO_R", "MB_CA_L", "MB_CA_R", "MB_ML_L", "MB_ML_R", "MB_PED_L", "MB_PED_R", "MB_VL_L", "MB_VL_R",
    "ME_L", "ME_R", "NO", "OCG", "PB", "PLP_L", "PLP_R", "PRW", "PVLP_L", "PVLP_R", "SAD", "SCL_L",
    "SCL_R", "SIP_L", "SIP_R", "SLP_L", "SLP_R", "SMP_L", "SMP_R", "SPS_L", "SPS_R", "VES_L", "VES_R",
    "WED_L", "WED_R",
]

# nanometers -> scene units matching the brain outline mesh's own export scale below.
NANOMETERS_PER_SCENE_UNIT = 100_000.0


def export_brain_outline(output_path: Path) -> None:
    """Export the whole-brain FlyWire outline mesh as a scaled-down GLB."""
    brain_mesh = flybrains.FLYWIRE.mesh.copy()
    brain_mesh.apply_scale(1.0 / NANOMETERS_PER_SCENE_UNIT)
    scene = trimesh.Scene({"brain_outline": brain_mesh})
    scene.export(output_path)
    print(f"brain outline: {len(brain_mesh.vertices)} vertices -> {output_path}")


def export_neuropil_centroids(output_path: Path) -> None:
    """Fetch every neuropil's mesh, compute its centroid in scene units, save as JSON."""
    volumes = flywire.get_neuropil_volumes(NEUROPIL_NAMES)
    centroids = {}
    for volume in volumes:
        vertices = numpy.asarray(volume.vertices, dtype=float) / NANOMETERS_PER_SCENE_UNIT
        centroid = vertices.mean(axis=0)
        centroids[volume.name] = [round(float(coordinate), 4) for coordinate in centroid]
    output_path.write_text(json.dumps(centroids, indent=2))
    print(f"neuropil centroids: {len(centroids)} regions -> {output_path}")


def main() -> None:
    """Extract both the brain outline mesh and neuropil centroid lookup table."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    export_brain_outline(OUTPUT_DIR / "brain-outline.glb")
    export_neuropil_centroids(OUTPUT_DIR / "neuropil-centroids.json")


if __name__ == "__main__":
    main()
