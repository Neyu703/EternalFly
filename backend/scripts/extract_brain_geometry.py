"""One-off script: extract the FlyWire brain outline mesh (GLB) and all 78 real neuropil
region meshes (GLB, one named node per region, colored by anatomical category) for the
frontend Brain-Glow visualization. Requires temporary packages `flybrains` and `fafbseg`
(not project dependencies - installed ad hoc, not in pyproject.toml). Not unit-tested -
throwaway asset-generation tooling, same convention as other scripts/."""

from pathlib import Path

import flybrains
import trimesh
from fafbseg import flywire

OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "frontend" / "public" / "models"

# nanometers -> scene units matching the brain outline mesh's own export scale below.
NANOMETERS_PER_SCENE_UNIT = 100_000.0

# Anatomical category -> base RGB color, chosen to echo the classic JRC2018 painted-brain
# renderings (optic lobes red/orange, central complex blue, mushroom body yellow/green, ...).
CATEGORY_COLORS = {
    "optic": (235, 90, 40),
    "central_complex": (70, 110, 235),
    "mushroom_body": (215, 210, 60),
    "olfactory": (60, 200, 190),
    "lateral_horn": (90, 200, 110),
    "other": (150, 130, 190),
}

NEUROPIL_CATEGORIES = {
    "optic": ["ME_L", "ME_R", "LO_L", "LO_R", "LOP_L", "LOP_R", "AME_L", "AME_R", "LA_L", "LA_R"],
    "central_complex": [
        "EB", "FB", "PB", "NO", "LAL_L", "LAL_R", "BU_L", "BU_R", "GA_L", "GA_R",
    ],
    "mushroom_body": [
        "MB_CA_L", "MB_CA_R", "MB_ML_L", "MB_ML_R", "MB_PED_L", "MB_PED_R", "MB_VL_L", "MB_VL_R",
    ],
    "olfactory": ["AL_L", "AL_R", "VES_L", "VES_R"],
    "lateral_horn": [
        "LH_L", "LH_R", "SLP_L", "SLP_R", "SMP_L", "SMP_R", "SIP_L", "SIP_R", "CRE_L", "CRE_R",
    ],
}

ALL_NEUROPIL_NAMES = [
    "AL_L", "AL_R", "AME_L", "AME_R", "AMMC_L", "AMMC_R", "AOTU_L", "AOTU_R", "ATL_L", "ATL_R",
    "AVLP_L", "AVLP_R", "BU_L", "BU_R", "CAN_L", "CAN_R", "CRE_L", "CRE_R", "EB", "EPA_L", "EPA_R",
    "FB", "FLA_L", "FLA_R", "GA_L", "GA_R", "GNG", "GOR_L", "GOR_R", "IB_L", "IB_R", "ICL_L", "ICL_R",
    "IPS_L", "IPS_R", "LAL_L", "LAL_R", "LA_L", "LA_R", "LH_L", "LH_R", "LOP_L", "LOP_R", "LO_L",
    "LO_R", "MB_CA_L", "MB_CA_R", "MB_ML_L", "MB_ML_R", "MB_PED_L", "MB_PED_R", "MB_VL_L", "MB_VL_R",
    "ME_L", "ME_R", "NO", "OCG", "PB", "PLP_L", "PLP_R", "PRW", "PVLP_L", "PVLP_R", "SAD", "SCL_L",
    "SCL_R", "SIP_L", "SIP_R", "SLP_L", "SLP_R", "SMP_L", "SMP_R", "SPS_L", "SPS_R", "VES_L", "VES_R",
    "WED_L", "WED_R",
]


def category_for_neuropil(neuropil_name: str) -> str:
    """Return the anatomical category a neuropil belongs to, defaulting to "other"."""
    for category_name, members in NEUROPIL_CATEGORIES.items():
        if neuropil_name in members:
            return category_name
    return "other"


def export_brain_outline(output_path: Path) -> None:
    """Export the whole-brain FlyWire outline mesh as a scaled-down GLB."""
    brain_mesh = flybrains.FLYWIRE.mesh.copy()
    brain_mesh.apply_scale(1.0 / NANOMETERS_PER_SCENE_UNIT)
    scene = trimesh.Scene({"brain_outline": brain_mesh})
    scene.export(output_path)
    print(f"brain outline: {len(brain_mesh.vertices)} vertices -> {output_path}")


def export_neuropil_regions(output_path: Path) -> None:
    """Fetch every neuropil's real mesh, color it by anatomical category, export as one GLB."""
    volumes = flywire.get_neuropil_volumes(ALL_NEUROPIL_NAMES)
    scene = trimesh.Scene()
    for volume in volumes:
        region_mesh = trimesh.Trimesh(vertices=volume.vertices, faces=volume.faces)
        region_mesh.apply_scale(1.0 / NANOMETERS_PER_SCENE_UNIT)
        category = category_for_neuropil(volume.name)
        red, green, blue = CATEGORY_COLORS[category]
        region_mesh.visual.vertex_colors = [red, green, blue, 255]
        scene.add_geometry(region_mesh, node_name=volume.name, geom_name=volume.name)
    scene.export(output_path)
    print(f"neuropil regions: {len(volumes)} regions -> {output_path}")


def main() -> None:
    """Extract the brain outline mesh and the colored per-neuropil region meshes."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    export_brain_outline(OUTPUT_DIR / "brain-outline.glb")
    export_neuropil_regions(OUTPUT_DIR / "neuropil-regions.glb")


if __name__ == "__main__":
    main()
