import * as THREE from "three";
import { mergeGeometries } from "three/examples/jsm/utils/BufferGeometryUtils.js";
import { meshesOf } from "../../utils/scene";

// Only edges where neighbouring faces bend by more than this become lines: each region's
// real contours instead of every triangle edge (~35k instead of ~150k line segments).
const EDGE_THRESHOLD_DEGREES = 30;

/** One neuropil region: its code (e.g. "ME_R"), the hidden source mesh used for pointer
 * picking, its center in the region model's frame, and its anatomical color as CSS hex. */
export type BrainRegion = { code: string; mesh: THREE.Mesh; center: THREE.Vector3; cssColor: string };

/** All regions drawn with two draw calls: one merged surface and one merged edge set,
 * each vertex tagged with its region's index (the `regionIndex` attribute). */
export type BrainRegionGeometry = {
    regions: BrainRegion[];
    surfaces: THREE.BufferGeometry;
    edges: THREE.BufferGeometry;
};

/** A per-vertex attribute holding the same value(s) on every vertex. */
function constantAttribute(vertexCount: number, values: number[]): THREE.Float32BufferAttribute {
    const data = new Float32Array(vertexCount * values.length);
    for (let vertex = 0; vertex < vertexCount; vertex += 1) data.set(values, vertex * values.length);
    return new THREE.Float32BufferAttribute(data, values.length);
}

/** Tags a geometry with its region's color and index on every vertex. */
function tagWithRegion(geometry: THREE.BufferGeometry, color: THREE.Color, regionIndex: number): void {
    const vertexCount = geometry.getAttribute("position").count;
    geometry.setAttribute("color", constantAttribute(vertexCount, [color.r, color.g, color.b]));
    geometry.setAttribute("regionIndex", constantAttribute(vertexCount, [regionIndex]));
}

/** The region's uniform baked color (the server bakes one color onto every vertex). */
function bakedRegionColor(mesh: THREE.Mesh): THREE.Color {
    const colorAttribute = mesh.geometry.getAttribute("color");
    if (!colorAttribute) return new THREE.Color("#9a86be");
    return new THREE.Color(colorAttribute.getX(0), colorAttribute.getY(0), colorAttribute.getZ(0));
}

/** Merges the 78 neuropil region meshes of the regions model into one surface geometry
 * (with smooth normals for rim lighting) and one contour-edge geometry. The source meshes
 * are kept, untouched, for pointer picking. */
export function buildBrainRegions(regionsScene: THREE.Object3D): BrainRegionGeometry {
    const sourceMeshes = meshesOf(regionsScene);

    const regions: BrainRegion[] = [];
    const surfaceParts: THREE.BufferGeometry[] = [];
    const edgeParts: THREE.BufferGeometry[] = [];
    sourceMeshes.forEach((mesh, regionIndex) => {
        const color = bakedRegionColor(mesh);
        const surface = new THREE.BufferGeometry();
        surface.setAttribute("position", mesh.geometry.getAttribute("position").clone());
        surface.setIndex(mesh.geometry.getIndex()!.clone());
        surface.computeVertexNormals();
        surface.computeBoundingBox();

        const edges = new THREE.EdgesGeometry(surface, EDGE_THRESHOLD_DEGREES);
        tagWithRegion(surface, color, regionIndex);
        tagWithRegion(edges, color, regionIndex);
        surfaceParts.push(surface);
        edgeParts.push(edges);
        regions.push({
            code: mesh.name,
            mesh,
            center: surface.boundingBox!.getCenter(new THREE.Vector3()),
            cssColor: `#${color.getHexString()}`,
        });
    });

    const surfaces = mergeGeometries(surfaceParts)!;
    const edges = mergeGeometries(edgeParts)!;
    [...surfaceParts, ...edgeParts].forEach((part) => part.dispose());
    return { regions, surfaces, edges };
}
