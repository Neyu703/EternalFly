import { useEffect, useMemo, useRef } from "react";
import { useFrame } from "@react-three/fiber";
import * as THREE from "three";
import { SHEET_MAX_Z, SHEET_MIN_Z, SHEET_OUTER_X, assignPageTexture, pageTexture } from "./pageTextures";
import type { ReadingChoreography } from "./ReadingChoreography";
import type { PageSurface } from "./pageSurface";
import { PageMatchedMaterial } from "./PageMatchedMaterial";
import { clamp01, easeInOutCubic, lerp } from "../../utils/math";
import { gridTriangleIndices } from "../../utils/scene";

const SEGMENT_COUNT = 28;
const ROW_COUNT = 8;
const SHEET_LIFT = 0.003;
// How far the sheet's outer edge leads while rising and trails while settling (radians).
const EDGE_LEAD = 0.55;
const EDGE_SAG = 0.3;
// The page's bottom corner (nearest the reader) turns ahead of its top corner.
const CORNER_LEAD = 0.22;
// How sharply that edge bend concentrates toward the outer edge (1 = evenly along the sheet).
const EDGE_BEND_FALLOFF = 1.4;

/** One strip of the sheet at a fixed depth z: its resting shape on the right page as a
 * chain of segments hinged at the spine. */
type SheetRow = { z: number; spineY: number; segmentLengths: number[]; restAngles: number[] };

/** Samples the right page's curved profile into hinge chains, one per sheet row. */
function buildSheetRows(surface: PageSurface): SheetRow[] {
    return Array.from({ length: ROW_COUNT }, (_, row) => {
        const z = lerp(SHEET_MIN_Z, SHEET_MAX_Z, row / (ROW_COUNT - 1));
        const nodeHeights = Array.from({ length: SEGMENT_COUNT + 1 }, (_, node) =>
            surface.heightAt((node / SEGMENT_COUNT) * SHEET_OUTER_X, z),
        );
        const segmentWidth = SHEET_OUTER_X / SEGMENT_COUNT;
        const segmentLengths: number[] = [];
        const restAngles: number[] = [];
        for (let segment = 0; segment < SEGMENT_COUNT; segment += 1) {
            const rise = nodeHeights[segment + 1] - nodeHeights[segment];
            segmentLengths.push(Math.hypot(segmentWidth, rise));
            restAngles.push(Math.atan2(rise, segmentWidth));
        }
        return { z, spineY: nodeHeights[0], segmentLengths, restAngles };
    });
}

/** Front (right-page side) and back (left-page side) geometries of the sheet. They share
 * one position and normal buffer and differ only in UVs: the back is mirrored, because
 * after the turn the sheet's spine edge sits on the right of the left page. */
function createSheetGeometries(): { front: THREE.BufferGeometry; back: THREE.BufferGeometry } {
    const vertexCount = (SEGMENT_COUNT + 1) * ROW_COUNT;
    const positionAttribute = new THREE.BufferAttribute(new Float32Array(vertexCount * 3), 3);
    positionAttribute.setUsage(THREE.DynamicDrawUsage);
    const frontUvs: number[] = [];
    const backUvs: number[] = [];
    for (let row = 0; row < ROW_COUNT; row += 1) {
        const v = 1 - row / (ROW_COUNT - 1);
        for (let node = 0; node <= SEGMENT_COUNT; node += 1) {
            const u = node / SEGMENT_COUNT;
            frontUvs.push(u, v);
            backUvs.push(1 - u, v);
        }
    }
    const indices = gridTriangleIndices(SEGMENT_COUNT, ROW_COUNT - 1);
    return {
        front: sheetGeometry(positionAttribute, frontUvs, indices),
        back: sheetGeometry(positionAttribute, backUvs, indices),
    };
}

/** One face of the sheet: the shared positions with this face's UVs. */
function sheetGeometry(positions: THREE.BufferAttribute, uvs: number[], indices: number[]): THREE.BufferGeometry {
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", positions);
    geometry.setAttribute("uv", new THREE.Float32BufferAttribute(uvs, 2));
    geometry.setIndex(indices);
    return geometry;
}

/** Bends every row's chain for the given linear turn progress and writes the result into
 * the shared position buffer. Each segment swings from its resting angle on the right
 * page to the mirrored angle on the left page; the outer edge leads while the page rises
 * and trails while it settles, and rows nearer the reader run slightly ahead. */
function layoutSheet(positions: THREE.BufferAttribute, rows: SheetRow[], progress: number): void {
    rows.forEach((row, rowIndex) => {
        const rowFraction = rowIndex / (ROW_COUNT - 1);
        const rowProgress = easeInOutCubic(clamp01(progress * (1 + CORNER_LEAD) - CORNER_LEAD * (1 - rowFraction)));
        const edgeBend = EDGE_LEAD * Math.sin(2 * Math.PI * rowProgress) - EDGE_SAG * Math.sin(Math.PI * rowProgress);
        let x = 0;
        let y = row.spineY + SHEET_LIFT;
        const rowStart = rowIndex * (SEGMENT_COUNT + 1);
        positions.setXYZ(rowStart, x, y, row.z);
        row.restAngles.forEach((restAngle, segment) => {
            const swungAngle = restAngle + (Math.PI - 2 * restAngle) * rowProgress;
            const angle = swungAngle + edgeBend * ((segment + 1) / SEGMENT_COUNT) ** EDGE_BEND_FALLOFF;
            x += row.segmentLengths[segment] * Math.cos(angle);
            y += row.segmentLengths[segment] * Math.sin(angle);
            positions.setXYZ(rowStart + segment + 1, x, y, row.z);
        });
    });
    positions.needsUpdate = true;
}

/** The page being turned: a paper sheet hinged at the spine that curls from the right
 * page over to the left one, printed with the old right page on its front and the next
 * spread's left page on its back. Hidden whenever no page is turning. */
export function FlippingPage({
    surface,
    choreography,
    pageMaterial,
}: {
    surface: PageSurface;
    choreography: ReadingChoreography;
    pageMaterial: THREE.MeshStandardMaterial;
}) {
    const rows = useMemo(() => buildSheetRows(surface), [surface]);
    const geometries = useMemo(() => createSheetGeometries(), []);
    const sheetRef = useRef<THREE.Group>(null);
    const frontRef = useRef<THREE.Mesh>(null);
    const backRef = useRef<THREE.Mesh>(null);

    useEffect(
        () => () => {
            geometries.front.dispose();
            geometries.back.dispose();
        },
        [geometries],
    );

    useFrame(() => {
        const sheet = sheetRef.current;
        const frontMesh = frontRef.current;
        const backMesh = backRef.current;
        if (!sheet || !frontMesh || !backMesh) return;
        const { flip } = choreography;
        sheet.visible = flip !== null;
        if (!flip) return;

        assignPageTexture(frontMesh.material as THREE.MeshStandardMaterial, pageTexture("paper", flip.fromSpread, "right"));
        assignPageTexture(backMesh.material as THREE.MeshStandardMaterial, pageTexture("paper", flip.fromSpread + 1, "left"));
        const frontGeometry = frontMesh.geometry;
        layoutSheet(frontGeometry.getAttribute("position") as THREE.BufferAttribute, rows, choreography.flipProgress);
        frontGeometry.computeVertexNormals();
        frontGeometry.computeBoundingSphere();
        backMesh.geometry.setAttribute("normal", frontGeometry.getAttribute("normal"));
        backMesh.geometry.boundingSphere = frontGeometry.boundingSphere;
    });

    return (
        <group ref={sheetRef} visible={false}>
            <mesh ref={frontRef} geometry={geometries.front}>
                <PageMatchedMaterial pageMaterial={pageMaterial} side={THREE.FrontSide} />
            </mesh>
            <mesh ref={backRef} geometry={geometries.back}>
                <PageMatchedMaterial pageMaterial={pageMaterial} side={THREE.BackSide} />
            </mesh>
        </group>
    );
}
