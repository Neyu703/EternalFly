import { useEffect, useMemo, useRef } from "react";
import { useFrame } from "@react-three/fiber";
import * as THREE from "three";
import { LINE_SPACING_Z, type PageSide } from "./readingLayout";
import { assignPageTexture, inkRegion, pageTexture } from "./pageTextures";
import { placeFlatOnPage, type PageSurface } from "./pageSurface";
import type { ReadingChoreography } from "./ReadingChoreography";
import { PageMatchedMaterial } from "./PageMatchedMaterial";
import { lerp } from "../../utils/math";
import { gridTriangleIndices } from "../../utils/scene";

const OVERLAY_LIFT = 0.0015;
const HIGHLIGHT_LIFT = 0.0008;
const HIGHLIGHT_PADDING_X = 0.014;
const HIGHLIGHT_DEPTH = LINE_SPACING_Z * 0.72;
const HIGHLIGHT_MAX_OPACITY = 0.5;
const OVERLAY_COLUMNS = 24;
const OVERLAY_ROWS = 10;

/** Builds a grid mesh covering a page's ink region that hugs the curved page surface,
 * with UVs matching pageTexture's layout (canvas top = the page's far edge). */
function buildInkOverlayGeometry(surface: PageSurface, side: PageSide): THREE.BufferGeometry {
    const region = inkRegion(side);
    const positions: number[] = [];
    const uvs: number[] = [];
    for (let row = 0; row <= OVERLAY_ROWS; row += 1) {
        const rowFraction = row / OVERLAY_ROWS;
        const z = lerp(region.minZ, region.maxZ, rowFraction);
        for (let column = 0; column <= OVERLAY_COLUMNS; column += 1) {
            const columnFraction = column / OVERLAY_COLUMNS;
            const x = lerp(region.minX, region.maxX, columnFraction);
            positions.push(x, surface.heightAt(x, z) + OVERLAY_LIFT, z);
            uvs.push(columnFraction, 1 - rowFraction);
        }
    }
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
    geometry.setAttribute("uv", new THREE.Float32BufferAttribute(uvs, 2));
    geometry.setIndex(gridTriangleIndices(OVERLAY_COLUMNS, OVERLAY_ROWS));
    geometry.computeVertexNormals();
    return geometry;
}

/** The right page's spread: during a turn it already shows what the lifting sheet uncovers. */
function rightPageSpread(choreography: ReadingChoreography): number {
    return choreography.flip ? choreography.flip.fromSpread + 1 : choreography.displayedSpread;
}

/** The text printed on the two visible pages of the current spread, plus a soft
 * highlighter that glides under the word the fly is reading. */
export function PageText({
    surface,
    choreography,
    pageMaterial,
}: {
    surface: PageSurface;
    choreography: ReadingChoreography;
    pageMaterial: THREE.MeshStandardMaterial;
}) {
    return (
        <>
            <ReadingHighlight surface={surface} choreography={choreography} />
            <InkPage
                side="left"
                surface={surface}
                pageMaterial={pageMaterial}
                spreadOf={() => choreography.displayedSpread}
            />
            <InkPage
                side="right"
                surface={surface}
                pageMaterial={pageMaterial}
                spreadOf={() => rightPageSpread(choreography)}
            />
        </>
    );
}

/** A words-only layer lying on one page, lit and tinted like the page itself, that
 * swaps to whichever spread `spreadOf` names each frame. */
function InkPage({
    side,
    surface,
    pageMaterial,
    spreadOf,
}: {
    side: PageSide;
    surface: PageSurface;
    pageMaterial: THREE.MeshStandardMaterial;
    spreadOf: () => number;
}) {
    const geometry = useMemo(() => buildInkOverlayGeometry(surface, side), [surface, side]);
    const meshRef = useRef<THREE.Mesh>(null);
    useEffect(() => () => geometry.dispose(), [geometry]);

    useFrame(() => {
        const mesh = meshRef.current;
        if (mesh) assignPageTexture(mesh.material as THREE.MeshStandardMaterial, pageTexture("ink", spreadOf(), side));
    });

    return (
        <mesh ref={meshRef} geometry={geometry} renderOrder={2}>
            <PageMatchedMaterial
                pageMaterial={pageMaterial}
                transparent
                depthWrite={false}
                polygonOffset
                polygonOffsetFactor={-2}
                polygonOffsetUnits={-2}
            />
        </mesh>
    );
}

/** A soft yellow highlighter marker under the word being read, following the page's
 * slope and fading with the choreography's highlight opacity. */
function ReadingHighlight({ surface, choreography }: { surface: PageSurface; choreography: ReadingChoreography }) {
    const highlightRef = useRef<THREE.Mesh>(null);

    useFrame(() => {
        const highlight = highlightRef.current;
        if (!highlight) return;
        const { highlightCenter } = choreography;
        placeFlatOnPage(highlight, surface, highlightCenter.x, highlightCenter.z, HIGHLIGHT_LIFT);
        highlight.scale.set(choreography.highlightWidth + HIGHLIGHT_PADDING_X, HIGHLIGHT_DEPTH, 1);
        (highlight.material as THREE.MeshBasicMaterial).opacity = choreography.highlightOpacity * HIGHLIGHT_MAX_OPACITY;
    });

    return (
        <mesh ref={highlightRef} renderOrder={1}>
            <planeGeometry args={[1, 1]} />
            <meshBasicMaterial
                color="#ffd84d"
                transparent
                opacity={0}
                depthWrite={false}
                toneMapped={false}
                polygonOffset
                polygonOffsetFactor={-1}
                polygonOffsetUnits={-1}
            />
        </mesh>
    );
}
