import { useRef } from "react";
import { useFrame } from "@react-three/fiber";
import * as THREE from "three";
import { placeFlatOnPage, type PageSurface } from "./pageSurface";
import type { ReadingChoreography } from "./ReadingChoreography";

const SHADOW_LIFT = 0.0012;
const SHADOW_BASE_SIZE = 0.14;
const SHADOW_MAX_OPACITY = 0.45;
// Height above the page at which the shadow has faded out completely.
const SHADOW_FADE_HEIGHT = 0.32;
// The shadow only lands on the pages; past their edges it fades out over this margin.
const PAGE_HALF_WIDTH = 0.4;
const PAGE_HALF_DEPTH = 0.3;
const EDGE_FADE_MARGIN = 0.04;

let sharedBlobTexture: THREE.CanvasTexture | null = null;

/** The soft radial shadow blob (dark middle, transparent rim), drawn once and kept for the
 * app's lifetime: it's tiny, and disposing a CanvasTexture that React StrictMode's trial
 * unmount hands back for reuse leaves it blank. */
function blobTexture(): THREE.CanvasTexture {
    if (sharedBlobTexture) return sharedBlobTexture;
    const size = 128;
    const canvas = document.createElement("canvas");
    canvas.width = size;
    canvas.height = size;
    const context = canvas.getContext("2d")!;
    const gradient = context.createRadialGradient(size / 2, size / 2, 0, size / 2, size / 2, size / 2);
    gradient.addColorStop(0, "rgba(0, 0, 0, 1)");
    gradient.addColorStop(0.45, "rgba(0, 0, 0, 0.55)");
    gradient.addColorStop(1, "rgba(0, 0, 0, 0)");
    context.fillStyle = gradient;
    context.fillRect(0, 0, size, size);
    sharedBlobTexture = new THREE.CanvasTexture(canvas);
    return sharedBlobTexture;
}

/** 1 inside the pages, fading to 0 over EDGE_FADE_MARGIN past an edge. */
function onPageAmount(x: number, z: number): number {
    const overshoot = Math.max(Math.abs(x) - PAGE_HALF_WIDTH, Math.abs(z) - PAGE_HALF_DEPTH, 0);
    return Math.max(0, 1 - overshoot / EDGE_FADE_MARGIN);
}

/** Contact shadow under the fly: tight and dark while it stands on the page, wider and
 * fainter the higher it hops or flies, so its height above the book stays readable. */
export function FlyShadow({ surface, choreography }: { surface: PageSurface; choreography: ReadingChoreography }) {
    const shadowRef = useRef<THREE.Mesh>(null);

    useFrame(() => {
        const shadow = shadowRef.current;
        if (!shadow) return;
        const { x, y, z } = choreography.flyPosition;
        const lift = Math.min(1, Math.max(0, y - surface.heightAt(x, z)) / SHADOW_FADE_HEIGHT);
        placeFlatOnPage(shadow, surface, x, z, SHADOW_LIFT);
        shadow.scale.setScalar(SHADOW_BASE_SIZE * (1 + lift * 1.2));
        (shadow.material as THREE.MeshBasicMaterial).opacity = SHADOW_MAX_OPACITY * (1 - lift) * onPageAmount(x, z);
    });

    return (
        <mesh ref={shadowRef} renderOrder={3}>
            <planeGeometry args={[1, 0.72]} />
            <meshBasicMaterial
                map={blobTexture()}
                transparent
                opacity={0}
                depthWrite={false}
                polygonOffset
                polygonOffsetFactor={-3}
                polygonOffsetUnits={-3}
            />
        </mesh>
    );
}
