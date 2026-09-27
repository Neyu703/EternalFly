import * as THREE from "three";
import { clamp, lerp } from "../../utils/math";
import { PAGE_OUTER_X } from "./readingLayout";

/** Height of the open book's page surface, in the book model's own frame. */
export type PageSurface = {
    heightAt: (x: number, z: number) => number;
};

const SAMPLE_MIN_X = -PAGE_OUTER_X;
const SAMPLE_MAX_X = PAGE_OUTER_X;
const SAMPLE_MIN_Z = -0.29;
const SAMPLE_MAX_Z = 0.29;
const SAMPLE_COUNT_X = 49;
const SAMPLE_COUNT_Z = 9;
const DOWNWARD = new THREE.Vector3(0, -1, 0);
const WORLD_Z_AXIS = new THREE.Vector3(0, 0, 1);
const SLOPE_PROBE_DISTANCE = 0.01;

/** Raycasts the book model once on a grid to learn the (curved) page surface, so the fly
 * can walk on it and the text, highlight and turning page can hug it. Works on a detached
 * clone so the result is in the model's own frame, independent of where it's placed. */
export function samplePageSurface(bookScene: THREE.Object3D): PageSurface {
    const probe = bookScene.clone(true);
    probe.position.set(0, 0, 0);
    probe.quaternion.identity();
    probe.scale.set(1, 1, 1);
    probe.updateMatrixWorld(true);

    const raycaster = new THREE.Raycaster();
    const rayOrigin = new THREE.Vector3();
    const heights = new Float32Array(SAMPLE_COUNT_X * SAMPLE_COUNT_Z);
    for (let zIndex = 0; zIndex < SAMPLE_COUNT_Z; zIndex += 1) {
        for (let xIndex = 0; xIndex < SAMPLE_COUNT_X; xIndex += 1) {
            rayOrigin.set(
                gridCoordinate(xIndex, SAMPLE_COUNT_X, SAMPLE_MIN_X, SAMPLE_MAX_X),
                1,
                gridCoordinate(zIndex, SAMPLE_COUNT_Z, SAMPLE_MIN_Z, SAMPLE_MAX_Z),
            );
            raycaster.set(rayOrigin, DOWNWARD);
            const topHit = raycaster.intersectObject(probe, true)[0];
            heights[zIndex * SAMPLE_COUNT_X + xIndex] = topHit ? topHit.point.y : 0;
        }
    }

    return {
        heightAt: (x, z) => bilinearHeight(heights, x, z),
    };
}

/** Lays a flat object (modelled in the XY plane) onto the page at (x, z), lifted by `lift`
 * and tilted to follow the page's slope across the spread. */
export function placeFlatOnPage(object: THREE.Object3D, surface: PageSurface, x: number, z: number, lift: number): void {
    const slope =
        (surface.heightAt(x + SLOPE_PROBE_DISTANCE, z) - surface.heightAt(x - SLOPE_PROBE_DISTANCE, z)) /
        (2 * SLOPE_PROBE_DISTANCE);
    object.position.set(x, surface.heightAt(x, z) + lift, z);
    object.rotation.set(-Math.PI / 2, 0, 0);
    object.rotateOnWorldAxis(WORLD_Z_AXIS, Math.atan(slope));
}

/** The coordinate of sample `index` of `count` samples spread evenly from min to max. */
function gridCoordinate(index: number, count: number, min: number, max: number): number {
    return lerp(min, max, index / (count - 1));
}

/** The fractional sample index at coordinate `value` (the inverse of gridCoordinate),
 * clamped into the sampled range [0, count - 1]. */
function gridPosition(value: number, count: number, min: number, max: number): number {
    return clamp(((value - min) / (max - min)) * (count - 1), 0, count - 1);
}

/** Bilinearly interpolated sampled height at (x, z), clamped to the sampled area. */
function bilinearHeight(heights: Float32Array, x: number, z: number): number {
    const column = gridPosition(x, SAMPLE_COUNT_X, SAMPLE_MIN_X, SAMPLE_MAX_X);
    const row = gridPosition(z, SAMPLE_COUNT_Z, SAMPLE_MIN_Z, SAMPLE_MAX_Z);
    const leftColumn = Math.floor(column);
    const topRow = Math.floor(row);
    const rightColumn = Math.min(leftColumn + 1, SAMPLE_COUNT_X - 1);
    const bottomRow = Math.min(topRow + 1, SAMPLE_COUNT_Z - 1);
    const columnFraction = column - leftColumn;
    const rowFraction = row - topRow;
    const heightAtRow = (rowIndex: number) =>
        heights[rowIndex * SAMPLE_COUNT_X + leftColumn] * (1 - columnFraction) +
        heights[rowIndex * SAMPLE_COUNT_X + rightColumn] * columnFraction;
    return heightAtRow(topRow) * (1 - rowFraction) + heightAtRow(bottomRow) * rowFraction;
}
