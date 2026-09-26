import * as THREE from "three";
import {
    LINES_PER_PAGE,
    TEXT_INNER_X,
    TEXT_OUTER_X,
    TEXT_TOP_Z,
    WORD_DEPTH_Z,
    lineWordBoxes,
    textStartX,
    type PageSide,
} from "./readingLayout";

/** "ink": transparent words-only layer lying on the book's own page; "paper": an opaque
 * whole-page sheet (white, tinted by the page material) for the page being turned. */
export type TextureKind = "ink" | "paper";

/** Axis-aligned rectangle of the book model's frame that a texture covers. */
export type PageRegion = { minX: number; maxX: number; minZ: number; maxZ: number };

export const SHEET_OUTER_X = 0.4;
export const SHEET_MIN_Z = -0.285;
export const SHEET_MAX_Z = 0.285;

const PAGE_NUMBER_Z = 0.262;
const PIXELS_PER_UNIT = 1800;
const INK_COLOR = "rgba(52, 42, 34, 0.68)";
const WORD_CORNER_RADIUS_PX = 5;
const TEXTURE_CACHE_LIMIT = 16;

const textureCache = new Map<string, THREE.CanvasTexture>();

/** The area an "ink" texture spans: the text block plus the page-number line below it. */
export function inkRegion(side: PageSide): PageRegion {
    const minX = textStartX(side);
    return { minX, maxX: minX + (TEXT_OUTER_X - TEXT_INNER_X), minZ: TEXT_TOP_Z - 0.01, maxZ: PAGE_NUMBER_Z + 0.02 };
}

/** The area a "paper" texture spans: the whole page, from the spine to its outer edge. */
export function paperRegion(side: PageSide): PageRegion {
    return side === "left"
        ? { minX: -SHEET_OUTER_X, maxX: 0, minZ: SHEET_MIN_Z, maxZ: SHEET_MAX_Z }
        : { minX: 0, maxX: SHEET_OUTER_X, minZ: SHEET_MIN_Z, maxZ: SHEET_MAX_Z };
}

/** The texture for one page of one spread, drawn once and kept in a small LRU cache
 * (evicted textures are disposed, so long reading sessions don't accumulate GPU memory). */
export function pageTexture(kind: TextureKind, spreadIndex: number, side: PageSide): THREE.CanvasTexture {
    const key = `${kind}:${spreadIndex}:${side}`;
    const cachedTexture = textureCache.get(key);
    if (cachedTexture) {
        textureCache.delete(key);
        textureCache.set(key, cachedTexture);
        return cachedTexture;
    }

    const texture = drawPageTexture(kind, spreadIndex, side);
    textureCache.set(key, texture);
    if (textureCache.size > TEXTURE_CACHE_LIMIT) {
        const oldestKey = textureCache.keys().next().value as string;
        textureCache.get(oldestKey)?.dispose();
        textureCache.delete(oldestKey);
    }
    return texture;
}

/** Puts texture on material, recompiling only when it actually changed. */
export function assignPageTexture(material: THREE.MeshStandardMaterial, texture: THREE.Texture): void {
    if (material.map === texture) return;
    material.map = texture;
    material.needsUpdate = true;
}

/** Draws a page's words (as rounded ink blocks) and its page number onto a canvas. */
function drawPageTexture(kind: TextureKind, spreadIndex: number, side: PageSide): THREE.CanvasTexture {
    const region = kind === "ink" ? inkRegion(side) : paperRegion(side);
    const canvas = document.createElement("canvas");
    canvas.width = Math.round((region.maxX - region.minX) * PIXELS_PER_UNIT);
    canvas.height = Math.round((region.maxZ - region.minZ) * PIXELS_PER_UNIT);
    const context = canvas.getContext("2d")!;

    if (kind === "paper") {
        context.fillStyle = "#ffffff";
        context.fillRect(0, 0, canvas.width, canvas.height);
    }

    context.fillStyle = INK_COLOR;
    for (let lineIndex = 0; lineIndex < LINES_PER_PAGE; lineIndex += 1) {
        for (const box of lineWordBoxes(spreadIndex, side, lineIndex)) {
            context.beginPath();
            context.roundRect(
                (box.centerX - box.width / 2 - region.minX) * PIXELS_PER_UNIT,
                (box.centerZ - WORD_DEPTH_Z / 2 - region.minZ) * PIXELS_PER_UNIT,
                box.width * PIXELS_PER_UNIT,
                WORD_DEPTH_Z * PIXELS_PER_UNIT,
                WORD_CORNER_RADIUS_PX,
            );
            context.fill();
        }
    }

    const pageNumber = spreadIndex * 2 + (side === "left" ? 1 : 2);
    const pageNumberX = side === "left" ? -TEXT_OUTER_X : TEXT_OUTER_X;
    context.font = `${Math.round(0.024 * PIXELS_PER_UNIT)}px Georgia, "Times New Roman", serif`;
    context.textAlign = side === "left" ? "left" : "right";
    context.textBaseline = "middle";
    context.fillText(
        String(pageNumber),
        (pageNumberX - region.minX) * PIXELS_PER_UNIT,
        (PAGE_NUMBER_Z - region.minZ) * PIXELS_PER_UNIT,
    );

    const texture = new THREE.CanvasTexture(canvas);
    texture.colorSpace = THREE.SRGBColorSpace;
    texture.anisotropy = 4;
    return texture;
}
