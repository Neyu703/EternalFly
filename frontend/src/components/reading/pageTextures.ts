import * as THREE from "three";
import { createCanvas2D } from "../../utils/scene";
import {
    LINES_PER_PAGE,
    PAGE_OUTER_X,
    TEXT_OUTER_X,
    TEXT_TOP_Z,
    TEXT_WIDTH_X,
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

export const SHEET_OUTER_X = PAGE_OUTER_X;
export const SHEET_MIN_Z = -0.285;
export const SHEET_MAX_Z = 0.285;

const PAGE_NUMBER_Z = 0.262;
const PAGE_NUMBER_FONT_SIZE = 0.024;
// The ink region reaches a little above the text block and below the page number, so
// neither is cut off at the texture's edge.
const INK_MARGIN_TOP_Z = 0.01;
const INK_MARGIN_BOTTOM_Z = 0.02;
// Keeps the words crisp on the pages' slanted view.
const TEXTURE_ANISOTROPY = 4;
const PIXELS_PER_UNIT = 1800;
const INK_COLOR = "rgba(52, 42, 34, 0.68)";
const WORD_CORNER_RADIUS_PX = 5;
const TEXTURE_CACHE_LIMIT = 16;

const textureCache = new Map<string, THREE.CanvasTexture>();

/** The area an "ink" texture spans: the text block plus the page-number line below it. */
export function inkRegion(side: PageSide): PageRegion {
    const minX = textStartX(side);
    return {
        minX,
        maxX: minX + TEXT_WIDTH_X,
        minZ: TEXT_TOP_Z - INK_MARGIN_TOP_Z,
        maxZ: PAGE_NUMBER_Z + INK_MARGIN_BOTTOM_Z,
    };
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
    const { canvas, context } = createCanvas2D(
        Math.round((region.maxX - region.minX) * PIXELS_PER_UNIT),
        Math.round((region.maxZ - region.minZ) * PIXELS_PER_UNIT),
    );

    if (kind === "paper") {
        context.fillStyle = "#ffffff";
        context.fillRect(0, 0, canvas.width, canvas.height);
    }
    drawWordBlocks(context, region, spreadIndex, side);
    drawPageNumber(context, region, spreadIndex, side);

    const texture = new THREE.CanvasTexture(canvas);
    texture.colorSpace = THREE.SRGBColorSpace;
    texture.anisotropy = TEXTURE_ANISOTROPY;
    return texture;
}

/** Draws every word of the page as a rounded ink block, positioned relative to region. */
function drawWordBlocks(context: CanvasRenderingContext2D, region: PageRegion, spreadIndex: number, side: PageSide): void {
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
}

/** Draws the page's number at the outer end of its page-number line. */
function drawPageNumber(context: CanvasRenderingContext2D, region: PageRegion, spreadIndex: number, side: PageSide): void {
    const pageNumber = spreadIndex * 2 + (side === "left" ? 1 : 2);
    const pageNumberX = side === "left" ? -TEXT_OUTER_X : TEXT_OUTER_X;
    context.font = `${Math.round(PAGE_NUMBER_FONT_SIZE * PIXELS_PER_UNIT)}px Georgia, "Times New Roman", serif`;
    context.textAlign = side === "left" ? "left" : "right";
    context.textBaseline = "middle";
    context.fillText(
        String(pageNumber),
        (pageNumberX - region.minX) * PIXELS_PER_UNIT,
        (PAGE_NUMBER_Z - region.minZ) * PIXELS_PER_UNIT,
    );
}
