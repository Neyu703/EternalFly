import * as THREE from "three";

/** Every mesh under root (root included), in traversal order. */
export function meshesOf(root: THREE.Object3D): THREE.Mesh[] {
    const meshes: THREE.Mesh[] = [];
    root.traverse((child) => {
        if (child instanceof THREE.Mesh) meshes.push(child);
    });
    return meshes;
}

/** Triangle indices for a grid of vertices laid out row by row, `columns + 1` per row and
 * `rows + 1` rows: two triangles per cell, wound the same way throughout. */
export function gridTriangleIndices(columns: number, rows: number): number[] {
    const verticesPerRow = columns + 1;
    const indices: number[] = [];
    for (let row = 0; row < rows; row += 1) {
        for (let column = 0; column < columns; column += 1) {
            const topLeft = row * verticesPerRow + column;
            const bottomLeft = topLeft + verticesPerRow;
            indices.push(topLeft, bottomLeft, topLeft + 1, bottomLeft, bottomLeft + 1, topLeft + 1);
        }
    }
    return indices;
}

/** A new canvas of the given pixel size and its 2D drawing context. */
export function createCanvas2D(width: number, height: number): {
    canvas: HTMLCanvasElement;
    context: CanvasRenderingContext2D;
} {
    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    return { canvas, context: canvas.getContext("2d")! };
}
