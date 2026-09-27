import type { ThreeElements } from "@react-three/fiber";
import type * as THREE from "three";

/** A standard material matching the book's own page material (color, roughness and
 * metalness), plus any further material props, for a surface that must look like the page. */
export function PageMatchedMaterial({
    pageMaterial,
    ...materialProps
}: { pageMaterial: THREE.MeshStandardMaterial } & ThreeElements["meshStandardMaterial"]) {
    return (
        <meshStandardMaterial
            color={pageMaterial.color}
            roughness={pageMaterial.roughness}
            metalness={pageMaterial.metalness}
            {...materialProps}
        />
    );
}
