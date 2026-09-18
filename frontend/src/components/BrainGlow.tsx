import { useEffect, useMemo, useRef } from "react";
import { useFrame } from "@react-three/fiber";
import { useGLTF } from "@react-three/drei";
import * as THREE from "three";

/** Live activity per neuropil region, keyed the same as the region mesh node names (0..1 firing rate). */
export type NeuropilActivity = Record<string, number>;

const IDLE_BRIGHTNESS = 0.35;
const ACTIVE_BRIGHTNESS = 0.9;

/**
 * A translucent 3D brain outline (real FlyWire FAFB template mesh) with the real, anatomically
 * colored neuropil region meshes inside it (optic lobes red/orange, central complex blue,
 * mushroom body yellow/green, ...). Each region's own base color (baked in server-side, see
 * backend/scripts/extract_brain_geometry.py) is read directly off its mesh and used as its glow
 * color, so quiet regions stay dim and firing regions light up brightly in their real hue.
 * Without live `activity`, regions gently pulse on their own so the page still reads as "alive".
 */
export function BrainGlow({ activity }: { activity?: NeuropilActivity }) {
  const { scene: brainScene } = useGLTF("/models/brain-outline.glb");
  const { scene: regionsScene } = useGLTF("/models/neuropil-regions.glb");
  const regionMeshes = useRef<Record<string, THREE.Mesh>>({});

  const brainCenter = useMemo(() => {
    const box = new THREE.Box3().setFromObject(brainScene);
    const center = new THREE.Vector3();
    box.getCenter(center);
    return center;
  }, [brainScene]);

  useEffect(() => {
    const material = new THREE.MeshStandardMaterial({
      color: "#3a5a7a",
      transparent: true,
      opacity: 0.12,
      depthWrite: false,
      side: THREE.DoubleSide,
    });
    brainScene.traverse((child) => {
      if (child instanceof THREE.Mesh) child.material = material;
    });
  }, [brainScene]);

  useEffect(() => {
    const foundMeshes: Record<string, THREE.Mesh> = {};
    regionsScene.traverse((child) => {
      if (!(child instanceof THREE.Mesh)) return;
      const baseColor = readBaseVertexColor(child);
      child.userData.baseColor = baseColor;
      // Unlit material: the region's own color drives brightness directly, untouched by
      // scene lighting, so quiet regions stay a dim true hue instead of being washed pale.
      child.material = new THREE.MeshBasicMaterial({
        color: baseColor.clone().multiplyScalar(IDLE_BRIGHTNESS),
        toneMapped: false,
      });
      foundMeshes[child.name] = child;
    });
    regionMeshes.current = foundMeshes;
  }, [regionsScene]);

  useFrame(({ clock }) => {
    for (const [regionName, mesh] of Object.entries(regionMeshes.current)) {
      const liveActivity = activity?.[regionName];
      const intensity =
        liveActivity ??
        IDLE_BRIGHTNESS + Math.max(0, Math.sin(clock.elapsedTime * 1.5 + hashPhase(regionName))) * 0.3;
      const baseColor = mesh.userData.baseColor as THREE.Color | undefined;
      const material = mesh.material as THREE.MeshBasicMaterial;
      if (!baseColor || !material) continue;
      const brightness = IDLE_BRIGHTNESS + Math.max(0, intensity) * ACTIVE_BRIGHTNESS;
      material.color.copy(baseColor).multiplyScalar(brightness);
    }
  });

  return (
    <group position={[-brainCenter.x, -brainCenter.y, -brainCenter.z]}>
      <primitive object={brainScene} />
      <primitive object={regionsScene} />
    </group>
  );
}

/** Reads a mesh's uniform per-vertex color (baked server-side, same value on every vertex). */
function readBaseVertexColor(mesh: THREE.Mesh): THREE.Color {
  const colorAttribute = mesh.geometry.getAttribute("color");
  if (!colorAttribute) return new THREE.Color("#9a86be");
  return new THREE.Color(colorAttribute.getX(0), colorAttribute.getY(0), colorAttribute.getZ(0));
}

/** Deterministic 0..2π phase per region name, so idle pulsing isn't perfectly synchronized. */
function hashPhase(regionName: string): number {
  let hash = 0;
  for (let index = 0; index < regionName.length; index += 1) {
    hash = (hash * 31 + regionName.charCodeAt(index)) % 1000;
  }
  return (hash / 1000) * Math.PI * 2;
}

useGLTF.preload("/models/brain-outline.glb");
useGLTF.preload("/models/neuropil-regions.glb");
