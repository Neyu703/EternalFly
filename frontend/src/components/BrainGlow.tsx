import { useEffect, useMemo, useRef } from "react";
import { useFrame } from "@react-three/fiber";
import { useGLTF } from "@react-three/drei";
import * as THREE from "three";

/** Live activity per neuropil region, keyed the same as the region mesh node names (0..1 firing rate). */
export type NeuropilActivity = Record<string, number>;

// Real per-region spike rates from the simulation are small fractions of their nominal
// 0..1 range even for a genuinely very active region (measured against the real cached
// connectome, see backend/scripts/calibrate_sentiment.py) - this is the raw rate that
// counts as "fully active" for glow purposes; values beyond it just clamp at 1.0.
const NEUROPIL_ACTIVITY_CEILING = 0.08;

const IDLE_SATURATION = 0.5; // how muted a quiet region's color is, as a fraction of its true baked saturation
const IDLE_OPACITY = 0.4;
const ACTIVE_OPACITY = 0.95;
const ACTIVE_LIGHTNESS_BOOST = 0.18; // how much brighter (whiter) a fully active region's color gets, on top of full saturation

/**
 * A translucent 3D brain outline (real FlyWire FAFB template mesh) with the real, anatomically
 * colored neuropil region meshes inside it (optic lobes red/orange, central complex blue,
 * mushroom body yellow/green, ...). Each region's own base color (baked in server-side, see
 * backend/scripts/extract_brain_geometry.py) is read directly off its mesh; regions stay a
 * muted, translucent version of that hue at rest so deeper regions remain visible through the
 * ones in front, then pop to their full saturated color and opacity as they fire. Without live
 * `activity`, regions gently pulse on their own so the page still reads as "alive".
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
      child.userData.baseHsl = baseColor.getHSL({ h: 0, s: 0, l: 0 });
      // Unlit, semi-transparent material: the region's own hue drives saturation/lightness
      // directly, untouched by scene lighting, and stays translucent at rest so regions
      // nested deeper inside the brain remain visible through the ones in front of them.
      child.material = new THREE.MeshBasicMaterial({
        transparent: true,
        opacity: IDLE_OPACITY,
        depthWrite: false,
        side: THREE.DoubleSide,
        toneMapped: false,
      });
      applyRegionAppearance(child.material as THREE.MeshBasicMaterial, child.userData.baseHsl, 0);
      foundMeshes[child.name] = child;
    });
    regionMeshes.current = foundMeshes;
  }, [regionsScene]);

  useFrame(({ clock }) => {
    for (const [regionName, mesh] of Object.entries(regionMeshes.current)) {
      const liveActivity = activity?.[regionName];
      const normalizedActivity =
        liveActivity !== undefined
          ? Math.max(0, Math.min(1, liveActivity / NEUROPIL_ACTIVITY_CEILING))
          : Math.max(0, Math.sin(clock.elapsedTime * 1.5 + hashPhase(regionName))) * 0.5;
      const baseHsl = mesh.userData.baseHsl as { h: number; s: number; l: number } | undefined;
      const material = mesh.material as THREE.MeshBasicMaterial;
      if (!baseHsl || !material) continue;
      applyRegionAppearance(material, baseHsl, normalizedActivity);
    }
  });

  return (
    <group position={[-brainCenter.x, -brainCenter.y, -brainCenter.z]}>
      <primitive object={brainScene} />
      <primitive object={regionsScene} />
    </group>
  );
}

function lerp(from: number, to: number, fraction: number): number {
  return from + (to - from) * fraction;
}

/** Sets a region's material color/opacity for a given 0..1 normalizedActivity: muted,
 * translucent at rest (IDLE_SATURATION/IDLE_OPACITY) rising to the region's full true
 * hue plus a lightness pop at full activity (ACTIVE_OPACITY/ACTIVE_LIGHTNESS_BOOST). */
function applyRegionAppearance(
  material: THREE.MeshBasicMaterial,
  baseHsl: { h: number; s: number; l: number },
  normalizedActivity: number,
): void {
  const saturation = lerp(baseHsl.s * IDLE_SATURATION, baseHsl.s, normalizedActivity);
  const lightness = lerp(baseHsl.l, Math.min(1, baseHsl.l + ACTIVE_LIGHTNESS_BOOST), normalizedActivity);
  material.color.setHSL(baseHsl.h, saturation, lightness);
  material.opacity = lerp(IDLE_OPACITY, ACTIVE_OPACITY, normalizedActivity);
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
