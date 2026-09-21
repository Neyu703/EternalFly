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
const ACTIVE_LIGHTNESS_BOOST = 0.18; // how much brighter (whiter) a fully active region's color gets, on top of full saturation

// The region volume itself stays a faint, mostly see-through fill so you can look through
// it into deeper regions; the wireframe edges are the primary visual carrier of "this
// region is firing", since a solid glowing blob reads as static while a mesh of lines
// popping brighter reads as visibly reactive.
const IDLE_FILL_OPACITY = 0.1;
const ACTIVE_FILL_OPACITY = 0.5;
const IDLE_WIREFRAME_OPACITY = 0.22;
const ACTIVE_WIREFRAME_OPACITY = 1.0;

/** One neuropil region's live-updated appearance: a faint translucent fill plus a
 * wireframe outline, both driven off the same base hue and activity level. */
type RegionAppearance = {
  fillMaterial: THREE.MeshBasicMaterial;
  wireframeMaterial: THREE.LineBasicMaterial;
  baseHsl: { h: number; s: number; l: number };
};

/**
 * A translucent 3D brain outline (real FlyWire FAFB template mesh) with the real, anatomically
 * colored neuropil region meshes inside it (optic lobes red/orange, central complex blue,
 * mushroom body yellow/green, ...) rendered as a faint fill plus a wireframe mesh of edges, so
 * deeper regions stay visible through the gaps in the ones in front. Each region's own base
 * color (baked in server-side, see backend/scripts/extract_brain_geometry.py) is read directly
 * off its mesh; regions stay muted at rest and pop to their full saturated color, higher
 * opacity and brighter wireframe as they fire. Without live `activity`, regions gently pulse
 * on their own so the page still reads as "alive".
 */
export function BrainGlow({ activity }: { activity?: NeuropilActivity }) {
  const { scene: brainScene } = useGLTF("/models/brain-outline.glb");
  const { scene: regionsScene } = useGLTF("/models/neuropil-regions.glb");
  const regionAppearances = useRef<Record<string, RegionAppearance>>({});

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
    const foundAppearances: Record<string, RegionAppearance> = {};
    regionsScene.traverse((child) => {
      if (!(child instanceof THREE.Mesh)) return;
      const baseHsl = readBaseVertexColor(child).getHSL({ h: 0, s: 0, l: 0 });

      // Unlit, semi-transparent fill: the region's own hue drives saturation/lightness
      // directly, untouched by scene lighting.
      const fillMaterial = new THREE.MeshBasicMaterial({
        transparent: true,
        depthWrite: false,
        side: THREE.DoubleSide,
        toneMapped: false,
      });
      child.material = fillMaterial;

      // A visible mesh-of-edges overlay, so the region reads as a translucent wireframe
      // volume with gaps rather than a continuous solid blob.
      const wireframeMaterial = new THREE.LineBasicMaterial({ transparent: true, toneMapped: false });
      child.add(new THREE.LineSegments(new THREE.WireframeGeometry(child.geometry), wireframeMaterial));

      const appearance: RegionAppearance = { fillMaterial, wireframeMaterial, baseHsl };
      applyRegionAppearance(appearance, 0);
      foundAppearances[child.name] = appearance;
    });
    regionAppearances.current = foundAppearances;
  }, [regionsScene]);

  useFrame(({ clock }) => {
    for (const [regionName, appearance] of Object.entries(regionAppearances.current)) {
      const liveActivity = activity?.[regionName];
      const normalizedActivity =
        liveActivity !== undefined
          ? Math.max(0, Math.min(1, liveActivity / NEUROPIL_ACTIVITY_CEILING))
          : Math.max(0, Math.sin(clock.elapsedTime * 1.5 + hashPhase(regionName))) * 0.5;
      applyRegionAppearance(appearance, normalizedActivity);
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

/** Sets a region's fill/wireframe color and opacity for a given 0..1 normalizedActivity:
 * muted and faint at rest, rising to the region's full true hue plus a lightness pop,
 * higher fill opacity and a brighter wireframe at full activity. */
function applyRegionAppearance(appearance: RegionAppearance, normalizedActivity: number): void {
  const { fillMaterial, wireframeMaterial, baseHsl } = appearance;
  const saturation = lerp(baseHsl.s * IDLE_SATURATION, baseHsl.s, normalizedActivity);
  const lightness = lerp(baseHsl.l, Math.min(1, baseHsl.l + ACTIVE_LIGHTNESS_BOOST), normalizedActivity);
  fillMaterial.color.setHSL(baseHsl.h, saturation, lightness);
  fillMaterial.opacity = lerp(IDLE_FILL_OPACITY, ACTIVE_FILL_OPACITY, normalizedActivity);
  wireframeMaterial.color.copy(fillMaterial.color);
  wireframeMaterial.opacity = lerp(IDLE_WIREFRAME_OPACITY, ACTIVE_WIREFRAME_OPACITY, normalizedActivity);
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
