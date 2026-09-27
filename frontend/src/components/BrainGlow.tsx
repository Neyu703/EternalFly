import { useEffect, useMemo, useRef } from "react";
import { useFrame, useThree, type ThreeEvent } from "@react-three/fiber";
import { useGLTF } from "@react-three/drei";
import * as THREE from "three";
import { buildBrainRegions, type BrainRegion } from "./brain/brainGeometry";
import { regionActivityLevel } from "./brain/brainRegionInfo";
import {
  createOutlineMaterial,
  createRegionEdgeMaterial,
  createRegionSurfaceMaterial,
  createRegionUniforms,
  type RegionUniforms,
} from "./brain/brainMaterials";

/** Live activity per neuropil region, keyed the same as the region mesh node names (0..1 firing rate). */
export type NeuropilActivity = Record<string, number>;

const OUTLINE_MODEL_URL = "/models/brain-outline.glb";
const REGIONS_MODEL_URL = "/models/neuropil-regions.glb";
// How fast displayed activity follows the ~20 Hz simulation ticks (1/s); smooths flicker.
const ACTIVITY_SMOOTHING_RATE = 8;
// A fully active region tries to flash this often per second, less active ones
// proportionally less.
const MAX_FLASH_ATTEMPTS_PER_SECOND = 5;
// How fast a flash fades out (1/s).
const FLASH_FADE_RATE = 10;
// Like a neuron's refractory period: a region flashes again only once its last flash has
// faded this long, so no region flashes more than 3 times a second (WCAG 2.3.1).
const FLASH_REFRACTORY_SECONDS = 0.34;
const REFRACTORY_FLASH_LEVEL = Math.exp(-FLASH_FADE_RATE * FLASH_REFRACTORY_SECONDS);
// Pointer moves shorter than this (px) between press and release count as a click, not a drag.
export const CLICK_SLOP_PX = 5;
const FOCUS_DISTANCE = 1.45;
const OVERVIEW_DISTANCE = 2.6;
const FOCUS_SECONDS = 1.1;

/** The outline model's single mesh, with normals computed for its rim shading. */
function outlineGeometryOf(outlineScene: THREE.Object3D): THREE.BufferGeometry {
  let outlineGeometry: THREE.BufferGeometry = new THREE.BufferGeometry();
  outlineScene.traverse((child) => {
    if (child instanceof THREE.Mesh) outlineGeometry = child.geometry;
  });
  if (!outlineGeometry.getAttribute("normal")) outlineGeometry.computeVertexNormals();
  return outlineGeometry;
}

/** Advances every region's displayed firing in the shared uniform. Its activity level
 * follows the live firing rate smoothly (without live data, regions pulse gently out of
 * phase so the brain still reads as alive). While isFiring, every region that fires flashes
 * at random moments, more often the more active it is, but never again within its
 * refractory time; each flash fades out quickly. */
function updateRegionFiring(
  uniforms: RegionUniforms,
  regions: BrainRegion[],
  activity: NeuropilActivity | undefined,
  isFiring: boolean,
  elapsedTime: number,
  delta: number,
): void {
  const firing = uniforms.regionFiring.value;
  const blend = 1 - Math.exp(-ACTIVITY_SMOOTHING_RATE * delta);
  const flashFade = Math.exp(-FLASH_FADE_RATE * delta);
  regions.forEach((region, index) => {
    const liveRate = activity?.[region.code];
    const targetLevel =
      liveRate !== undefined
        ? regionActivityLevel(liveRate)
        : Math.max(0, Math.sin(elapsedTime * 1.5 + index * 2.39)) * 0.5;
    firing[index * 2] += (targetLevel - firing[index * 2]) * blend;
    const flash = firing[index * 2 + 1];
    const canFlash = isFiring && liveRate !== undefined && flash <= REFRACTORY_FLASH_LEVEL;
    const flashChance = canFlash ? 1 - Math.exp(-MAX_FLASH_ATTEMPTS_PER_SECOND * targetLevel * delta) : 0;
    firing[index * 2 + 1] = Math.random() < flashChance ? 1 : flash * flashFade;
  });
}

/**
 * The real FlyWire FAFB brain (template outline plus all 78 anatomically colored neuropil
 * regions), drawn in three draw calls: a rim-lit outline shell, the merged region surfaces
 * and their merged contour edges, all additive so overlapping regions glow instead of
 * needing depth sorting. Each region's brightness and saturation follow its live firing
 * rate (log-scaled, so every firing region is lit), and while isFiring, firing regions
 * flash as often as they fire. The untouched region meshes stay hidden in the scene for
 * pointer picking: hovering reports a region, clicking selects it, and a selection dims
 * the other regions while the camera glides toward it (and back out when cleared).
 */
export function BrainGlow({
  activity,
  isFiring,
  hoveredCode,
  selectedCode,
  onHoverRegion,
  onSelectRegion,
  onRegionsReady,
}: {
  activity?: NeuropilActivity;
  isFiring: boolean;
  hoveredCode: string | null;
  selectedCode: string | null;
  onHoverRegion: (code: string | null, clientX: number, clientY: number) => void;
  onSelectRegion: (code: string) => void;
  onRegionsReady: (regions: BrainRegion[]) => void;
}) {
  const { scene: outlineScene } = useGLTF(OUTLINE_MODEL_URL);
  const { scene: regionsScene } = useGLTF(REGIONS_MODEL_URL);
  const brain = useMemo(() => buildBrainRegions(regionsScene), [regionsScene]);
  const outlineGeometry = useMemo(() => outlineGeometryOf(outlineScene), [outlineScene]);
  const regionMaterials = useMemo(() => {
    const uniforms = createRegionUniforms(brain.regions.length);
    return {
      surface: createRegionSurfaceMaterial(uniforms, brain.regions.length),
      edge: createRegionEdgeMaterial(uniforms, brain.regions.length),
    };
  }, [brain]);
  const outlineMaterial = useMemo(() => createOutlineMaterial(), []);
  const brainCenter = useMemo(
    () => new THREE.Box3().setFromObject(outlineScene).getCenter(new THREE.Vector3()),
    [outlineScene],
  );
  const regionIndexByCode = useMemo(
    () => new Map(brain.regions.map((region, index) => [region.code, index])),
    [brain],
  );
  const surfacesRef = useRef<THREE.Mesh>(null);

  useEffect(() => onRegionsReady(brain.regions), [brain, onRegionsReady]);

  useFrame(({ clock }, delta) => {
    const surfaces = surfacesRef.current;
    if (!surfaces) return;
    const uniforms = (surfaces.material as THREE.ShaderMaterial).uniforms as unknown as RegionUniforms;
    updateRegionFiring(uniforms, brain.regions, activity, isFiring, clock.elapsedTime, delta);
    uniforms.hoveredRegion.value = hoveredCode ? (regionIndexByCode.get(hoveredCode) ?? -1) : -1;
    uniforms.selectedRegion.value = selectedCode ? (regionIndexByCode.get(selectedCode) ?? -1) : -1;
  });

  /** Reports the nearest region under the pointer. Deliberately doesn't stop propagation:
   * that would make R3F send pointer-out to the regions behind it, clearing the hover. */
  function handlePointerMove(event: ThreeEvent<PointerEvent>) {
    onHoverRegion(event.intersections[0].object.name, event.nativeEvent.clientX, event.nativeEvent.clientY);
  }

  /** Selects the clicked region, unless the press was really the start of an orbit drag. */
  function handleClick(event: ThreeEvent<MouseEvent>) {
    event.stopPropagation();
    if (event.delta > CLICK_SLOP_PX) return;
    onSelectRegion(event.intersections[0].object.name);
  }

  const selectedRegion = selectedCode ? brain.regions[regionIndexByCode.get(selectedCode) ?? -1] : undefined;

  return (
    <group position={[-brainCenter.x, -brainCenter.y, -brainCenter.z]}>
      <mesh geometry={outlineGeometry} material={outlineMaterial} renderOrder={0} />
      <mesh ref={surfacesRef} geometry={brain.surfaces} material={regionMaterials.surface} renderOrder={1} />
      <lineSegments geometry={brain.edges} material={regionMaterials.edge} renderOrder={2} />
      <primitive
        object={regionsScene}
        visible={false}
        onPointerMove={handlePointerMove}
        onPointerOut={() => onHoverRegion(null, 0, 0)}
        onClick={handleClick}
      />
      <CameraFocus region={selectedRegion} />
    </group>
  );
}

/** Glides the orbit camera toward the selected region (closer, centered on it) or back
 * to the whole-brain overview, then hands control back to the user. */
function CameraFocus({ region }: { region: BrainRegion | undefined }) {
  const camera = useThree((state) => state.camera);
  const controls = useThree((state) => state.controls) as unknown as {
    target: THREE.Vector3;
    update: () => void;
  } | null;
  const remainingSecondsRef = useRef(0);
  const targetRef = useRef(new THREE.Vector3());

  useEffect(() => {
    remainingSecondsRef.current = FOCUS_SECONDS;
  }, [region]);

  useFrame((_, delta) => {
    if (!controls || remainingSecondsRef.current <= 0) return;
    remainingSecondsRef.current -= delta;
    const target = targetRef.current;
    if (region) {
      region.mesh.localToWorld(target.copy(region.center));
    } else {
      target.set(0, 0, 0);
    }
    const blend = 1 - Math.exp(-5 * delta);
    const offset = camera.position.clone().sub(controls.target);
    controls.target.lerp(target, blend);
    const distance = THREE.MathUtils.lerp(offset.length(), region ? FOCUS_DISTANCE : OVERVIEW_DISTANCE, blend);
    camera.position.copy(controls.target).add(offset.setLength(distance));
    controls.update();
  });

  return null;
}

useGLTF.preload(OUTLINE_MODEL_URL);
useGLTF.preload(REGIONS_MODEL_URL);
