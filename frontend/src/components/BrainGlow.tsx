import { useEffect, useMemo, useRef, useState } from "react";
import { useFrame } from "@react-three/fiber";
import { useGLTF } from "@react-three/drei";
import * as THREE from "three";

type NeuropilCentroids = Record<string, [number, number, number]>;

/** Live activity per neuropil region, keyed the same as neuropil-centroids.json (0..1 firing rate). */
export type NeuropilActivity = Record<string, number>;

const GLOW_COLOR = new THREE.Color("#7fd4ff");
const IDLE_INTENSITY = 0.15;

/**
 * A translucent 3D brain outline (real FlyWire FAFB template mesh) with a glowing sphere at
 * every neuropil's centroid. Without live `activity`, regions gently pulse on their own so the
 * visualization still reads as "alive" before the simulation is wired in.
 */
export function BrainGlow({ activity }: { activity?: NeuropilActivity }) {
  const { scene: brainScene } = useGLTF("/models/brain-outline.glb");
  const [centroids, setCentroids] = useState<NeuropilCentroids | null>(null);
  const sphereRefs = useRef<Record<string, THREE.Mesh>>({});

  useEffect(() => {
    fetch("/models/neuropil-centroids.json")
      .then((response) => response.json())
      .then(setCentroids);
  }, []);

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
      opacity: 0.15,
      depthWrite: false,
      side: THREE.DoubleSide,
    });
    brainScene.traverse((child) => {
      if (child instanceof THREE.Mesh) child.material = material;
    });
  }, [brainScene]);

  useFrame(({ clock }) => {
    if (!centroids) return;
    for (const [regionName, sphere] of Object.entries(sphereRefs.current)) {
      const liveActivity = activity?.[regionName];
      const intensity =
        liveActivity ?? IDLE_INTENSITY + Math.sin(clock.elapsedTime * 2 + hashPhase(regionName)) * 0.1;
      const material = sphere.material as THREE.MeshStandardMaterial;
      material.emissiveIntensity = Math.max(0, intensity) * 4;
      const glowScale = 1 + Math.max(0, intensity) * 1.5;
      sphere.scale.setScalar(glowScale);
    }
  });

  if (!centroids) return null;

  return (
    <group position={[-brainCenter.x, -brainCenter.y, -brainCenter.z]}>
      <primitive object={brainScene} />
      {Object.entries(centroids).map(([regionName, position]) => (
        <mesh
          key={regionName}
          position={position}
          ref={(mesh) => {
            if (mesh) sphereRefs.current[regionName] = mesh;
          }}
        >
          <sphereGeometry args={[0.08, 8, 8]} />
          <meshStandardMaterial color={GLOW_COLOR} emissive={GLOW_COLOR} emissiveIntensity={0.5} />
        </mesh>
      ))}
    </group>
  );
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
