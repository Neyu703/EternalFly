import { useEffect, useRef } from "react";
import { useFrame } from "@react-three/fiber";
import { useGLTF } from "@react-three/drei";
import type * as THREE from "three";

/**
 * "shy fly" by Maf'j Alvarez (CC-BY 3.0, see public/models/CREDITS.md), with an idle
 * wing-flutter animation applied to its wing sub-meshes.
 */
export function Fly() {
  const { scene } = useGLTF("/models/shy-fly.glb");
  const wingObjects = useRef<THREE.Object3D[]>([]);

  useEffect(() => {
    const foundWings: THREE.Object3D[] = [];
    scene.traverse((child) => {
      if (child.name.includes("BezierCurve")) foundWings.push(child);
    });
    wingObjects.current = foundWings;
  }, [scene]);

  useFrame(({ clock }) => {
    const flutterAngle = Math.sin(clock.elapsedTime * 25) * 0.5;
    wingObjects.current.forEach((wing, index) => {
      wing.rotation.z = index % 2 === 0 ? flutterAngle : -flutterAngle;
    });
  });

  return <primitive object={scene} />;
}

useGLTF.preload("/models/shy-fly.glb");
