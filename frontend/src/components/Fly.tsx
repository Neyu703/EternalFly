import { useRef } from "react";
import { useFrame } from "@react-three/fiber";
import * as THREE from "three";

/** Procedurally built low-poly fly: capsule body segments, flat wings, idle wing-flutter animation. */
export function Fly() {
  const leftWingRef = useRef<THREE.Mesh>(null);
  const rightWingRef = useRef<THREE.Mesh>(null);

  useFrame(({ clock }) => {
    const flutterAngle = Math.sin(clock.elapsedTime * 25) * 0.6;
    if (leftWingRef.current) leftWingRef.current.rotation.z = flutterAngle;
    if (rightWingRef.current) rightWingRef.current.rotation.z = -flutterAngle;
  });

  return (
    <group>
      <mesh position={[0, 0, 0]}>
        <capsuleGeometry args={[0.3, 0.4, 8, 16]} />
        <meshStandardMaterial color="#4a4a5a" />
      </mesh>
      <mesh position={[0, -0.6, 0]}>
        <capsuleGeometry args={[0.25, 0.5, 8, 16]} />
        <meshStandardMaterial color="#33333f" />
      </mesh>
      <mesh position={[0, 0.5, 0]}>
        <sphereGeometry args={[0.22, 16, 16]} />
        <meshStandardMaterial color="#26262f" />
      </mesh>
      <mesh ref={leftWingRef} position={[-0.3, 0.1, 0]}>
        <planeGeometry args={[0.6, 0.25]} />
        <meshStandardMaterial color="#cbd8e6" transparent opacity={0.4} side={THREE.DoubleSide} />
      </mesh>
      <mesh ref={rightWingRef} position={[0.3, 0.1, 0]}>
        <planeGeometry args={[0.6, 0.25]} />
        <meshStandardMaterial color="#cbd8e6" transparent opacity={0.4} side={THREE.DoubleSide} />
      </mesh>
    </group>
  );
}
