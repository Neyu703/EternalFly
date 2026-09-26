import { Canvas } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import { BrainGlow, type NeuropilActivity } from "./BrainGlow";

/** 3D panel: the live brain-activity glow, centered in its own canvas. Orbiting is
 * limited to rotate and a bounded zoom, so the brain can't be panned or zoomed out of view. */
export function BrainScene({ activity }: { activity?: NeuropilActivity }) {
  return (
    <Canvas camera={{ position: [0, 0, 2.6], fov: 50 }}>
      <ambientLight intensity={1.2} />
      <directionalLight position={[2, 3, 2]} intensity={1.8} />
      <directionalLight position={[-2, -1, -2]} intensity={0.4} />
      <group scale={0.28}>
        <BrainGlow activity={activity} />
      </group>
      <OrbitControls enablePan={false} minDistance={1.2} maxDistance={6} />
    </Canvas>
  );
}
