import { Canvas } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import { Fly } from "./Fly";
import { Book } from "./Book";

/** 3D panel: the fly perched on its book. Orbiting is limited to rotate and a bounded
 * zoom, so the model can't be panned or zoomed out of view. */
export function FlyBookScene() {
  return (
    <Canvas camera={{ position: [0, 0.71, 1.34], fov: 50 }}>
      <ambientLight intensity={1.2} />
      <directionalLight position={[2, 3, 2]} intensity={1.8} />
      <directionalLight position={[-2, -1, -2]} intensity={0.4} />
      <group position={[-0.05, -0.17, 0.3]} scale={0.032} rotation={[0.6, -Math.PI / 2, 0]}>
        <Fly />
      </group>
      <group position={[0, -0.3, 0.5]} scale={0.8}>
        <Book />
      </group>
      <OrbitControls target={[0, -0.3, 0.4]} enablePan={false} minDistance={0.8} maxDistance={3.5} />
    </Canvas>
  );
}
