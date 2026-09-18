import { useGLTF } from "@react-three/drei";

/** "Open Book" by Quaternius (CC0, see public/models/CREDITS.md). Page-flip animation lands in a later milestone. */
export function Book() {
  const { scene } = useGLTF("/models/open-book.glb");
  return <primitive object={scene} />;
}

useGLTF.preload("/models/open-book.glb");
