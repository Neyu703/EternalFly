/** Simple two-page book prop the fly reads. Page-flip animation lands here in a later milestone. */
export function Book() {
  return (
    <group position={[0, -1.2, 0.8]} rotation={[-0.3, 0, 0]}>
      <mesh position={[-0.5, 0, 0]}>
        <boxGeometry args={[0.9, 0.05, 1.2]} />
        <meshStandardMaterial color="#f5f0e6" />
      </mesh>
      <mesh position={[0.5, 0, 0]}>
        <boxGeometry args={[0.9, 0.05, 1.2]} />
        <meshStandardMaterial color="#f5f0e6" />
      </mesh>
    </group>
  );
}
