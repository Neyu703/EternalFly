import { Canvas } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import { Fly } from "./components/Fly";
import { Book } from "./components/Book";
import "./App.css";

/** Root scene: the fly and its book, lit and orbit-controllable during development. */
function App() {
  return (
    <Canvas camera={{ position: [0, 1.2, 4.5], fov: 50 }} style={{ width: "100vw", height: "100vh" }}>
      <ambientLight intensity={1.2} />
      <directionalLight position={[2, 3, 2]} intensity={1.8} />
      <directionalLight position={[-2, -1, -2]} intensity={0.4} />
      <Fly />
      <Book />
      <OrbitControls target={[0, -0.5, 0.4]} />
    </Canvas>
  );
}

export default App;
