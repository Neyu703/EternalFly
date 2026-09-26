import { useEffect, useMemo, useState } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { OrbitControls, useGLTF } from "@react-three/drei";
import { Fly } from "./Fly";
import { Book, BOOK_MODEL_URL } from "./Book";
import { FlyShadow } from "./reading/FlyShadow";
import { samplePageSurface } from "./reading/pageSurface";
import { ReadingChoreography } from "./reading/ReadingChoreography";

// A long stall (tab in the background, debugger) must not fling the fly across the scene.
const MAX_FRAME_DELTA_SECONDS = 0.1;

/** What the reading scene animates from: how far the book has been read, whether reading
 * is paused, and how excited the fly currently is (arousal, 0..1). */
type ReadingInput = { wordsRead: number; isPaused: boolean; arousal: number };

/** 3D panel: the fly reading its book — walking along the lines word by word, hopping to
 * the next line, and taking off while the page turns. Orbiting is limited to rotate and a
 * bounded zoom, so the scene can't be panned or zoomed out of view. */
export function FlyBookScene(readingInput: ReadingInput) {
  return (
    <Canvas camera={{ position: [0, 0.71, 1.34], fov: 50 }}>
      <ambientLight intensity={1.2} />
      <directionalLight position={[2, 3, 2]} intensity={1.8} />
      <directionalLight position={[-2, -1, -2]} intensity={0.4} />
      <group position={[0, -0.3, 0.5]} scale={0.8}>
        <ReadingStage {...readingInput} />
      </group>
      <OrbitControls target={[0, -0.3, 0.4]} enablePan={false} minDistance={0.8} maxDistance={3.5} />
    </Canvas>
  );
}

/** Loads the book, learns its page surface, and advances the shared choreography once per
 * frame (at priority -1, i.e. before Book and Fly read it in their own frame callbacks). */
function ReadingStage({ wordsRead, isPaused, arousal }: ReadingInput) {
  const { scene: bookScene } = useGLTF(BOOK_MODEL_URL);
  const surface = useMemo(() => samplePageSurface(bookScene), [bookScene]);
  const [choreography] = useState(() => new ReadingChoreography());

  useEffect(() => {
    choreography.setReadingInput(wordsRead, isPaused, arousal);
  }, [choreography, wordsRead, isPaused, arousal]);

  useFrame((_, delta) => choreography.update(Math.min(delta, MAX_FRAME_DELTA_SECONDS), surface), -1);

  return (
    <>
      <Book bookScene={bookScene} surface={surface} choreography={choreography} />
      <FlyShadow surface={surface} choreography={choreography} />
      <Fly choreography={choreography} />
    </>
  );
}
