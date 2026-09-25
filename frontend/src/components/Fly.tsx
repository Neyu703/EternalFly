import { useRef, useEffect } from "react";
import { useFrame } from "@react-three/fiber";
import { useGLTF } from "@react-three/drei";
import * as THREE from "three";

// All position/size constants below are in the raw shy-fly.glb model's own local units
// (not the small 0.032-scale world units FlyBookScene renders it at - see its wrapping
// <group scale={0.032}>) since Fly's own returned group sits directly inside that
// wrapper with no scale of its own, so anything animated here (bodyGroupRef, the
// proboscis) shares the model's native ~1-10-unit scale.

const IDLE_FLUTTER_HZ = 25;
const IDLE_FLUTTER_AMPLITUDE = 0.08;
// Real octopaminergic arousal readout (see region_activity.arousal / brain_loader.py's
// AROUSAL_CEILING) speeds up and widens the wingbeat, the way a genuinely alert fly's
// wings blur faster than a resting one's.
const AROUSED_FLUTTER_HZ = 55;
const AROUSED_FLUTTER_AMPLITUDE = 0.22;

// Giant Fiber (behaviors.escape) triggers a single startled hop: a brief upward arc
// with a forward tilt and the wings snapping open, re-triggerable once it's finished
// and the escape readout has stayed low then risen again past the threshold - a
// continuous firing rate has no "edge" of its own, so this is what turns a level into
// a one-shot event on screen.
const ESCAPE_JUMP_TRIGGER_THRESHOLD = 0.5;
const ESCAPE_JUMP_DURATION_SECONDS = 0.6;
const ESCAPE_JUMP_COOLDOWN_SECONDS = 0.6;
const ESCAPE_JUMP_HEIGHT = 1.4;
const ESCAPE_JUMP_TILT_RADIANS = 0.4;
const ESCAPE_JUMP_WING_SPLAY_RADIANS = 0.6;

// MDN backing and DNa02 turn_left/turn_right are continuous readouts, not one-shot
// events - the fly shuffles back and steers proportionally to how hard they're firing.
const BACKING_MAX_DISTANCE = 1.5;
const TURN_MAX_RADIANS_PER_SECOND = 2.5;

// The model's own head tip, in its local space - found by measuring each mesh's local
// bounding box at runtime (see this file's git history for the one-off debug pass) and
// picking Sphere.001's frontmost sub-mesh, the smallest blob at the body's most
// negative-Z (frontmost) extreme. The proboscis extends further out along the same
// -Z "forward" axis, its length driven by the real feeding motor pool's readout.
const HEAD_TIP_LOCAL = new THREE.Vector3(0, 2.0, -6.9);
const PROBOSCIS_MAX_LENGTH = 3.0;
const PROBOSCIS_RADIUS = 0.18;

/** Real descending/motor readouts driving the fly's visible reactions - see
 * reading_session.py's BEHAVIOR_NAMES, already normalized 0..1 there. */
export type FlyBehaviors = {
  escape: number;
  feeding: number;
  backing: number;
  turn_left: number;
  turn_right: number;
};

/**
 * "shy fly" by Maf'j Alvarez (CC-BY 3.0, see public/models/CREDITS.md). Idle wing
 * flutter always runs; when real behaviors/arousal are supplied (the real fly's own
 * descending/motor neuron readouts, see reading_session.py), the fly visibly reacts to
 * what just fired in its simulated brain instead of just idling: wingbeat speeds up
 * with arousal, a Giant Fiber escape spike triggers a startled hop, MDN backing shuffles
 * it back, DNa02 left/right steering turns it, and the real feeding motor pool extends
 * a procedural proboscis.
 */
export function Fly({ behaviors, arousal = 0 }: { behaviors?: FlyBehaviors; arousal?: number }) {
  const { scene } = useGLTF("/models/shy-fly.glb");
  const wingObjects = useRef<THREE.Object3D[]>([]);
  const bodyGroupRef = useRef<THREE.Group>(null);
  const jumpStartTimeRef = useRef<number | null>(null);
  const headingRef = useRef(0);

  useEffect(() => {
    const foundWings: THREE.Object3D[] = [];
    scene.traverse((child) => {
      if (child.name.includes("BezierCurve")) foundWings.push(child);
    });
    wingObjects.current = foundWings;
  }, [scene]);

  useFrame(({ clock }, deltaSeconds) => {
    const clampedArousal = Math.max(0, Math.min(1, arousal));
    const flutterHz = IDLE_FLUTTER_HZ + (AROUSED_FLUTTER_HZ - IDLE_FLUTTER_HZ) * clampedArousal;
    const flutterAmplitude = IDLE_FLUTTER_AMPLITUDE + (AROUSED_FLUTTER_AMPLITUDE - IDLE_FLUTTER_AMPLITUDE) * clampedArousal;
    const flutterAngle = Math.sin(clock.elapsedTime * flutterHz) * flutterAmplitude;

    const bodyGroup = bodyGroupRef.current;
    let wingSplayAngle = 0;

    if (bodyGroup && behaviors) {
      const previousJumpStart = jumpStartTimeRef.current;
      const sinceLastJump = previousJumpStart === null ? Infinity : clock.elapsedTime - previousJumpStart;
      const canRetrigger = sinceLastJump > ESCAPE_JUMP_DURATION_SECONDS + ESCAPE_JUMP_COOLDOWN_SECONDS;
      if (behaviors.escape > ESCAPE_JUMP_TRIGGER_THRESHOLD && canRetrigger) {
        jumpStartTimeRef.current = clock.elapsedTime;
      }
      const jumpStart = jumpStartTimeRef.current;
      const jumpElapsed = jumpStart === null ? Infinity : clock.elapsedTime - jumpStart;
      const jumpProgress = Math.min(1, jumpElapsed / ESCAPE_JUMP_DURATION_SECONDS);
      const jumpArc = jumpElapsed < ESCAPE_JUMP_DURATION_SECONDS ? Math.sin(jumpProgress * Math.PI) : 0;

      headingRef.current += (behaviors.turn_right - behaviors.turn_left) * TURN_MAX_RADIANS_PER_SECOND * deltaSeconds;
      const backOffset = behaviors.backing * BACKING_MAX_DISTANCE;

      bodyGroup.rotation.y = headingRef.current;
      bodyGroup.position.set(
        -Math.sin(headingRef.current) * backOffset,
        jumpArc * ESCAPE_JUMP_HEIGHT,
        -Math.cos(headingRef.current) * backOffset,
      );
      bodyGroup.rotation.x = jumpArc * ESCAPE_JUMP_TILT_RADIANS;
      wingSplayAngle = jumpArc * ESCAPE_JUMP_WING_SPLAY_RADIANS;
    }

    wingObjects.current.forEach((wing, index) => {
      wing.rotation.z = index % 2 === 0 ? flutterAngle : -flutterAngle;
      wing.rotation.x = wingSplayAngle;
    });
  });

  return (
    <group ref={bodyGroupRef}>
      <primitive object={scene} />
      {behaviors && <Proboscis feedingLevel={behaviors.feeding} />}
    </group>
  );
}

/** Procedural feeding response: a cylinder extending from the model's own head tip
 * along its forward axis, length proportional to the real feeding motor pool's
 * readout - no length at all (fully retracted) when feeding is 0. */
function Proboscis({ feedingLevel }: { feedingLevel: number }) {
  const length = Math.max(0.0001, feedingLevel) * PROBOSCIS_MAX_LENGTH;
  return (
    <mesh
      position={[HEAD_TIP_LOCAL.x, HEAD_TIP_LOCAL.y, HEAD_TIP_LOCAL.z - length / 2]}
      rotation={[Math.PI / 2, 0, 0]}
      visible={feedingLevel > 0.001}
    >
      <cylinderGeometry args={[PROBOSCIS_RADIUS, PROBOSCIS_RADIUS, length, 8]} />
      <meshStandardMaterial color="#5a4632" />
    </mesh>
  );
}

useGLTF.preload("/models/shy-fly.glb");
