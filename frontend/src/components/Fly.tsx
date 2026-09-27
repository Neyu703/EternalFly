import { useEffect, useRef } from "react";
import { useFrame } from "@react-three/fiber";
import { useGLTF } from "@react-three/drei";
import * as THREE from "three";
import type { ReadingChoreography } from "./reading/ReadingChoreography";
import { meshesOf } from "../utils/scene";

const FLY_MODEL_URL = "/models/shy-fly.glb";
const FLY_SCALE = 0.017;

// Leg nodes (glTF node names with the loader's dots stripped), grouped into the two
// alternating tripods of an insect gait: front and rear leg of one side plus the middle
// leg of the other side lift together.
const LEG_NODES = [
  { name: "Sphere007", tripodPhase: 0 }, // left front
  { name: "Sphere003", tripodPhase: 0 }, // right middle
  { name: "Sphere005", tripodPhase: 0 }, // left rear
  { name: "Sphere000", tripodPhase: Math.PI }, // right front
  { name: "Sphere006", tripodPhase: Math.PI }, // left middle
  { name: "Sphere004", tripodPhase: Math.PI }, // right rear
];
const WING_NODE_NAMES = ["BezierCurve001_Mesh001", "BezierCurve_Mesh"];

const LEG_SWING = 0.35;
const LEG_LIFT = 0.3;
const LEG_TUCK = 0.45;
const WING_REST_FOLD = 1.1;
const WING_FLAP_MEAN = 0.35;
const WING_FLAP_AMPLITUDE = 0.75;
const WALK_BOB = 0.0025;
const BREATHING_BOB = 0.001;
// Breathing rate of that resting bob (rad/s).
const BREATHING_RATE = 2.4;

/** A leg or wing: its node (re-pivoted onto its joint), which side of the body it's on
 * (-1 left, +1 right) and, for legs, its tripod's phase offset. */
type Limb = { node: THREE.Object3D; side: number; tripodPhase: number };

type FlyRig = { legs: Limb[]; wings: Limb[] };

/** All vertex positions of a node's meshes in the model's frame (the model's nodes carry
 * no transforms of their own, so raw geometry coordinates are model coordinates). */
function modelVertices(node: THREE.Object3D): THREE.Vector3[] {
  return meshesOf(node).flatMap((mesh) => {
    const positions = mesh.geometry.getAttribute("position");
    return Array.from({ length: positions.count }, (_, index) => new THREE.Vector3().fromBufferAttribute(positions, index));
  });
}

/** Centroid of the vertices scoring in the top `fraction` of `score`'s range. */
function extremeCentroid(vertices: THREE.Vector3[], score: (vertex: THREE.Vector3) => number, fraction = 0.15): THREE.Vector3 {
  const scores = vertices.map(score);
  const maxScore = Math.max(...scores);
  const threshold = maxScore - (maxScore - Math.min(...scores)) * fraction;
  const selected = vertices.filter((_, index) => scores[index] >= threshold);
  return selected.reduce((sum, vertex) => sum.add(vertex), new THREE.Vector3()).divideScalar(selected.length);
}

/** Moves a node's origin onto `pivot` without moving its meshes, so rotating the node turns
 * the limb around that joint. Idempotent, as the loader reuses the cached scene on remount. */
function setPivot(node: THREE.Object3D, pivot: THREE.Vector3): void {
  node.position.copy(pivot);
  node.children.forEach((child) => child.position.copy(pivot).negate());
}

/** Finds the fly's legs and wings and re-pivots each onto its hip or wing root. */
function rigFly(flyScene: THREE.Object3D): FlyRig {
  const legs = LEG_NODES.flatMap(({ name, tripodPhase }) => {
    const node = flyScene.getObjectByName(name);
    if (!node) return [];
    const hip = extremeCentroid(modelVertices(node), (vertex) => vertex.y);
    setPivot(node, hip);
    return [{ node, side: Math.sign(hip.x), tripodPhase }];
  });
  const wings = WING_NODE_NAMES.flatMap((name) => {
    const node = flyScene.getObjectByName(name);
    if (!node) return [];
    const vertices = modelVertices(node);
    const side = Math.sign(vertices.reduce((sum, vertex) => sum + vertex.x, 0));
    const root = extremeCentroid(vertices, (vertex) => -side * vertex.x);
    setPivot(node, root);
    return [{ node, side, tripodPhase: 0 }];
  });
  return { legs, wings };
}

/** Model yaw that turns the fly's head (model -z) toward `heading` (radians from +x toward +z). */
function headingToYaw(heading: number): number {
  return Math.atan2(-Math.cos(heading), -Math.sin(heading));
}

/** "shy fly" by Maf'j Alvarez (CC-BY 3.0, see public/models/CREDITS.md), posed every frame
 * from the reading choreography: body placement and attitude, a tripod walking gait while
 * it moves along the lines, legs tucked in flight, and wings folded at rest, fluttering
 * with excitement and beating fast while airborne. */
export function Fly({ choreography }: { choreography: ReadingChoreography }) {
  const { scene } = useGLTF(FLY_MODEL_URL);
  const rigRef = useRef<FlyRig | null>(null);
  const flyRef = useRef<THREE.Group>(null);

  useEffect(() => {
    rigRef.current = rigFly(scene);
  }, [scene]);

  useFrame(({ clock }) => {
    const fly = flyRef.current;
    if (!fly || !rigRef.current) return;
    const { flyPosition, legStride, gaitPhase, airborneAmount, wingEnergy, wingPhase } = choreography;

    const bob =
      WALK_BOB * legStride * Math.abs(Math.sin(gaitPhase)) +
      BREATHING_BOB * (1 - airborneAmount) * Math.sin(clock.elapsedTime * BREATHING_RATE);
    fly.position.set(flyPosition.x, flyPosition.y + bob, flyPosition.z);
    fly.rotation.set(choreography.bodyPitch, headingToYaw(choreography.flyHeading), choreography.bodyRoll, "YXZ");

    rigRef.current.legs.forEach(({ node, side, tripodPhase }) => {
      const phase = gaitPhase + tripodPhase;
      node.rotation.y = side * LEG_SWING * legStride * Math.sin(phase);
      node.rotation.z = side * (LEG_LIFT * legStride * Math.max(0, Math.cos(phase)) + LEG_TUCK * airborneAmount);
    });

    const flap = wingEnergy * (WING_FLAP_MEAN + WING_FLAP_AMPLITUDE * Math.sin(wingPhase));
    rigRef.current.wings.forEach(({ node, side }) => {
      node.rotation.y = -side * WING_REST_FOLD * (1 - wingEnergy);
      node.rotation.z = side * flap;
    });
  });

  return (
    <group ref={flyRef} scale={FLY_SCALE}>
      <primitive object={scene} />
    </group>
  );
}

useGLTF.preload(FLY_MODEL_URL);
