import { useEffect, useMemo, useRef, useState } from "react";
import { useFrame } from "@react-three/fiber";
import * as THREE from "three";
import type { RefObject } from "react";
import type { FiredNeuronIndices } from "../hooks/useWebSocketTickData";

// How long a spiked neuron's point stays lit before fully fading, in seconds - a
// display choice, not a biological time constant.
const SPIKE_DECAY_SECONDS = 0.25;
const POINT_SIZE = 2.5; // gl_PointSize scale factor, see the vertex shader below
const SPIKE_COLOR = new THREE.Color("#7fd4ff");

// The vertex shader's clamp() upper bound on gl_PointSize, in screen pixels - reproduced
// and confirmed by bisection: with 139k points, an unclamped (or too generously clamped,
// e.g. 48px) perspective-scaled point size lets every point near the camera balloon to a
// huge on-screen footprint at once, and rasterizing that many huge overlapping additive-
// blended discs hangs the page outright under software rendering (no real GPU - see
// this file's own environment). A small cap keeps the total fill rate bounded regardless
// of camera distance; 10px still reads clearly as "a swarm of glowing dots".
const MAX_POINT_SIZE_PIXELS = 10.0;

// Re-uploading the whole ~557KB lastSpikeTime buffer to the GPU (attribute.needsUpdate)
// on every new binary spike message - potentially every animation frame - stalls the
// renderer well before SPIKE_DECAY_SECONDS would ever notice the difference. The CPU-
// side write still happens on every message (see useFrame below), only the actual GPU
// upload is throttled to this interval, so fast-arriving messages coalesce into one
// upload instead of one each.
const MIN_UPLOAD_INTERVAL_SECONDS = 0.1;

const VERTEX_SHADER = /* glsl */ `
  attribute float lastSpikeTime;
  uniform float uTime;
  uniform float uDecaySeconds;
  uniform float uPointSize;
  uniform float uMaxPointSize;
  varying float vIntensity;

  void main() {
    float age = uTime - lastSpikeTime;
    vIntensity = clamp(1.0 - age / uDecaySeconds, 0.0, 1.0);
    vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
    // max(0.01, ...) also guards the handful of neurons with no real position (see
    // geometry.py's neuron_scene_positions - unmatched root_ids default to the scene
    // origin) landing at or behind the camera, where an unguarded 1/-z blows up.
    gl_PointSize = clamp(uPointSize * (300.0 / max(0.01, -mvPosition.z)), 0.0, uMaxPointSize);
    gl_Position = projectionMatrix * mvPosition;
  }
`;

const FRAGMENT_SHADER = /* glsl */ `
  uniform vec3 uColor;
  varying float vIntensity;

  void main() {
    if (vIntensity <= 0.0) discard;
    float distanceFromCenter = length(gl_PointCoord - vec2(0.5));
    float alpha = smoothstep(0.5, 0.0, distanceFromCenter) * vIntensity;
    gl_FragColor = vec4(uColor, alpha);
  }
`;

/**
 * A 139k-point cloud, one point per real FlyWire neuron - positions baked server-side
 * into public/models/neuron-positions.bin (same coordinate frame as BrainGlow's
 * meshes, see backend/eternalfly/geometry.py), rendered inside BrainGlow's own group so
 * it lines up with the brain outline/region meshes without any extra transform.
 * Neurons that fired in the most recently streamed simulation frame (delivered as a
 * binary WebSocket message right after each JSON frame - see server.py's
 * encode_fired_neuron_indices and useWebSocketTickData's firedNeuronIndicesRef) glow and
 * fade out over SPIKE_DECAY_SECONDS. The "last spike time" lives in a per-vertex buffer
 * attribute read by the shader, not React state - up to 20k indices can arrive every
 * simulation frame, far too often to re-render for.
 */
export function SpikeCloud({ firedNeuronIndicesRef }: { firedNeuronIndicesRef: RefObject<FiredNeuronIndices> }) {
  const [neuronPositions, setNeuronPositions] = useState<Float32Array | null>(null);
  const materialRef = useRef<THREE.ShaderMaterial>(null);
  const lastSeenVersionRef = useRef(0);
  const lastUploadTimeRef = useRef(-Infinity);
  const pendingUploadRef = useRef(false);

  useEffect(() => {
    let cancelled = false;
    fetch("/models/neuron-positions.bin")
      .then((response) => response.arrayBuffer())
      .then((buffer) => {
        if (!cancelled) setNeuronPositions(new Float32Array(buffer));
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const geometry = useMemo(() => {
    if (!neuronPositions) return null;
    const neuronCount = neuronPositions.length / 3;
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(neuronPositions, 3));
    const lastSpikeTimeAttribute = new THREE.BufferAttribute(new Float32Array(neuronCount).fill(-1000), 1);
    lastSpikeTimeAttribute.setUsage(THREE.DynamicDrawUsage);
    geo.setAttribute("lastSpikeTime", lastSpikeTimeAttribute);
    return geo;
  }, [neuronPositions]);

  const uniforms = useMemo(
    () => ({
      uTime: { value: 0 },
      uDecaySeconds: { value: SPIKE_DECAY_SECONDS },
      uPointSize: { value: POINT_SIZE },
      uMaxPointSize: { value: MAX_POINT_SIZE_PIXELS },
      uColor: { value: SPIKE_COLOR },
    }),
    [],
  );

  useFrame(({ clock }) => {
    if (materialRef.current) materialRef.current.uniforms.uTime.value = clock.elapsedTime;

    const attribute = geometry?.getAttribute("lastSpikeTime") as THREE.BufferAttribute | undefined;
    if (!attribute) return;

    const latest = firedNeuronIndicesRef.current;
    if (latest.version !== lastSeenVersionRef.current) {
      lastSeenVersionRef.current = latest.version;
      const lastSpikeTime = attribute.array as Float32Array;
      for (const neuronIndex of latest.indices) {
        if (neuronIndex < lastSpikeTime.length) lastSpikeTime[neuronIndex] = clock.elapsedTime;
      }
      pendingUploadRef.current = true;
    }

    if (pendingUploadRef.current && clock.elapsedTime - lastUploadTimeRef.current >= MIN_UPLOAD_INTERVAL_SECONDS) {
      attribute.needsUpdate = true;
      lastUploadTimeRef.current = clock.elapsedTime;
      pendingUploadRef.current = false;
    }
  });

  if (!geometry) return null;

  return (
    <points geometry={geometry}>
      <shaderMaterial
        ref={materialRef}
        vertexShader={VERTEX_SHADER}
        fragmentShader={FRAGMENT_SHADER}
        uniforms={uniforms}
        transparent
        depthWrite={false}
        blending={THREE.AdditiveBlending}
      />
    </points>
  );
}
