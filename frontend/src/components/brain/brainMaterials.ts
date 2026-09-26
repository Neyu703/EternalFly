import * as THREE from "three";

/** Uniforms shared by the region surface and edge materials, so one write per frame
 * drives both draw calls. Region indexes match BrainRegionGeometry.regions; -1 = none. */
export type RegionUniforms = {
    regionActivity: { value: Float32Array };
    hoveredRegion: { value: number };
    selectedRegion: { value: number };
};

/** Fresh uniforms for regionCount regions: all idle, nothing hovered or selected. */
export function createRegionUniforms(regionCount: number): RegionUniforms {
    return {
        regionActivity: { value: new Float32Array(regionCount) },
        hoveredRegion: { value: -1 },
        selectedRegion: { value: -1 },
    };
}

// Per-vertex region lookups shared by surfaces and edges. `focus` dims every region but
// the selected one while a selection exists.
const REGION_VERTEX_SHADER = /* glsl */ `
    attribute float regionIndex;
    uniform float regionActivity[REGION_COUNT];
    uniform float hoveredRegion;
    uniform float selectedRegion;
    varying vec3 vColor;
    varying float vActivity;
    varying float vHover;
    varying float vFocus;
    #ifdef USE_RIM
        varying vec3 vViewNormal;
        varying vec3 vViewDirection;
    #endif

    void main() {
        vActivity = regionActivity[int(regionIndex + 0.5)];
        vHover = 1.0 - step(0.5, abs(regionIndex - hoveredRegion));
        float isSelected = 1.0 - step(0.5, abs(regionIndex - selectedRegion));
        vFocus = mix(1.0, mix(0.14, 1.0, isSelected), step(0.0, selectedRegion));
        vColor = color.rgb;
        vec4 viewPosition = modelViewMatrix * vec4(position, 1.0);
        #ifdef USE_RIM
            vViewNormal = normalize(normalMatrix * normal);
            vViewDirection = normalize(-viewPosition.xyz);
        #endif
        gl_Position = projectionMatrix * viewPosition;
    }
`;

// Resting regions show a muted version of their anatomical color; activity saturates and
// brightens them, and a hover lifts them toward white.
const REGION_TINT = /* glsl */ `
    vec3 regionTint() {
        float luminance = dot(vColor, vec3(0.2126, 0.7152, 0.0722));
        vec3 restingTint = mix(vec3(luminance), vColor, 0.62);
        vec3 tint = mix(restingTint, vColor, vActivity);
        return mix(tint, vec3(1.0), 0.12 * vActivity * vActivity + 0.35 * vHover);
    }
`;

const SURFACE_FRAGMENT_SHADER = /* glsl */ `
    varying vec3 vColor;
    varying float vActivity;
    varying float vHover;
    varying float vFocus;
    varying vec3 vViewNormal;
    varying vec3 vViewDirection;
    ${REGION_TINT}

    void main() {
        // Fresnel rim: faces seen edge-on glow, faces seen head-on stay see-through.
        float rim = pow(1.0 - abs(dot(normalize(vViewNormal), normalize(vViewDirection))), 2.0);
        float energy = 0.3 + 0.7 * vActivity;
        // Dozens of shells overlap in the central brain; low per-layer intensity keeps
        // the additive sum from blowing out to white there.
        float alpha = ((0.006 + 0.13 * rim) * energy + vHover * (0.05 + 0.3 * rim)) * vFocus;
        gl_FragColor = vec4(regionTint(), alpha);
        #include <colorspace_fragment>
    }
`;

const EDGE_FRAGMENT_SHADER = /* glsl */ `
    varying vec3 vColor;
    varying float vActivity;
    varying float vHover;
    varying float vFocus;
    ${REGION_TINT}

    void main() {
        float alpha = (0.03 + 0.24 * vActivity + 0.4 * vHover) * vFocus;
        gl_FragColor = vec4(regionTint(), alpha);
        #include <colorspace_fragment>
    }
`;

const OUTLINE_VERTEX_SHADER = /* glsl */ `
    varying vec3 vViewNormal;
    varying vec3 vViewDirection;

    void main() {
        vec4 viewPosition = modelViewMatrix * vec4(position, 1.0);
        vViewNormal = normalize(normalMatrix * normal);
        vViewDirection = normalize(-viewPosition.xyz);
        gl_Position = projectionMatrix * viewPosition;
    }
`;

const OUTLINE_FRAGMENT_SHADER = /* glsl */ `
    uniform vec3 outlineColor;
    varying vec3 vViewNormal;
    varying vec3 vViewDirection;

    void main() {
        float rim = pow(1.0 - abs(dot(normalize(vViewNormal), normalize(vViewDirection))), 3.0);
        gl_FragColor = vec4(outlineColor, 0.012 + 0.3 * rim);
        #include <colorspace_fragment>
    }
`;

// Additive glow needs no depth sorting. Shells render front faces only: the far side's
// rim adds little but doubles the overlap (and fragment cost) in the dense center.
const GLOW_SETTINGS = {
    transparent: true,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
} as const;

/** Rim-lit, activity-driven surfaces of all regions (one draw call). */
export function createRegionSurfaceMaterial(uniforms: RegionUniforms, regionCount: number): THREE.ShaderMaterial {
    return new THREE.ShaderMaterial({
        ...GLOW_SETTINGS,
        uniforms,
        defines: { REGION_COUNT: regionCount, USE_RIM: "" },
        vertexShader: REGION_VERTEX_SHADER,
        fragmentShader: SURFACE_FRAGMENT_SHADER,
        vertexColors: true,
    });
}

/** Contour lines of all regions, brightening with activity (one draw call). */
export function createRegionEdgeMaterial(uniforms: RegionUniforms, regionCount: number): THREE.ShaderMaterial {
    return new THREE.ShaderMaterial({
        ...GLOW_SETTINGS,
        uniforms,
        defines: { REGION_COUNT: regionCount },
        vertexShader: REGION_VERTEX_SHADER,
        fragmentShader: EDGE_FRAGMENT_SHADER,
        vertexColors: true,
    });
}

/** The whole brain's shell: nearly invisible face-on, a soft glowing silhouette edge-on. */
export function createOutlineMaterial(): THREE.ShaderMaterial {
    return new THREE.ShaderMaterial({
        ...GLOW_SETTINGS,
        uniforms: { outlineColor: { value: new THREE.Color("#8fb3ff") } },
        vertexShader: OUTLINE_VERTEX_SHADER,
        fragmentShader: OUTLINE_FRAGMENT_SHADER,
    });
}
