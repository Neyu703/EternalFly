import * as THREE from "three";

/** Uniforms shared by the region surface and edge materials, so one write per frame
 * drives both draw calls. regionFiring holds each region's (activity level, flash), both
 * 0..1, as consecutive pairs. Region indexes match BrainRegionGeometry.regions; -1 = none. */
export type RegionUniforms = {
    regionFiring: { value: Float32Array };
    hoveredRegion: { value: number };
    selectedRegion: { value: number };
};

/** Fresh uniforms for regionCount regions: all idle, nothing hovered or selected. */
export function createRegionUniforms(regionCount: number): RegionUniforms {
    return {
        regionFiring: { value: new Float32Array(regionCount * 2) },
        hoveredRegion: { value: -1 },
        selectedRegion: { value: -1 },
    };
}

// Per-vertex region lookups shared by surfaces and edges. `dimmed` marks every region but
// the selected one while a selection exists.
const REGION_VERTEX_SHADER = /* glsl */ `
    attribute float regionIndex;
    uniform vec2 regionFiring[REGION_COUNT];
    uniform float hoveredRegion;
    uniform float selectedRegion;
    varying vec3 vColor;
    varying float vActivity;
    varying float vFlash;
    varying float vHover;
    varying float vDimmed;
    #ifdef USE_RIM
        varying vec3 vViewNormal;
        varying vec3 vViewDirection;
    #endif

    void main() {
        vec2 firing = regionFiring[int(regionIndex + 0.5)];
        vActivity = firing.x;
        vFlash = firing.y;
        vHover = 1.0 - step(0.5, abs(regionIndex - hoveredRegion));
        float isSelected = 1.0 - step(0.5, abs(regionIndex - selectedRegion));
        vDimmed = step(0.0, selectedRegion) * (1.0 - isSelected);
        vColor = color.rgb;
        vec4 viewPosition = modelViewMatrix * vec4(position, 1.0);
        #ifdef USE_RIM
            vViewNormal = normalize(normalMatrix * normal);
            vViewDirection = normalize(-viewPosition.xyz);
        #endif
        gl_Position = projectionMatrix * viewPosition;
    }
`;

// Resting regions show a muted version of their anatomical color; activity saturates it,
// and a flash or a hover lifts it toward white. While another region is selected, a
// region's steady glow dims hard but its flashes only partly, so firing stays visible.
const REGION_SHADING = /* glsl */ `
    vec3 regionTint() {
        float luminance = dot(vColor, vec3(0.2126, 0.7152, 0.0722));
        vec3 restingTint = mix(vec3(luminance), vColor, 0.62);
        vec3 tint = mix(restingTint, vColor, vActivity);
        return mix(tint, vec3(1.0), 0.12 * vFlash + 0.35 * vHover);
    }

    float steadyFocus() {
        return mix(1.0, 0.14, vDimmed);
    }

    float flashFocus() {
        return mix(1.0, 0.45, vDimmed);
    }
`;

const SURFACE_FRAGMENT_SHADER = /* glsl */ `
    varying vec3 vColor;
    varying float vActivity;
    varying float vFlash;
    varying float vHover;
    varying float vDimmed;
    varying vec3 vViewNormal;
    varying vec3 vViewDirection;
    ${REGION_SHADING}

    void main() {
        // Fresnel rim: faces seen edge-on glow, faces seen head-on stay see-through.
        float rim = pow(1.0 - abs(dot(normalize(vViewNormal), normalize(vViewDirection))), 2.0);
        // Dozens of shells overlap in the central brain; low per-layer intensity keeps
        // the additive sum from blowing out to white there.
        float glow = (0.003 + 0.05 * rim) * (0.25 + 0.75 * vActivity) + vHover * (0.05 + 0.3 * rim);
        float flash = vFlash * (0.01 + 0.2 * rim);
        gl_FragColor = vec4(regionTint(), glow * steadyFocus() + flash * flashFocus());
        #include <colorspace_fragment>
    }
`;

const EDGE_FRAGMENT_SHADER = /* glsl */ `
    varying vec3 vColor;
    varying float vActivity;
    varying float vFlash;
    varying float vHover;
    varying float vDimmed;
    ${REGION_SHADING}

    void main() {
        float glow = 0.01 + 0.05 * vActivity + 0.4 * vHover;
        gl_FragColor = vec4(regionTint(), glow * steadyFocus() + 0.35 * vFlash * flashFocus());
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
