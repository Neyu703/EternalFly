/** Frame-rate independent exponential approach of current toward target. */
export function damp(current: number, target: number, rate: number, delta: number): number {
    return target + (current - target) * Math.exp(-rate * delta);
}

/** Like damp, but along the shortest way around the circle. */
export function dampAngle(current: number, target: number, rate: number, delta: number): number {
    return current + shortestAngle(target - current) * (1 - Math.exp(-rate * delta));
}

/** Wraps an angle difference into (-π, π]. */
export function shortestAngle(angle: number): number {
    return Math.atan2(Math.sin(angle), Math.cos(angle));
}

/** Limits value to [min, max]. */
export function clamp(value: number, min: number, max: number): number {
    return Math.min(max, Math.max(min, value));
}

/** Limits value to [0, 1]. */
export function clamp01(value: number): number {
    return clamp(value, 0, 1);
}

/** Linear interpolation from `from` to `to`. */
export function lerp(from: number, to: number, fraction: number): number {
    return from + (to - from) * fraction;
}

/** 0 below edgeStart, 1 above edgeEnd, a smooth S-curve in between. */
export function smoothstep(edgeStart: number, edgeEnd: number, value: number): number {
    const fraction = clamp01((value - edgeStart) / (edgeEnd - edgeStart));
    return fraction * fraction * (3 - 2 * fraction);
}

/** Gentle ease in and out over 0..1, used for hop arcs. */
export function easeInOutSine(progress: number): number {
    return 0.5 - Math.cos(Math.PI * progress) / 2;
}

/** Stronger ease in and out over 0..1, used for page turns. */
export function easeInOutCubic(progress: number): number {
    return progress < 0.5 ? 4 * progress ** 3 : 1 - (-2 * progress + 2) ** 3 / 2;
}
