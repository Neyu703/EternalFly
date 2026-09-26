import * as THREE from "three";
import { WORDS_PER_SPREAD, readingPositionOf, wordBoxAt, type WordBox } from "./readingLayout";
import type { PageSurface } from "./pageSurface";

const PREFERS_REDUCED_MOTION =
    typeof window !== "undefined" && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

// Where the fly stands relative to the word it reads (its head sits just past the word).
const HEAD_OFFSET_X = 0.02;
const STANDING_CLEARANCE = 0.002;

// Motion tuning. Springs are critically damped; frequencies in rad/s.
const WALK_SPRING_FREQUENCY = 10;
const FLIGHT_SPRING_FREQUENCY = 4;
const HOP_MIN_DISTANCE = 0.085;
const SKIM_WORDS_PER_SECOND = 14;
const SKIM_HEIGHT = 0.075;
const GAIT_CYCLES_PER_UNIT = 32;
const MAX_SPRING_STEP_SECONDS = 1 / 120;
// While a page turns, the fly hovers in front of the book, clear of the sweeping sheet.
const WATCH_POINT = new THREE.Vector3(0.04, 0.26, 0.4);

// Page turns.
const TAKEOFF_LEAD_SECONDS = 0.3;
const BASE_FLIP_SECONDS = 1.6;
const MAX_FLIP_BACKLOG = 6;

/** A page being turned: the right page of fromSpread swings over to become the left page
 * of fromSpread + 1. lead is the takeoff time before the sheet starts moving. */
export type PageFlip = { fromSpread: number; elapsed: number; lead: number; duration: number };

type Hop = { from: THREE.Vector3; to: THREE.Vector3; elapsed: number; duration: number; height: number };

const springTarget = new THREE.Vector3();
const springAcceleration = new THREE.Vector3();
const previousPosition = new THREE.Vector3();

/**
 * Motion model for the reading scene, advanced once per frame: turns the simulation's
 * word count into where the fly stands, walks, hops and flies, how its legs and wings
 * move, which spread the book shows, how far the current page turn is, and where the
 * reading highlight glides. Book and Fly only read this state; nothing here renders.
 */
export class ReadingChoreography {
    displayedSpread = 0;
    flip: PageFlip | null = null;

    readonly flyPosition = new THREE.Vector3();
    readonly flyVelocity = new THREE.Vector3();
    /** Facing in the page plane, radians from +x (the reading direction) toward +z. */
    flyHeading = 0;
    bodyPitch = 0;
    bodyRoll = 0;
    gaitPhase = 0;
    legStride = 0;
    airborneAmount = 0;
    wingEnergy = 0;
    wingPhase = 0;

    readonly highlightCenter = new THREE.Vector3();
    highlightWidth = 0;
    highlightOpacity = 0;

    private wordsRead = 0;
    private isPaused = false;
    private arousal = 0;
    private hop: Hop | null = null;
    private isFlying = false;
    private isPlaced = false;
    private wordsPerSecond = 0;
    private lastWordsRead: number | null = null;
    private lastWordChangeTime = 0;
    private elapsedTime = 0;

    /** Latest simulation input; takes effect on the next update. */
    setReadingInput(wordsRead: number, isPaused: boolean, arousal: number): void {
        this.wordsRead = wordsRead;
        this.isPaused = isPaused;
        this.arousal = Math.max(0, Math.min(1, arousal));
    }

    /** Linear 0..1 progress of the current page turn's sheet (0 while none is turning);
     * the sheet eases it per row so one corner can lead. */
    get flipProgress(): number {
        if (!this.flip) return 0;
        if (this.flip.duration === 0) return 1;
        return clamp01((this.flip.elapsed - this.flip.lead) / this.flip.duration);
    }

    /** Advances the whole scene by delta seconds. */
    update(delta: number, surface: PageSurface): void {
        this.elapsedTime += delta;
        this.trackReadingSpeed(delta);
        this.advancePageFlips(delta);

        const word = wordBoxAt(readingPositionOf(this.currentWordIndex()));
        const isTurning = this.flip !== null || this.displayedSpread < this.targetSpread();
        this.moveFly(delta, word, isTurning, surface);
        this.orientFly(delta, surface);
        this.animateLimbs(delta);
        this.moveHighlight(delta, word, isTurning, surface);
    }

    /** Index of the word being read right now (wordsRead counts it as already read). */
    private currentWordIndex(): number {
        return Math.max(0, this.wordsRead - 1);
    }

    /** The spread the current word lives on. */
    private targetSpread(): number {
        return Math.floor(this.currentWordIndex() / WORDS_PER_SPREAD);
    }

    /** Smoothed words per second, used to switch to skimming at high reading speeds. */
    private trackReadingSpeed(delta: number): void {
        if (this.lastWordsRead === null) {
            this.lastWordsRead = this.wordsRead;
            this.lastWordChangeTime = this.elapsedTime;
            return;
        }
        const wordsAdvanced = this.wordsRead - this.lastWordsRead;
        if (wordsAdvanced > 0) {
            const secondsSinceLastChange = Math.max(this.elapsedTime - this.lastWordChangeTime, 0.02);
            this.wordsPerSecond += (wordsAdvanced / secondsSinceLastChange - this.wordsPerSecond) * 0.35;
            this.lastWordChangeTime = this.elapsedTime;
        } else if (wordsAdvanced < 0) {
            this.wordsPerSecond = 0;
        } else if (this.elapsedTime - this.lastWordChangeTime > 1.5) {
            this.wordsPerSecond *= Math.exp(-delta * 2);
        }
        this.lastWordsRead = this.wordsRead;
    }

    /** Turns one page at a time toward the current word's spread, faster when behind. */
    private advancePageFlips(delta: number): void {
        const targetSpread = this.targetSpread();
        if (targetSpread < this.displayedSpread) {
            // New book or restart: jump straight back instead of turning pages backwards.
            this.displayedSpread = targetSpread;
            this.flip = null;
        } else if (targetSpread - this.displayedSpread > MAX_FLIP_BACKLOG) {
            // Reading outpaces even rushed page turns: skip ahead to keep up.
            this.displayedSpread = targetSpread - MAX_FLIP_BACKLOG;
            this.flip = null;
        }
        if (!this.flip && this.displayedSpread < targetSpread) {
            const backlog = targetSpread - this.displayedSpread;
            this.flip = {
                fromSpread: this.displayedSpread,
                elapsed: 0,
                lead: PREFERS_REDUCED_MOTION || backlog > 1 || this.isFlying ? 0 : TAKEOFF_LEAD_SECONDS,
                duration: PREFERS_REDUCED_MOTION ? 0 : BASE_FLIP_SECONDS / Math.min(backlog, 4),
            };
        }
        if (this.flip) {
            this.flip.elapsed += delta;
            if (this.flip.elapsed >= this.flip.lead + this.flip.duration) {
                this.displayedSpread = this.flip.fromSpread + 1;
                this.flip = null;
            }
        }
    }

    /** Walks, hops or flies the fly toward where it should be: on the current word, or
     * airborne while a page turns or while skimming at very high reading speeds. */
    private moveFly(delta: number, word: WordBox, isTurning: boolean, surface: PageSurface): void {
        const readingSpot = standingSpot(word.centerX - HEAD_OFFSET_X, word.centerZ, surface);
        if (!this.isPlaced) {
            this.flyPosition.copy(readingSpot);
            this.isPlaced = true;
        }

        const wantsToFly = isTurning || (!this.isPaused && this.wordsPerSecond > SKIM_WORDS_PER_SECOND);
        if (wantsToFly) {
            this.hop = null;
            this.isFlying = true;
            this.stepSpring(isTurning ? this.watchPoint() : this.skimPoint(readingSpot), FLIGHT_SPRING_FREQUENCY, delta);
            return;
        }
        if (this.isFlying) {
            this.isFlying = false;
            this.startHop(readingSpot);
        }
        if (this.hop) {
            this.advanceHop(delta);
        } else if (Math.hypot(this.flyPosition.x - readingSpot.x, this.flyPosition.z - readingSpot.z) > HOP_MIN_DISTANCE) {
            this.startHop(readingSpot);
        } else {
            this.stepSpring(readingSpot, WALK_SPRING_FREQUENCY, delta);
            // On foot the page carries the fly: stick to the surface, no vertical momentum.
            this.flyPosition.y = standingSpot(this.flyPosition.x, this.flyPosition.z, surface).y;
            this.flyVelocity.y = 0;
        }
    }

    /** Hover spot while a page turns, drifting gently so the fly never freezes mid-air. */
    private watchPoint(): THREE.Vector3 {
        const time = this.elapsedTime;
        return springTarget.set(
            WATCH_POINT.x + Math.sin(time * 0.9) * 0.03,
            WATCH_POINT.y + Math.sin(time * 1.7) * 0.015,
            WATCH_POINT.z + Math.sin(time * 1.3) * 0.02,
        );
    }

    /** Low flight spot just above the current word, for skimming at high speed. */
    private skimPoint(readingSpot: THREE.Vector3): THREE.Vector3 {
        return springTarget.set(readingSpot.x, readingSpot.y + SKIM_HEIGHT + Math.sin(this.elapsedTime * 2.3) * 0.01, readingSpot.z);
    }

    /** Starts a jump along an arc to `to` (line breaks, page changes, landings). */
    private startHop(to: THREE.Vector3): void {
        const distance = this.flyPosition.distanceTo(to);
        if (PREFERS_REDUCED_MOTION) {
            this.flyPosition.copy(to);
            this.flyVelocity.set(0, 0, 0);
            return;
        }
        this.hop = {
            from: this.flyPosition.clone(),
            to: to.clone(),
            elapsed: 0,
            duration: 0.26 + distance * 0.9,
            height: 0.03 + distance * 0.22,
        };
    }

    /** Moves along the current hop's arc and derives velocity from the movement. */
    private advanceHop(delta: number): void {
        const hop = this.hop!;
        hop.elapsed += delta;
        const progress = clamp01(hop.elapsed / hop.duration);
        previousPosition.copy(this.flyPosition);
        this.flyPosition.lerpVectors(hop.from, hop.to, easeInOutSine(progress));
        this.flyPosition.y += Math.sin(Math.PI * progress) * hop.height;
        this.flyVelocity.subVectors(this.flyPosition, previousPosition).divideScalar(Math.max(delta, 1e-4));
        if (progress >= 1) this.hop = null;
    }

    /** Critically damped spring step toward target, sub-stepped for stability. */
    private stepSpring(target: THREE.Vector3, frequency: number, delta: number): void {
        let remaining = delta;
        while (remaining > 0) {
            const step = Math.min(remaining, MAX_SPRING_STEP_SECONDS);
            springAcceleration
                .subVectors(target, this.flyPosition)
                .multiplyScalar(frequency * frequency)
                .addScaledVector(this.flyVelocity, -2 * frequency);
            this.flyVelocity.addScaledVector(springAcceleration, step);
            this.flyPosition.addScaledVector(this.flyVelocity, step);
            remaining -= step;
        }
    }

    /** Faces the reading direction while on the page and the travel direction in flight;
     * follows the page's slope on foot, noses into the motion and banks into turns aloft. */
    private orientFly(delta: number, surface: PageSurface): void {
        const isAirborne = this.isFlying || this.hop !== null;
        this.airborneAmount = damp(this.airborneAmount, isAirborne ? 1 : 0, 10, delta);

        const horizontalSpeed = Math.hypot(this.flyVelocity.x, this.flyVelocity.z);
        const desiredHeading =
            this.isFlying && horizontalSpeed > 0.12 ? Math.atan2(this.flyVelocity.z, this.flyVelocity.x) : 0;
        const previousHeading = this.flyHeading;
        this.flyHeading = dampAngle(this.flyHeading, desiredHeading, 5, delta);
        const yawRate = shortestAngle(this.flyHeading - previousHeading) / Math.max(delta, 1e-4);

        const headingX = Math.cos(this.flyHeading);
        const headingZ = Math.sin(this.flyHeading);
        const surfaceAhead = surface.heightAt(this.flyPosition.x + headingX * 0.08, this.flyPosition.z + headingZ * 0.08);
        const surfaceBehind = surface.heightAt(this.flyPosition.x - headingX * 0.08, this.flyPosition.z - headingZ * 0.08);
        const surfacePitch = Math.atan2(surfaceAhead - surfaceBehind, 0.16);
        const forwardSpeed = this.flyVelocity.x * headingX + this.flyVelocity.z * headingZ;
        const flightPitch = this.flyVelocity.y * 1.4 - forwardSpeed * 0.8;
        const targetPitch = clamp(lerp(surfacePitch, flightPitch, this.airborneAmount), -0.6, 0.6);
        this.bodyPitch = damp(this.bodyPitch, targetPitch, 8, delta);

        const bankRoll = clamp(-yawRate * 0.12, -0.5, 0.5) * this.airborneAmount;
        const walkSway = 0.04 * this.legStride * Math.sin(this.gaitPhase);
        this.bodyRoll = damp(this.bodyRoll, bankRoll + walkSway, 8, delta);
    }

    /** Leg gait from walking speed; wing energy from flight, excitement (arousal) and pause. */
    private animateLimbs(delta: number): void {
        const horizontalSpeed = Math.hypot(this.flyVelocity.x, this.flyVelocity.z);
        const isWalking = !this.isFlying && this.hop === null;
        this.legStride = damp(this.legStride, isWalking ? clamp01(horizontalSpeed / 0.06) : 0, 10, delta);
        if (isWalking) this.gaitPhase += horizontalSpeed * GAIT_CYCLES_PER_UNIT * Math.PI * 2 * delta;

        const isAirborne = this.isFlying || this.hop !== null;
        const restingEnergy = this.isPaused ? 0.02 : 0.08 + 0.45 * this.arousal;
        const targetEnergy = Math.min(isAirborne ? 1 : restingEnergy, PREFERS_REDUCED_MOTION ? 0.3 : 1);
        this.wingEnergy = damp(this.wingEnergy, targetEnergy, 6, delta);
        this.wingPhase += Math.PI * 2 * (2.5 + 15 * this.wingEnergy) * delta;
    }

    /** Glides the highlighter under the current word; fades it while a page turns and
     * jumps (instead of sweeping across the text) when the reading moves to a new line. */
    private moveHighlight(delta: number, word: WordBox, isTurning: boolean, surface: PageSurface): void {
        const targetY = surface.heightAt(word.centerX, word.centerZ);
        const jump = Math.hypot(this.highlightCenter.x - word.centerX, this.highlightCenter.z - word.centerZ);
        if (jump > 0.1 || this.highlightWidth === 0) {
            this.highlightCenter.set(word.centerX, targetY, word.centerZ);
            this.highlightWidth = word.width;
            this.highlightOpacity = 0;
        } else {
            this.highlightCenter.set(
                damp(this.highlightCenter.x, word.centerX, 14, delta),
                targetY,
                damp(this.highlightCenter.z, word.centerZ, 14, delta),
            );
            this.highlightWidth = damp(this.highlightWidth, word.width, 14, delta);
        }
        this.highlightOpacity = damp(this.highlightOpacity, isTurning ? 0 : 1, 7, delta);
    }
}

/** A point standing on the page surface at (x, z). */
function standingSpot(x: number, z: number, surface: PageSurface): THREE.Vector3 {
    return new THREE.Vector3(x, surface.heightAt(x, z) + STANDING_CLEARANCE, z);
}

/** Frame-rate independent exponential approach of current toward target. */
function damp(current: number, target: number, rate: number, delta: number): number {
    return target + (current - target) * Math.exp(-rate * delta);
}

/** Like damp, but along the shortest way around the circle. */
function dampAngle(current: number, target: number, rate: number, delta: number): number {
    return current + shortestAngle(target - current) * (1 - Math.exp(-rate * delta));
}

/** Wraps an angle difference into (-π, π]. */
function shortestAngle(angle: number): number {
    return Math.atan2(Math.sin(angle), Math.cos(angle));
}

/** Limits value to [min, max]. */
function clamp(value: number, min: number, max: number): number {
    return Math.min(max, Math.max(min, value));
}

/** Limits value to [0, 1]. */
function clamp01(value: number): number {
    return clamp(value, 0, 1);
}

/** Linear interpolation from `from` to `to`. */
function lerp(from: number, to: number, fraction: number): number {
    return from + (to - from) * fraction;
}

/** Gentle ease in and out over 0..1, used for hop arcs. */
function easeInOutSine(progress: number): number {
    return 0.5 - Math.cos(Math.PI * progress) / 2;
}

/** Stronger ease in and out over 0..1, used for page turns. */
export function easeInOutCubic(progress: number): number {
    return progress < 0.5 ? 4 * progress ** 3 : 1 - (-2 * progress + 2) ** 3 / 2;
}
