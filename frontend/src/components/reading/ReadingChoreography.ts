import * as THREE from "three";
import {
    TEXT_INNER_X,
    TEXT_OUTER_X,
    WORDS_PER_LINE,
    WORDS_PER_SPREAD,
    lineBoxAtIndex,
    lineEndIndex,
    wordBoxAtIndex,
} from "./readingLayout";
import type { PageSurface } from "./pageSurface";
import { PREFERS_REDUCED_MOTION } from "../../utils/motion";

// Where the fly stands relative to the word it reads (its head sits just past the word).
const HEAD_OFFSET_X = 0.02;
const STANDING_CLEARANCE = 0.002;
// Average distance between neighbouring words' centers along a line.
const WORD_PITCH_X = (TEXT_OUTER_X - TEXT_INNER_X) / WORDS_PER_LINE;

// Weight of each new word advance in the running reading-speed averages.
const SPEED_SMOOTHING = 0.3;

// Reading cursor: glides through newly arrived words (at speed the backend sends several
// per message) instead of jumping, and never passes the word being read.
const CURSOR_MIN_WORDS_PER_SECOND = 10;
const CURSOR_PACE_HEADROOM = 1.3;
const CURSOR_CATCH_UP_RATE = 6;

// Pace thresholds in words per second. Below AIRBORNE the fly walks the lines (with
// hysteresis, so it doesn't flicker between walking and flying) and above it skims them.
// From SCAN_START it increasingly scans whole lines, drifting out to the watch point and
// following more loosely, until at SCAN_FULL it watches the pages riffle from there.
const AIRBORNE_ENTER_WORDS_PER_SECOND = 5;
const AIRBORNE_LEAVE_WORDS_PER_SECOND = 3.5;
const SCAN_START_WORDS_PER_SECOND = 15;
const SCAN_FULL_WORDS_PER_SECOND = 40;
const SCAN_SMOOTHING_RATE = 1.5;

// Motion tuning. Springs are critically damped; frequencies in rad/s.
const WALK_SPRING_FREQUENCY = 10;
const SKIM_SPRING_FREQUENCY = 14;
const ESCAPE_SPRING_FREQUENCY = 8;
const WATCH_SPRING_FREQUENCY = 3;
const WATCH_FLIGHT_SPRING_FREQUENCY = 4;
const HOP_MIN_DISTANCE = 0.085;
const SKIM_HEIGHT = 0.06;
const GAIT_CYCLES_PER_UNIT = 32;
const MAX_SPRING_STEP_SECONDS = 1 / 120;
const HIGHLIGHT_FADE_RATE = 18;
// The watch point: hovering in front of the book, level with the reading along the lines
// and clear of any turning sheet. The fly waits there while a page turns, and reads from
// there once single words go by too fast.
const WATCH_HEIGHT = 0.18;
const WATCH_DEPTH_Z = 0.46;

// Page turns: unhurried at a reading pace, at speed never longer than a fixed share of
// the time a spread takes to read, so turning keeps up with the reading.
const TAKEOFF_LEAD_SECONDS = 0.3;
const BASE_FLIP_SECONDS = 1.6;
const MIN_FLIP_SECONDS = 0.25;
const FLIP_SHARE_OF_SPREAD = 0.25;
const MAX_FLIP_BACKLOG = 2;

/** A page being turned: the right page of fromSpread swings over to become the left page
 * of fromSpread + 1. lead is the takeoff time before the sheet starts moving. */
export type PageFlip = { fromSpread: number; elapsed: number; lead: number; duration: number };

type Hop = { from: THREE.Vector3; to: THREE.Vector3; elapsed: number; duration: number; height: number };

/** Why the fly is in the air: reading on the wing at speed, or waiting at the watch point
 * while a page turns under a walking reader. */
type Flight = "reading" | "watch";

const springTarget = new THREE.Vector3();
const skimTarget = new THREE.Vector3();
const targetVelocity = new THREE.Vector3();
const springAcceleration = new THREE.Vector3();
const velocityError = new THREE.Vector3();
const previousPosition = new THREE.Vector3();
const NO_VELOCITY = new THREE.Vector3();

/**
 * Motion model for the reading scene, advanced once per frame: turns the simulation's
 * word count into where the fly stands, walks, hops and flies, how its legs and wings
 * move, which spread the book shows, how far the current page turn is, and where the
 * reading highlight sits. Everything follows a reading cursor that glides through the
 * words as they arrive, and adapts to the pace: at speed the fly reads on the wing and
 * page turns shorten, so the scene keeps up with any reading speed. Book and Fly only
 * read this state; nothing here renders.
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
    /** The word index being read as the scene shows it, fractional while gliding on. */
    private readingCursor = 0;
    /** Words per second while reading, 0 while paused. */
    private pace = 0;
    private isReadingOnTheWing = false;
    /** 0 = following word by word, 1 = scanning whole lines from the watch point. */
    private scanAmount = 0;
    private hop: Hop | null = null;
    private flight: Flight | null = null;
    private isPlaced = false;
    private wordsPerSecond = 0;
    private averageWordsPerChange = 0;
    private averageSecondsPerChange = 0;
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
        this.trackReadingSpeed();
        this.advanceReadingCursor(delta);
        this.updatePace(delta);
        this.advancePageFlips(delta);

        const isTurning = this.flip !== null || this.displayedSpread < this.cursorSpread();
        this.moveFly(delta, isTurning, surface);
        this.orientFly(delta, surface);
        this.animateLimbs(delta);
        this.moveHighlight(delta, isTurning, surface);
    }

    /** Index of the word being read right now (wordsRead counts it as already read). */
    private currentWordIndex(): number {
        return Math.max(0, this.wordsRead - 1);
    }

    /** The spread the reading cursor is on. */
    private cursorSpread(): number {
        return Math.floor(this.readingCursor / WORDS_PER_SPREAD);
    }

    /** Words per second, which sets the pace of everything else: recent word advances over
     * the time they took (a ratio of averages, so bursty arrivals don't inflate it), falling
     * off once no new word has come for longer than usual. */
    private trackReadingSpeed(): void {
        const wordsAdvanced = this.wordsRead - (this.lastWordsRead ?? this.wordsRead);
        this.lastWordsRead = this.wordsRead;
        if (wordsAdvanced < 0 || wordsAdvanced > WORDS_PER_SPREAD) {
            // Restart, new book or a leap (e.g. reconnecting mid-book): start measuring afresh.
            this.averageWordsPerChange = 0;
            this.averageSecondsPerChange = 0;
            this.lastWordChangeTime = this.elapsedTime;
        } else if (wordsAdvanced > 0) {
            const secondsTaken = this.elapsedTime - this.lastWordChangeTime;
            this.averageWordsPerChange += (wordsAdvanced - this.averageWordsPerChange) * SPEED_SMOOTHING;
            this.averageSecondsPerChange += (secondsTaken - this.averageSecondsPerChange) * SPEED_SMOOTHING;
            this.lastWordChangeTime = this.elapsedTime;
        }
        const secondsSinceLastChange = this.elapsedTime - this.lastWordChangeTime;
        this.wordsPerSecond =
            this.averageWordsPerChange / Math.max(this.averageSecondsPerChange, secondsSinceLastChange, 1e-3);
    }

    /** Glides the reading cursor on toward the word being read: at least as fast as words
     * arrive, faster the further it trails, never past it. Jumps on a restart or new book,
     * and on a leap of more than a spread (e.g. after the tab was in the background). */
    private advanceReadingCursor(delta: number): void {
        const targetIndex = this.currentWordIndex();
        const gap = targetIndex - this.readingCursor;
        if (gap < 0 || gap > WORDS_PER_SPREAD) {
            this.readingCursor = targetIndex;
            return;
        }
        const glideSpeed = Math.max(
            CURSOR_MIN_WORDS_PER_SECOND,
            this.wordsPerSecond * CURSOR_PACE_HEADROOM,
            gap * CURSOR_CATCH_UP_RATE,
        );
        this.readingCursor = Math.min(targetIndex, this.readingCursor + glideSpeed * delta);
    }

    /** Sets how the fly reads at the current pace: on foot or on the wing, and how far it
     * has moved from following single words toward scanning whole lines. */
    private updatePace(delta: number): void {
        this.pace = this.isPaused ? 0 : this.wordsPerSecond;
        if (this.pace > AIRBORNE_ENTER_WORDS_PER_SECOND) this.isReadingOnTheWing = true;
        else if (this.pace < AIRBORNE_LEAVE_WORDS_PER_SECOND) this.isReadingOnTheWing = false;
        const targetScan = smoothstep(SCAN_START_WORDS_PER_SECOND, SCAN_FULL_WORDS_PER_SECOND, this.pace);
        this.scanAmount = damp(this.scanAmount, targetScan, SCAN_SMOOTHING_RATE, delta);
    }

    /** Turns one page at a time toward the cursor's spread, faster when behind. */
    private advancePageFlips(delta: number): void {
        const targetSpread = this.cursorSpread();
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
            // The sheet waits for a fly on or low over the page to get out of its way; one
            // already out at the watch point needs no head start.
            const takeoffLead = TAKEOFF_LEAD_SECONDS * (1 - this.scanAmount);
            this.flip = {
                fromSpread: this.displayedSpread,
                elapsed: 0,
                lead: PREFERS_REDUCED_MOTION || backlog > 1 ? 0 : takeoffLead,
                duration: PREFERS_REDUCED_MOTION ? 0 : Math.max(MIN_FLIP_SECONDS, this.flipSeconds() / backlog),
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

    /** How long a page turn takes: unhurried while reading slowly, at speed a fixed share
     * of the time the spread takes to read. */
    private flipSeconds(): number {
        const spreadSeconds = WORDS_PER_SPREAD / Math.max(this.pace, 1e-3);
        return clamp(spreadSeconds * FLIP_SHARE_OF_SPREAD, MIN_FLIP_SECONDS, BASE_FLIP_SECONDS);
    }

    /** Moves the fly: reading on the wing at speed; otherwise on foot at the current word,
     * or waiting at the watch point while a page turns. */
    private moveFly(delta: number, isTurning: boolean, surface: PageSurface): void {
        const readingSpot = this.readingSpotAt(this.readingCursor, surface);
        if (!this.isPlaced) {
            this.flyPosition.copy(readingSpot);
            this.isPlaced = true;
        }
        if (this.isReadingOnTheWing) {
            this.hop = null;
            this.flight = "reading";
            this.flyAlongReading(delta, readingSpot, isTurning);
        } else if (isTurning) {
            this.hop = null;
            this.flight = "watch";
            this.stepSpring(this.watchPoint(readingSpot), NO_VELOCITY, WATCH_FLIGHT_SPRING_FREQUENCY, delta);
        } else {
            this.walkAlongReading(delta, readingSpot, surface);
        }
    }

    /** Where the fly stands to read the word at a (fractional) word index. */
    private readingSpotAt(wordIndex: number, surface: PageSurface): THREE.Vector3 {
        const word = wordBoxAtIndex(wordIndex);
        return standingSpot(word.centerX - HEAD_OFFSET_X, word.centerZ, surface);
    }

    /** On foot: walks along the line in step with the reading, and hops to the next line or
     * page (or down from a flight) to where the reading will be when it lands. */
    private walkAlongReading(delta: number, readingSpot: THREE.Vector3, surface: PageSurface): void {
        const isLanding = this.flight !== null;
        this.flight = null;
        if (this.hop) {
            this.advanceHop(delta);
            return;
        }
        const distance = Math.hypot(this.flyPosition.x - readingSpot.x, this.flyPosition.z - readingSpot.z);
        if (isLanding || distance > HOP_MIN_DISTANCE) {
            this.startHop(surface);
            return;
        }
        this.stepSpring(readingSpot, targetVelocity.set(this.lineGlideSpeed(), 0, 0), WALK_SPRING_FREQUENCY, delta);
        // On foot the page carries the fly: stick to the surface, no vertical momentum.
        this.flyPosition.y = standingSpot(this.flyPosition.x, this.flyPosition.z, surface).y;
        this.flyVelocity.y = 0;
    }

    /** On the wing: skims low over the word being read and, as the pace grows, drifts out
     * to the watch point and follows ever more loosely; ducks out there whenever a page
     * turns, clear of the sweeping sheet. */
    private flyAlongReading(delta: number, readingSpot: THREE.Vector3, isTurning: boolean): void {
        const watchAmount = isTurning ? 1 : this.scanAmount;
        skimTarget.set(readingSpot.x, readingSpot.y + SKIM_HEIGHT + Math.sin(this.elapsedTime * 2.3) * 0.01, readingSpot.z);
        const target = this.watchPoint(readingSpot).lerp(skimTarget, 1 - watchAmount);
        targetVelocity.set(this.lineGlideSpeed() * (1 - watchAmount), 0, 0);
        const baseFrequency = isTurning ? ESCAPE_SPRING_FREQUENCY : SKIM_SPRING_FREQUENCY;
        this.stepSpring(target, targetVelocity, lerp(baseFrequency, WATCH_SPRING_FREQUENCY, this.scanAmount), delta);
    }

    /** How fast (page units per second) the reading moves along the current line; zero on
     * a line's last word, from where it jumps to the next line instead. */
    private lineGlideSpeed(): number {
        return this.readingCursor < lineEndIndex(this.readingCursor) ? this.pace * WORD_PITCH_X : 0;
    }

    /** The watch point level with readingSpot, drifting gently so the fly never freezes
     * mid-air. */
    private watchPoint(readingSpot: THREE.Vector3): THREE.Vector3 {
        const time = this.elapsedTime;
        return springTarget.set(
            readingSpot.x + Math.sin(time * 0.9) * 0.03,
            WATCH_HEIGHT + Math.sin(time * 1.7) * 0.015,
            WATCH_DEPTH_Z + Math.sin(time * 1.3) * 0.02,
        );
    }

    /** Starts a jump along an arc (line breaks, page changes, landings), aimed at where the
     * reading will be when the fly comes down so it doesn't land behind. */
    private startHop(surface: PageSurface): void {
        const roughDuration = hopDuration(this.flyPosition.distanceTo(this.readingSpotAt(this.readingCursor, surface)));
        const landing = this.readingSpotAt(this.cursorAfter(roughDuration), surface);
        if (PREFERS_REDUCED_MOTION) {
            this.flyPosition.copy(landing);
            this.flyVelocity.set(0, 0, 0);
            return;
        }
        const distance = this.flyPosition.distanceTo(landing);
        this.hop = {
            from: this.flyPosition.clone(),
            to: landing,
            elapsed: 0,
            duration: hopDuration(distance),
            height: 0.03 + distance * 0.22,
        };
    }

    /** The reading cursor `seconds` from now at the current pace, kept on its line (the
     * reading jumps to the next line from there). */
    private cursorAfter(seconds: number): number {
        return Math.min(this.readingCursor + this.pace * seconds, lineEndIndex(this.readingCursor));
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

    /** Critically damped spring step toward a target moving at `velocity`; feeding that
     * motion forward keeps the fly level with a gliding target instead of trailing it.
     * Sub-stepped for stability. */
    private stepSpring(target: THREE.Vector3, velocity: THREE.Vector3, frequency: number, delta: number): void {
        let remaining = delta;
        while (remaining > 0) {
            const step = Math.min(remaining, MAX_SPRING_STEP_SECONDS);
            velocityError.subVectors(velocity, this.flyVelocity);
            springAcceleration
                .subVectors(target, this.flyPosition)
                .multiplyScalar(frequency * frequency)
                .addScaledVector(velocityError, 2 * frequency);
            this.flyVelocity.addScaledVector(springAcceleration, step);
            this.flyPosition.addScaledVector(this.flyVelocity, step);
            remaining -= step;
        }
    }

    /** Faces the reading direction while reading and the travel direction on the way to
     * the watch point; follows the page's slope on foot, noses into the motion and banks
     * into turns aloft. */
    private orientFly(delta: number, surface: PageSurface): void {
        const isAirborne = this.flight !== null || this.hop !== null;
        this.airborneAmount = damp(this.airborneAmount, isAirborne ? 1 : 0, 10, delta);

        const horizontalSpeed = Math.hypot(this.flyVelocity.x, this.flyVelocity.z);
        const desiredHeading =
            this.flight === "watch" && horizontalSpeed > 0.12 ? Math.atan2(this.flyVelocity.z, this.flyVelocity.x) : 0;
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
        const isAirborne = this.flight !== null || this.hop !== null;
        this.legStride = damp(this.legStride, isAirborne ? 0 : clamp01(horizontalSpeed / 0.06), 10, delta);
        if (!isAirborne) this.gaitPhase += horizontalSpeed * GAIT_CYCLES_PER_UNIT * Math.PI * 2 * delta;

        const restingEnergy = this.isPaused ? 0.02 : 0.08 + 0.45 * this.arousal;
        const targetEnergy = Math.min(isAirborne ? 1 : restingEnergy, PREFERS_REDUCED_MOTION ? 0.3 : 1);
        this.wingEnergy = damp(this.wingEnergy, targetEnergy, 6, delta);
        this.wingPhase += Math.PI * 2 * (2.5 + 15 * this.wingEnergy) * delta;
    }

    /** Lays the highlighter under the word the cursor is on, widening it to the whole line
     * as the pace outruns single words; fades it out while a page turns. */
    private moveHighlight(delta: number, isTurning: boolean, surface: PageSurface): void {
        const word = wordBoxAtIndex(this.readingCursor);
        const line = lineBoxAtIndex(this.readingCursor);
        const centerX = lerp(word.centerX, line.centerX, this.scanAmount);
        this.highlightCenter.set(centerX, surface.heightAt(centerX, word.centerZ), word.centerZ);
        this.highlightWidth = lerp(word.width, line.width, this.scanAmount);
        this.highlightOpacity = damp(this.highlightOpacity, isTurning ? 0 : 1, HIGHLIGHT_FADE_RATE, delta);
    }
}

/** A point standing on the page surface at (x, z). */
function standingSpot(x: number, z: number, surface: PageSurface): THREE.Vector3 {
    return new THREE.Vector3(x, surface.heightAt(x, z) + STANDING_CLEARANCE, z);
}

/** How long a hop covering `distance` takes. */
function hopDuration(distance: number): number {
    return 0.26 + distance * 0.9;
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

/** 0 below edgeStart, 1 above edgeEnd, a smooth S-curve in between. */
function smoothstep(edgeStart: number, edgeEnd: number, value: number): number {
    const fraction = clamp01((value - edgeStart) / (edgeEnd - edgeStart));
    return fraction * fraction * (3 - 2 * fraction);
}

/** Gentle ease in and out over 0..1, used for hop arcs. */
function easeInOutSine(progress: number): number {
    return 0.5 - Math.cos(Math.PI * progress) / 2;
}

/** Stronger ease in and out over 0..1, used for page turns. */
export function easeInOutCubic(progress: number): number {
    return progress < 0.5 ? 4 * progress ** 3 : 1 - (-2 * progress + 2) ** 3 / 2;
}
