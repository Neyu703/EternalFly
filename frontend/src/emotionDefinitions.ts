import type { EmotionName } from "./types";
import { AROUSAL_METRIC } from "./metrics";

/** How one emotional state is shown: its label, the brain circuit behind it (its tooltip)
 * and its color. */
export type EmotionDefinition = { label: string; circuit: string; color: string };

/** Display definition per emotional state (the backend sends the lowercase keys). The
 * colors are validated as a set (OKLCH lightness 0.5-0.75, >= 3:1 against the card
 * surfaces, every pair separated in normal vision and under simulated protanopia,
 * deuteranopia and tritanopia); arousal keeps the arousal metric's color, since it's the
 * same signal. */
export const EMOTION_DEFINITIONS: Record<EmotionName, EmotionDefinition> = {
    reward: {
        label: "Reward",
        circuit: "Firing of the reward dopamine neurons (PAM) of the mushroom body's medial lobe above their resting level",
        color: "#6da730",
    },
    aversion: {
        label: "Aversion",
        circuit: "Firing of the punishment dopamine neurons (PPL1) of the mushroom body's vertical lobe above their resting level",
        color: "#8557c8",
    },
    arousal: {
        label: "Arousal",
        circuit: "Firing of the octopamine neurons driving the central complex above their resting level",
        color: AROUSAL_METRIC.color,
    },
};
