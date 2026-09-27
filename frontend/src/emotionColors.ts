import type { EmotionName } from "./types";

/** Display color per emotional state, shared by the HUD's emotion bars and
 * BookOverviewPanel's end-of-book emotion sparklines. Arousal keeps the arousal metric's
 * color, since it's the same signal. Validated as a set (OKLCH lightness 0.5-0.75, >= 3:1
 * against the card surfaces, every pair separated in normal vision and under simulated
 * protanopia, deuteranopia and tritanopia). */
export const EMOTION_COLORS: Record<EmotionName, string> = {
  reward: "#6da730",
  aversion: "#8557c8",
  arousal: "#d14186",
};
