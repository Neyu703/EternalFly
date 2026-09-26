import type { EmotionName } from "./types";

/** Display color per Plutchik emotion, shared by the HUD's emotion bars and
 * BookOverviewPanel's end-of-book emotion sparklines. Stepped for the dark UI surfaces
 * and validated as a set in EMOTION_NAMES order (dark lightness band, >= 3:1 against
 * the card surfaces, adjacent-pair color-vision-deficiency and normal-vision separation). */
export const EMOTION_COLORS: Record<EmotionName, string> = {
  joy: "#b48c05",
  trust: "#05a480",
  fear: "#8557c8",
  surprise: "#04a3be",
  sadness: "#366bd3",
  disgust: "#6da730",
  anger: "#c92e3b",
  anticipation: "#d97605",
};
