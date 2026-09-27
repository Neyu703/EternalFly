import { EMOTION_NAMES, type EmotionName } from "../types";

/** The fly's arousal (0..1) out of a tick's region activity, 0 when it's missing. */
export function arousalOf(regionActivity: Record<string, number>): number {
  return regionActivity.arousal ?? 0;
}

/** The emotion with the highest value in emotionValues, or null while every emotion is
 * still at zero (so an idle fly isn't reported as feeling the first emotion in the list). */
export function strongestEmotion(emotionValues: Record<string, number>): EmotionName | null {
    let strongest: EmotionName | null = null;
    let strongestValue = 0;
    for (const emotionName of EMOTION_NAMES) {
        const value = emotionValues[emotionName] ?? 0;
        if (value > strongestValue) {
            strongest = emotionName;
            strongestValue = value;
        }
    }
    return strongest;
}
