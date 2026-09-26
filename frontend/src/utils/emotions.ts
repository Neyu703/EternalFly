import { EMOTION_NAMES, type EmotionName } from "../types";

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
