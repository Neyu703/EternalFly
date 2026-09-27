import { EMOTION_NAMES, type EmotionName } from "../types";
import { EMOTION_DEFINITIONS } from "../emotionDefinitions";
import { clamp01 } from "../utils/math";
import { strongestEmotion } from "../utils/emotions";
import { formatPercent } from "../utils/format";
import "./EmotionPanel.css";

/** HUD card with the fly's current emotional states (reward, aversion, arousal: the ones
 * its brain has circuits for) as labeled bars, headed by whichever is strongest right now. */
export function EmotionPanel({ emotions }: { emotions: Record<string, number> }) {
    const dominantEmotion = strongestEmotion(emotions);

    return (
        <section className="card" aria-labelledby="emotion-panel-title">
            <div className="card-header">
                <h2 id="emotion-panel-title" className="overline">
                    Emotions
                </h2>
                {dominantEmotion && (
                    <span className="card-meta emotion-panel-dominant">
                        <span className="swatch" style={{ background: EMOTION_DEFINITIONS[dominantEmotion].color }} />
                        {EMOTION_DEFINITIONS[dominantEmotion].label} dominates
                    </span>
                )}
            </div>
            <ul className="emotion-list">
                {EMOTION_NAMES.map((emotionName) => (
                    <EmotionBar key={emotionName} emotionName={emotionName} value={emotions[emotionName] ?? 0} />
                ))}
            </ul>
        </section>
    );
}

/** One emotion's current 0..1 intensity: label, colored bar and percentage, with the
 * brain circuit behind it as the tooltip. */
function EmotionBar({ emotionName, value }: { emotionName: EmotionName; value: number }) {
    const clampedValue = clamp01(value);
    const { label, circuit, color } = EMOTION_DEFINITIONS[emotionName];
    return (
        <li className="emotion-row" title={circuit}>
            <span className="emotion-label truncate">{label}</span>
            <span className="progress">
                <span
                    className="progress-fill"
                    style={{ width: `${clampedValue * 100}%`, background: color }}
                />
            </span>
            <span className="emotion-value">{formatPercent(clampedValue)}</span>
        </li>
    );
}
