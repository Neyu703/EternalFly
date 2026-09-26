import { EMOTION_LABELS, EMOTION_NAMES, type EmotionName } from "../types";
import { EMOTION_COLORS } from "../emotionColors";
import { strongestEmotion } from "../utils/emotions";
import { formatPercent } from "../utils/format";
import "./EmotionPanel.css";

/** HUD card with the fly's 8 current Plutchik emotions as labeled bars, headed by
 * whichever emotion is strongest right now. */
export function EmotionPanel({ emotions }: { emotions: Record<string, number> }) {
    const dominantEmotion = strongestEmotion(emotions);

    return (
        <section className="card emotion-panel" aria-labelledby="emotion-panel-title">
            <div className="card-header">
                <h2 id="emotion-panel-title" className="overline">
                    Emotionen
                </h2>
                {dominantEmotion && (
                    <span className="card-meta emotion-panel-dominant">
                        <span className="swatch" style={{ background: EMOTION_COLORS[dominantEmotion] }} />
                        {EMOTION_LABELS[dominantEmotion]} überwiegt
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

/** One emotion's current 0..1 intensity: label, colored bar and percentage. */
function EmotionBar({ emotionName, value }: { emotionName: EmotionName; value: number }) {
    const clampedValue = Math.max(0, Math.min(1, value));
    return (
        <li className="emotion-row">
            <span className="emotion-label">{EMOTION_LABELS[emotionName]}</span>
            <span className="emotion-track">
                <span
                    className="emotion-fill"
                    style={{ width: `${clampedValue * 100}%`, background: EMOTION_COLORS[emotionName] }}
                />
            </span>
            <span className="emotion-value">{formatPercent(clampedValue)}</span>
        </li>
    );
}
