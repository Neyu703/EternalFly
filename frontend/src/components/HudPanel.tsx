import { EMOTION_NAMES, type TickData } from "../types";
import "./HudPanel.css";

const EMOTION_COLORS: Record<string, string> = {
  joy: "#ffd166",
  trust: "#06d6a0",
  fear: "#8338ec",
  surprise: "#ff9f1c",
  sadness: "#4361ee",
  disgust: "#7cb518",
  anger: "#ef476f",
  anticipation: "#f4a261",
};

/** Dashboard panel: current reading position, rating and the 8 Plutchik emotion bars. */
export function HudPanel({ tick }: { tick: TickData }) {
  return (
    <div className="hud-panel">
      <div className="hud-section hud-word-section">
        <span className="hud-label">Liest:</span>
        <span className="hud-word">{tick.currentWord ?? "—"}</span>
      </div>

      <div className="hud-section hud-progress-section">
        <span className="hud-label">
          {tick.wordsRead} von {tick.totalWords} Wörtern
        </span>
        <div className="hud-progress-row">
          <div className="hud-progress-track">
            <div className="hud-progress-fill" style={{ width: `${tick.pageProgress * 100}%` }} />
          </div>
          <span className="hud-progress-percent">{Math.round(tick.pageProgress * 100)}%</span>
        </div>
      </div>

      <div className="hud-rating">
        <span className="hud-rating-value">{tick.rating0To10.toFixed(1)}</span>
        <span className="hud-rating-suffix">/ 10 Dopamin-Level</span>
      </div>

      <div className="hud-emotions">
        {EMOTION_NAMES.map((emotionName) => (
          <EmotionBar key={emotionName} name={emotionName} value={tick.emotions[emotionName] ?? 0} />
        ))}
      </div>

      {tick.wantsNewBook && <div className="hud-bored-banner">Boah, langweilig — anderes Buch?</div>}
    </div>
  );
}

function EmotionBar({ name, value }: { name: string; value: number }) {
  const clampedValue = Math.max(0, Math.min(1, value));
  return (
    <div className="hud-emotion-row">
      <span className="hud-emotion-label">{name}</span>
      <div className="hud-emotion-track">
        <div
          className="hud-emotion-fill"
          style={{ width: `${clampedValue * 100}%`, background: EMOTION_COLORS[name] }}
        />
      </div>
    </div>
  );
}
