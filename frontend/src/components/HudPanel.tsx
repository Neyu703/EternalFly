import { EMOTION_NAMES, type TickData } from "../types";
import type { ControlMessage } from "../hooks/useWebSocketTickData";
import { EMOTION_COLORS } from "../emotionColors";
import { LoadBookButton } from "./LoadBookButton";
import { CalibreBookList } from "./CalibreBookList";
import { PlaybackControls } from "./PlaybackControls";
import { LiveLogPanel } from "./LiveLogPanel";
import "./HudPanel.css";

/** Dashboard panel: current reading position, rating, the 8 Plutchik emotion bars, and
 * playback controls (pause/speed/autoplay-on-finish). The live neural activity chart is
 * rendered separately, floating over the brain scene (see App.tsx). */
export function HudPanel({
  tick,
  isPaused,
  onTogglePaused,
  sendControlMessage,
}: {
  tick: TickData;
  isPaused: boolean;
  onTogglePaused: () => void;
  sendControlMessage: (message: ControlMessage) => void;
}) {
  return (
    <div className="hud-panel">
      <PlaybackControls isPaused={isPaused} onTogglePaused={onTogglePaused} sendControlMessage={sendControlMessage} />

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

      <div className="hud-emotions">
        {EMOTION_NAMES.map((emotionName) => (
          <EmotionBar key={emotionName} name={emotionName} value={tick.emotions[emotionName] ?? 0} />
        ))}
      </div>

      <div className="hud-section hud-senses-section">
        <span className="hud-label">Sinne (aktuelles Wort)</span>
        <div className="hud-senses">
          {Object.entries(tick.senses)
            .sort(([nameA], [nameB]) => nameA.localeCompare(nameB))
            .map(([senseName, value]) => (
              <SenseBar key={senseName} name={senseName} value={value} />
            ))}
        </div>
      </div>

      {tick.wantsNewBook && (
        <div className="hud-bored-banner">
          Boah, langweilig — anderes Buch?
          <CalibreBookList />
        </div>
      )}

      <LoadBookButton />
      <LiveLogPanel tick={tick} />
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

/** One real sensory channel's current drive level (already 0..1 - see
 * reading_session.py's FrameResult.senses), e.g. sweet/bitter/looming/heat/cold. */
function SenseBar({ name, value }: { name: string; value: number }) {
  const clampedValue = Math.max(0, Math.min(1, value));
  return (
    <div className="hud-sense-row">
      <span className="hud-sense-label">{name}</span>
      <div className="hud-sense-track">
        <div className="hud-sense-fill" style={{ width: `${clampedValue * 100}%` }} />
      </div>
    </div>
  );
}
