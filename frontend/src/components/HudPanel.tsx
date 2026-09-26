import type { TickData } from "../types";
import type { ControlMessage } from "../hooks/useWebSocketTickData";
import { PlaybackControls } from "./PlaybackControls";
import { EmotionPanel } from "./EmotionPanel";
import { LiveLogPanel } from "./LiveLogPanel";
import "./HudPanel.css";

/** Bottom dashboard strip: playback controls (pause/speed/autoplay-on-finish), the 8
 * Plutchik emotion bars and the per-word live log, as three cards. The reading position
 * and the live neural metrics live in the stage footers instead (see App.tsx). */
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
      <EmotionPanel emotions={tick.emotions} />
      <LiveLogPanel tick={tick} />
    </div>
  );
}
