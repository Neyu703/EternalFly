import type { TickData } from "../types";
import { PlaybackControls, type PlaybackControlsProps } from "./PlaybackControls";
import { EmotionPanel } from "./EmotionPanel";
import { LiveLogPanel } from "./LiveLogPanel";
import "./HudPanel.css";

/** Bottom dashboard strip: playback controls (pause/speed/autoplay-on-finish), the fly's
 * emotion bars and the per-word live log, as three cards. The reading position
 * and the live neural metrics live in the stage footers instead (see App.tsx). */
export function HudPanel({
  tick,
  isPaused,
  onTogglePaused,
  sendControlMessage,
}: PlaybackControlsProps & { tick: TickData }) {
  return (
    <div className="hud-panel">
      <PlaybackControls isPaused={isPaused} onTogglePaused={onTogglePaused} sendControlMessage={sendControlMessage} />
      <EmotionPanel emotions={tick.emotions} />
      <LiveLogPanel tick={tick} />
    </div>
  );
}
