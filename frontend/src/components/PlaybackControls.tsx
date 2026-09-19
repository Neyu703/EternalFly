import { useEffect, useState } from "react";
import type { AutoplayMode, ControlMessage } from "../hooks/useWebSocketTickData";
import "./PlaybackControls.css";

const MIN_SPEED_MULTIPLIER = 0.25;
const MAX_SPEED_MULTIPLIER = 4;
const SPEED_STEP = 0.25;
const DEFAULT_AUTOPLAY_MODE: AutoplayMode = "restart";

/** Play/pause toggle, a reading-speed slider, and a selector for what the fly does once
 * it finishes a book (read it again from the start, or shuffle in a random Calibre book).
 * Sends every change straight to the backend over the shared WebSocket control channel. */
export function PlaybackControls({
  sendControlMessage,
}: {
  sendControlMessage: (message: ControlMessage) => void;
}) {
  const [isPaused, setIsPaused] = useState(false);
  const [speedMultiplier, setSpeedMultiplier] = useState(1);
  const [autoplayMode, setAutoplayMode] = useState<AutoplayMode>(DEFAULT_AUTOPLAY_MODE);

  useEffect(() => {
    sendControlMessage({ type: "set_autoplay_mode", mode: DEFAULT_AUTOPLAY_MODE });
    // Only on mount: establishes the default end-of-book behavior for a freshly opened
    // connection. Later changes are sent directly by handleAutoplayModeChange below.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function togglePaused() {
    const nextIsPaused = !isPaused;
    setIsPaused(nextIsPaused);
    sendControlMessage({ type: "set_paused", paused: nextIsPaused });
  }

  function handleSpeedChange(event: React.ChangeEvent<HTMLInputElement>) {
    const nextSpeedMultiplier = Number(event.target.value);
    setSpeedMultiplier(nextSpeedMultiplier);
    sendControlMessage({ type: "set_speed_multiplier", value: nextSpeedMultiplier });
  }

  function handleAutoplayModeChange(event: React.ChangeEvent<HTMLSelectElement>) {
    const nextAutoplayMode = event.target.value as AutoplayMode;
    setAutoplayMode(nextAutoplayMode);
    sendControlMessage({ type: "set_autoplay_mode", mode: nextAutoplayMode });
  }

  return (
    <div className="playback-controls">
      <button className="playback-toggle" onClick={togglePaused}>
        {isPaused ? "▶ Weiterlesen" : "⏸ Pausieren"}
      </button>

      <label className="playback-speed">
        <span className="hud-label">Tempo {speedMultiplier.toFixed(2)}×</span>
        <input
          type="range"
          min={MIN_SPEED_MULTIPLIER}
          max={MAX_SPEED_MULTIPLIER}
          step={SPEED_STEP}
          value={speedMultiplier}
          onChange={handleSpeedChange}
        />
      </label>

      <label className="playback-autoplay-mode">
        <span className="hud-label">Am Ende</span>
        <select value={autoplayMode} onChange={handleAutoplayModeChange}>
          <option value="restart">Von vorne</option>
          <option value="shuffle">Zufälliges Buch</option>
          <option value="off">Anhalten</option>
        </select>
      </label>
    </div>
  );
}
