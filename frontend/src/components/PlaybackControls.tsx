import { useEffect, useState } from "react";
import type { AutoplayMode, ControlMessage } from "../hooks/useWebSocketTickData";
import "./PlaybackControls.css";

const WPM_PRESETS = [60, 150, 300, 600, 1200, 3000, 10000] as const;
const CUSTOM_WPM_OPTION = "custom";
const DEFAULT_WORDS_PER_MINUTE = 150;
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
  const [wordsPerMinute, setWordsPerMinute] = useState(DEFAULT_WORDS_PER_MINUTE);
  const [isCustomWpm, setIsCustomWpm] = useState(false);
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

  function sendWordsPerMinute(nextWordsPerMinute: number) {
    setWordsPerMinute(nextWordsPerMinute);
    sendControlMessage({ type: "set_words_per_minute", value: nextWordsPerMinute });
  }

  function handleWpmPresetChange(event: React.ChangeEvent<HTMLSelectElement>) {
    if (event.target.value === CUSTOM_WPM_OPTION) {
      setIsCustomWpm(true);
      return;
    }
    setIsCustomWpm(false);
    sendWordsPerMinute(Number(event.target.value));
  }

  function handleCustomWpmChange(event: React.ChangeEvent<HTMLInputElement>) {
    const nextWordsPerMinute = Number(event.target.value);
    if (!Number.isFinite(nextWordsPerMinute) || nextWordsPerMinute <= 0) return;
    sendWordsPerMinute(nextWordsPerMinute);
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
        <span className="hud-label">Tempo</span>
        <select value={isCustomWpm ? CUSTOM_WPM_OPTION : wordsPerMinute} onChange={handleWpmPresetChange}>
          {WPM_PRESETS.map((preset) => (
            <option key={preset} value={preset}>
              {preset} Wörter/Min
            </option>
          ))}
          <option value={CUSTOM_WPM_OPTION}>Benutzerdefiniert…</option>
        </select>
        {isCustomWpm && (
          <input
            type="number"
            className="playback-speed-custom-input"
            min={1}
            step={1}
            value={wordsPerMinute}
            onChange={handleCustomWpmChange}
            placeholder="Wörter/Min"
          />
        )}
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
