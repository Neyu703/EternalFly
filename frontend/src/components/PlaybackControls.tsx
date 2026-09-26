import { useEffect, useState } from "react";
import type { AutoplayMode, ControlMessage } from "../hooks/useWebSocketTickData";
import { formatInteger } from "../utils/format";
import { PauseIcon, PlayIcon } from "./icons";
import "./PlaybackControls.css";

const WPM_PRESETS = [60, 150, 300, 600, 1200, 3000, 10000] as const;
const CUSTOM_WPM_OPTION = "custom";
const DEFAULT_WORDS_PER_MINUTE = 150;
const DEFAULT_AUTOPLAY_MODE: AutoplayMode = "restart";

/** HUD card with the play/pause toggle, a reading-speed selector, and what the fly does
 * once it finishes a book (read it again from the start, shuffle in a random Calibre
 * book, or stop). Sends every change straight to the backend over the shared WebSocket
 * control channel. isPaused/onTogglePaused are lifted to the app root so other components
 * (e.g. the brain stage's sparklines) can also react to the paused state. */
export function PlaybackControls({
  isPaused,
  onTogglePaused,
  sendControlMessage,
}: {
  isPaused: boolean;
  onTogglePaused: () => void;
  sendControlMessage: (message: ControlMessage) => void;
}) {
  const [wordsPerMinute, setWordsPerMinute] = useState(DEFAULT_WORDS_PER_MINUTE);
  const [isCustomWpm, setIsCustomWpm] = useState(false);
  const [autoplayMode, setAutoplayMode] = useState<AutoplayMode>(DEFAULT_AUTOPLAY_MODE);

  useEffect(() => {
    sendControlMessage({ type: "set_autoplay_mode", mode: DEFAULT_AUTOPLAY_MODE });
    // Only on mount: establishes the default end-of-book behavior for a freshly opened
    // connection. Later changes are sent directly by handleAutoplayModeChange below.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

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
    <section className="card playback" aria-labelledby="playback-title">
      <h2 id="playback-title" className="overline">
        Wiedergabe
      </h2>

      <button className="button button--primary playback-toggle" onClick={onTogglePaused}>
        {isPaused ? <PlayIcon /> : <PauseIcon />}
        {isPaused ? "Weiterlesen" : "Pausieren"}
      </button>

      <label className="playback-field">
        <span className="playback-field-label">Tempo</span>
        <select
          className="select"
          value={isCustomWpm ? CUSTOM_WPM_OPTION : wordsPerMinute}
          onChange={handleWpmPresetChange}
        >
          {WPM_PRESETS.map((preset) => (
            <option key={preset} value={preset}>
              {formatInteger(preset)} Wörter/Min
            </option>
          ))}
          <option value={CUSTOM_WPM_OPTION}>Benutzerdefiniert…</option>
        </select>
      </label>

      {isCustomWpm && (
        <label className="playback-field">
          <span className="playback-field-label">Wörter/Min</span>
          <input
            type="number"
            className="input"
            min={1}
            step={1}
            value={wordsPerMinute}
            onChange={handleCustomWpmChange}
          />
        </label>
      )}

      <label className="playback-field">
        <span className="playback-field-label">Am Ende</span>
        <select className="select" value={autoplayMode} onChange={handleAutoplayModeChange}>
          <option value="restart">Von vorne</option>
          <option value="shuffle">Zufälliges Buch</option>
          <option value="off">Anhalten</option>
        </select>
      </label>
    </section>
  );
}
