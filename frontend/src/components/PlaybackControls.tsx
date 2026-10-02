import { useEffect, useState } from "react";
import type { AutoplayMode, ControlMessage } from "../hooks/useWebSocketTickData";
import { formatInteger } from "../utils/format";
import { PauseIcon, PlayIcon } from "./icons";
import "./PlaybackControls.css";

const WPM_PRESETS = [60, 150, 300, 600, 1200, 3000, 10000] as const;
const CUSTOM_WPM_OPTION = "custom";
const DEFAULT_WORDS_PER_MINUTE = 150;
const DEFAULT_AUTOPLAY_MODE: AutoplayMode = "restart";
// What each end-of-book mode is called, in the order the selector lists them.
const AUTOPLAY_MODE_LABELS: Record<AutoplayMode, string> = { restart: "Start over", shuffle: "Random book", off: "Stop" };

/** Whether value is a usable reading speed: a finite number of words above zero. */
function isValidWordsPerMinute(value: number): boolean {
  return Number.isFinite(value) && value > 0;
}

/** What PlaybackControls needs from the app: the lifted pause state and the control channel. */
export type PlaybackControlsProps = {
  isPaused: boolean;
  onTogglePaused: () => void;
  sendControlMessage: (message: ControlMessage) => void;
};

/** HUD card with the play/pause toggle, a reading-speed selector, and what the fly does
 * once it finishes a book (read it again from the start, shuffle in a random Calibre
 * book, or stop). Sends every change straight to the backend over the shared WebSocket
 * control channel. isPaused/onTogglePaused are lifted to the app root so other components
 * (e.g. the brain stage's sparklines) can also react to the paused state. */
export function PlaybackControls({ isPaused, onTogglePaused, sendControlMessage }: PlaybackControlsProps) {
  const [wordsPerMinute, setWordsPerMinute] = useState(DEFAULT_WORDS_PER_MINUTE);
  const [isCustomWpm, setIsCustomWpm] = useState(false);
  // The custom field's raw text, so it can be cleared or hold an invalid entry while typing.
  const [customWpmText, setCustomWpmText] = useState(String(DEFAULT_WORDS_PER_MINUTE));
  const isCustomWpmValid = isValidWordsPerMinute(Number(customWpmText));
  const [autoplayMode, setAutoplayMode] = useState<AutoplayMode>(DEFAULT_AUTOPLAY_MODE);

  useEffect(() => {
    // On mount (sendControlMessage is stable): tells a freshly opened connection the
    // defaults shown here, so the backend reads at the pace the selector says. Later
    // changes are sent by the change handlers below.
    sendControlMessage({ type: "set_words_per_minute", value: DEFAULT_WORDS_PER_MINUTE });
    sendControlMessage({ type: "set_autoplay_mode", mode: DEFAULT_AUTOPLAY_MODE });
  }, [sendControlMessage]);

  function sendWordsPerMinute(nextWordsPerMinute: number) {
    setWordsPerMinute(nextWordsPerMinute);
    sendControlMessage({ type: "set_words_per_minute", value: nextWordsPerMinute });
  }

  function handleWpmPresetChange(event: React.ChangeEvent<HTMLSelectElement>) {
    if (event.target.value === CUSTOM_WPM_OPTION) {
      setCustomWpmText(String(wordsPerMinute));
      setIsCustomWpm(true);
      return;
    }
    setIsCustomWpm(false);
    sendWordsPerMinute(Number(event.target.value));
  }

  function handleCustomWpmChange(event: React.ChangeEvent<HTMLInputElement>) {
    setCustomWpmText(event.target.value);
    const nextWordsPerMinute = Number(event.target.value);
    if (isValidWordsPerMinute(nextWordsPerMinute)) sendWordsPerMinute(nextWordsPerMinute);
  }

  function handleAutoplayModeChange(event: React.ChangeEvent<HTMLSelectElement>) {
    const nextAutoplayMode = event.target.value as AutoplayMode;
    setAutoplayMode(nextAutoplayMode);
    sendControlMessage({ type: "set_autoplay_mode", mode: nextAutoplayMode });
  }

  return (
    <section className="card playback" aria-labelledby="playback-title">
      <h2 id="playback-title" className="overline">
        Playback
      </h2>

      <button className="button button--primary playback-toggle" onClick={onTogglePaused} aria-keyshortcuts="Space">
        {isPaused ? <PlayIcon /> : <PauseIcon />}
        {isPaused ? "Resume" : "Pause"}
        <kbd className="playback-toggle-shortcut" aria-hidden="true">
          Space
        </kbd>
      </button>

      <label className="playback-field">
        <span className="playback-field-label">Speed</span>
        <select
          className="select"
          value={isCustomWpm ? CUSTOM_WPM_OPTION : wordsPerMinute}
          onChange={handleWpmPresetChange}
        >
          {WPM_PRESETS.map((preset) => (
            <option key={preset} value={preset}>
              {formatInteger(preset)} words/min
            </option>
          ))}
          <option value={CUSTOM_WPM_OPTION}>Custom…</option>
        </select>
      </label>

      {isCustomWpm && (
        <label className="playback-field">
          <span className="playback-field-label">Words/min</span>
          <input
            type="number"
            className="input"
            min={1}
            step={1}
            value={customWpmText}
            onChange={handleCustomWpmChange}
            aria-invalid={!isCustomWpmValid}
            aria-describedby={isCustomWpmValid ? undefined : "custom-wpm-error"}
          />
        </label>
      )}
      {isCustomWpm && !isCustomWpmValid && (
        <p id="custom-wpm-error" className="error-text playback-field-error">
          Enter a speed above 0.
        </p>
      )}

      <label className="playback-field">
        <span className="playback-field-label">At the end</span>
        <select className="select" value={autoplayMode} onChange={handleAutoplayModeChange}>
          {Object.entries(AUTOPLAY_MODE_LABELS).map(([mode, label]) => (
            <option key={mode} value={mode}>
              {label}
            </option>
          ))}
        </select>
      </label>
    </section>
  );
}
