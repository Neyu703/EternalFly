import { useState } from "react";
import { AppHeader, type ConnectionStatus } from "./components/AppHeader";
import { FlyBookScene } from "./components/FlyBookScene";
import { BrainScene } from "./components/BrainScene";
import { ReadingStatus } from "./components/ReadingStatus";
import { NeuralActivityChart } from "./components/NeuralActivityChart";
import { CalibreBookList } from "./components/CalibreBookList";
import { HudPanel } from "./components/HudPanel";
import { Swatch } from "./components/Swatch";
import { BookOverviewPanel } from "./components/BookOverviewPanel";
import { useWebSocketTickData } from "./hooks/useWebSocketTickData";
import { useBookHistory } from "./hooks/useBookHistory";
import { BACKEND_HOST, BACKEND_WS_URL } from "./backendUrl";
import { arousalOf } from "./utils/emotions";
import "./App.css";

/** Header status for the simulation link: open sockets are live or paused, a closed one
 * is still connecting until the first tick ever arrived, and disconnected afterwards. */
function connectionStatusOf(isConnected: boolean, isPaused: boolean, hasReceivedTick: boolean): ConnectionStatus {
  if (!isConnected) return hasReceivedTick ? "disconnected" : "connecting";
  return isPaused ? "paused" : "live";
}

/** Dashboard layout: header on top, the two 3D stages (fly+book, brain) side by side, each
 * with its own live-data footer, and the controls/emotions/log strip below. Owns isPaused
 * (rather than PlaybackControls) so both the play/pause button and the brain stage's
 * sparklines - which freeze while paused - agree on it. */
function App() {
  const { tick, isConnected, sendControlMessage } = useWebSocketTickData(BACKEND_WS_URL);
  const [isPaused, setIsPaused] = useState(false);
  const { finishedBookHistory, dismissFinishedBookHistory } = useBookHistory(tick);
  // The 3D stages stop animating the reading and the firing while paused or offline.
  const isSimulationFrozen = isPaused || !isConnected;

  function togglePaused() {
    const nextIsPaused = !isPaused;
    setIsPaused(nextIsPaused);
    sendControlMessage({ type: "set_paused", paused: nextIsPaused });
  }

  return (
    <div className="app">
      <AppHeader connectionStatus={connectionStatusOf(isConnected, isPaused, tick !== null)} />

      <main className="stage">
        <section className="stage-panel stage-panel--reader" aria-label="Reader">
          <StageCaption title="Reader" subtitle="Drosophila melanogaster" hint="Drag to rotate · Scroll to zoom" />
          <div className="stage-canvas">
            <FlyBookScene
              wordsRead={tick?.wordsRead ?? 0}
              isPaused={isSimulationFrozen}
              arousal={tick ? arousalOf(tick.regionActivity) : 0}
            />
          </div>
          {tick?.wantsNewBook && (
            <aside className="overlay-panel bored-notice" role="status">
              <p className="bored-notice-title">
                <Swatch color="var(--status-warning)" />
                Ugh, boring — another book?
              </p>
              <p className="bored-notice-text">The fly is losing interest. Load a new book at the top right.</p>
              <CalibreBookList />
            </aside>
          )}
          <div className="stage-footer">{tick && <ReadingStatus tick={tick} />}</div>
        </section>

        <section className="stage-panel stage-panel--brain" aria-label="Brain">
          <StageCaption
            title="Brain"
            subtitle="Real FlyWire connectome · ≈ 139,000 neurons"
            hint="Click a region for details · Drag to rotate"
          />
          <div className="stage-canvas">
            <BrainScene activity={tick?.neuropilActivity} isPaused={isSimulationFrozen} />
          </div>
          <div className="stage-footer">{tick && <NeuralActivityChart tick={tick} isPaused={isPaused} />}</div>
        </section>
      </main>

      <section className="hud" aria-label="Controls and live data">
        {tick ? (
          <HudPanel tick={tick} isPaused={isPaused} onTogglePaused={togglePaused} sendControlMessage={sendControlMessage} />
        ) : (
          <div className="hud-connecting" role="status">
            <span className="spinner" aria-hidden="true" />
            <div>
              <p className="hud-connecting-title">Connecting to the simulation…</p>
              <p className="hud-connecting-hint">The backend has to be running at {BACKEND_HOST} (make backend).</p>
            </div>
          </div>
        )}
      </section>

      {finishedBookHistory && (
        <BookOverviewPanel history={finishedBookHistory} onDismiss={dismissFinishedBookHistory} />
      )}
    </div>
  );
}

/** Top-left label of a 3D stage, plus a hover hint on how to interact with it. */
function StageCaption({ title, subtitle, hint }: { title: string; subtitle: string; hint: string }) {
  return (
    <>
      <div className="stage-caption">
        <span className="overline">{title}</span>
        <span className="stage-caption-subtitle">{subtitle}</span>
      </div>
      <span className="stage-hint" aria-hidden="true">
        {hint}
      </span>
    </>
  );
}

export default App;
