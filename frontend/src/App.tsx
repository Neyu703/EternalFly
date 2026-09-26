import { useState } from "react";
import { AppHeader, type ConnectionStatus } from "./components/AppHeader";
import { FlyBookScene } from "./components/FlyBookScene";
import { BrainScene } from "./components/BrainScene";
import { ReadingStatus } from "./components/ReadingStatus";
import { NeuralActivityChart } from "./components/NeuralActivityChart";
import { CalibreBookList } from "./components/CalibreBookList";
import { HudPanel } from "./components/HudPanel";
import { BookOverviewPanel } from "./components/BookOverviewPanel";
import { useWebSocketTickData } from "./hooks/useWebSocketTickData";
import { useBookHistory } from "./hooks/useBookHistory";
import "./App.css";

const BACKEND_WS_URL = "ws://127.0.0.1:8000/ws";

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

  function togglePaused() {
    const nextIsPaused = !isPaused;
    setIsPaused(nextIsPaused);
    sendControlMessage({ type: "set_paused", paused: nextIsPaused });
  }

  return (
    <div className="app">
      <AppHeader connectionStatus={connectionStatusOf(isConnected, isPaused, tick !== null)} />

      <main className="stage">
        <section className="stage-panel stage-panel--reader" aria-label="Leser">
          <StageCaption title="Leser" subtitle="Drosophila melanogaster" />
          <div className="stage-canvas">
            <FlyBookScene />
          </div>
          {tick?.wantsNewBook && (
            <aside className="bored-notice" role="status">
              <p className="bored-notice-title">
                <span className="swatch" style={{ background: "var(--status-warning)" }} />
                Boah, langweilig — anderes Buch?
              </p>
              <p className="bored-notice-text">Die Fliege verliert das Interesse. Lade oben rechts ein neues Buch.</p>
              <CalibreBookList />
            </aside>
          )}
          <div className="stage-footer">{tick && <ReadingStatus tick={tick} />}</div>
        </section>

        <section className="stage-panel stage-panel--brain" aria-label="Gehirn">
          <StageCaption title="Gehirn" subtitle="Echtes FlyWire-Konnektom · ≈ 139.000 Neuronen" />
          <div className="stage-canvas">
            <BrainScene activity={tick?.neuropilActivity} />
          </div>
          <div className="stage-footer">{tick && <NeuralActivityChart tick={tick} isPaused={isPaused} />}</div>
        </section>
      </main>

      <section className="hud" aria-label="Steuerung und Live-Daten">
        {tick ? (
          <HudPanel tick={tick} isPaused={isPaused} onTogglePaused={togglePaused} sendControlMessage={sendControlMessage} />
        ) : (
          <div className="hud-connecting" role="status">
            <span className="spinner" aria-hidden="true" />
            <div>
              <p className="hud-connecting-title">Verbinde mit der Simulation…</p>
              <p className="hud-connecting-hint">Das Backend muss unter 127.0.0.1:8000 laufen (make backend).</p>
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

/** Top-left label of a 3D stage, plus a hover hint that the model can be orbited. */
function StageCaption({ title, subtitle }: { title: string; subtitle: string }) {
  return (
    <>
      <div className="stage-caption">
        <span className="overline">{title}</span>
        <span className="stage-caption-subtitle">{subtitle}</span>
      </div>
      <span className="stage-hint" aria-hidden="true">
        Ziehen zum Drehen · Scrollen zum Zoomen
      </span>
    </>
  );
}

export default App;
