import { useState } from "react";
import { FlyBookScene } from "./components/FlyBookScene";
import { BrainScene } from "./components/BrainScene";
import { HudPanel } from "./components/HudPanel";
import { NeuralActivityChart } from "./components/NeuralActivityChart";
import { BookOverviewPanel } from "./components/BookOverviewPanel";
import { useWebSocketTickData } from "./hooks/useWebSocketTickData";
import { useBookHistory } from "./hooks/useBookHistory";
import { normalizeRegionActivity } from "./regionActivity";
import "./App.css";

const BACKEND_WS_URL = "ws://127.0.0.1:8000/ws";

/** Dashboard layout: the two 3D panels (fly+book, brain) side by side on top, live data panel below.
 * Owns isPaused (rather than PlaybackControls) so both the play/pause button and the Live
 * Neural Activity chart - which needs to freeze its sparklines while paused - agree on it. */
function App() {
  const { tick, sendControlMessage, firedNeuronIndicesRef } = useWebSocketTickData(BACKEND_WS_URL);
  const [isPaused, setIsPaused] = useState(false);
  const { finishedBookHistory, dismissFinishedBookHistory } = useBookHistory(tick);

  function togglePaused() {
    const nextIsPaused = !isPaused;
    setIsPaused(nextIsPaused);
    sendControlMessage({ type: "set_paused", paused: nextIsPaused });
  }

  return (
    <div className="dashboard">
      <div className="dashboard-top">
        <div className="dashboard-panel dashboard-panel--fly">
          <FlyBookScene
            behaviors={
              tick
                ? {
                    escape: tick.behaviors.escape ?? 0,
                    feeding: tick.behaviors.feeding ?? 0,
                    backing: tick.behaviors.backing ?? 0,
                    turn_left: tick.behaviors.turn_left ?? 0,
                    turn_right: tick.behaviors.turn_right ?? 0,
                  }
                : undefined
            }
            arousal={tick ? normalizeRegionActivity(tick.regionActivity.arousal ?? 0, "arousal") : undefined}
          />
        </div>
        <div className="dashboard-panel dashboard-panel--brain">
          <BrainScene activity={tick?.neuropilActivity} firedNeuronIndicesRef={firedNeuronIndicesRef} />
          {tick && <NeuralActivityChart tick={tick} isPaused={isPaused} />}
        </div>
      </div>
      <div className="dashboard-panel dashboard-panel--hud">
        {tick ? (
          <HudPanel tick={tick} isPaused={isPaused} onTogglePaused={togglePaused} sendControlMessage={sendControlMessage} />
        ) : (
          <div className="hud-connecting">Verbinde mit der Simulation…</div>
        )}
      </div>
      {finishedBookHistory && (
        <BookOverviewPanel history={finishedBookHistory} onDismiss={dismissFinishedBookHistory} />
      )}
    </div>
  );
}

export default App;
