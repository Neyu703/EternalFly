import { FlyBookScene } from "./components/FlyBookScene";
import { BrainScene } from "./components/BrainScene";
import { HudPanel } from "./components/HudPanel";
import { useWebSocketTickData } from "./hooks/useWebSocketTickData";
import "./App.css";

const BACKEND_WS_URL = "ws://127.0.0.1:8000/ws";

/** Dashboard layout: the two 3D panels (fly+book, brain) side by side on top, live data panel below. */
function App() {
  const { tick, sendControlMessage } = useWebSocketTickData(BACKEND_WS_URL);

  return (
    <div className="dashboard">
      <div className="dashboard-top">
        <div className="dashboard-panel dashboard-panel--fly">
          <FlyBookScene />
        </div>
        <div className="dashboard-panel dashboard-panel--brain">
          <BrainScene activity={tick?.neuropilActivity} />
        </div>
      </div>
      <div className="dashboard-panel dashboard-panel--hud">
        {tick ? (
          <HudPanel tick={tick} sendControlMessage={sendControlMessage} />
        ) : (
          <div className="hud-connecting">Verbinde mit der Simulation…</div>
        )}
      </div>
    </div>
  );
}

export default App;
