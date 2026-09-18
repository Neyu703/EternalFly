import { FlyBookScene } from "./components/FlyBookScene";
import { BrainScene } from "./components/BrainScene";
import { HudPanel } from "./components/HudPanel";
import { useMockTickData } from "./hooks/useMockTickData";
import "./App.css";

/** Dashboard layout: the two 3D panels (fly+book, brain) side by side on top, live data panel below. */
function App() {
  const tick = useMockTickData();

  return (
    <div className="dashboard">
      <div className="dashboard-top">
        <div className="dashboard-panel dashboard-panel--fly">
          <FlyBookScene />
        </div>
        <div className="dashboard-panel dashboard-panel--brain">
          <BrainScene activity={tick.neuropilActivity} />
        </div>
      </div>
      <div className="dashboard-panel dashboard-panel--hud">
        <HudPanel tick={tick} />
      </div>
    </div>
  );
}

export default App;
