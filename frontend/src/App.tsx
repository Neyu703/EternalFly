import { FlyBookScene } from "./components/FlyBookScene";
import { BrainScene } from "./components/BrainScene";
import { HudPanel } from "./components/HudPanel";
import { useMockTickData } from "./hooks/useMockTickData";
import "./App.css";

/** Dashboard layout: two independent 3D panels (fly+book, brain) plus a live data panel. */
function App() {
  const tick = useMockTickData();

  return (
    <div className="dashboard">
      <div className="dashboard-panel dashboard-panel--fly">
        <FlyBookScene />
      </div>
      <div className="dashboard-panel dashboard-panel--brain">
        <BrainScene activity={tick.regionActivity} />
      </div>
      <div className="dashboard-panel dashboard-panel--hud">
        <HudPanel tick={tick} />
      </div>
    </div>
  );
}

export default App;
