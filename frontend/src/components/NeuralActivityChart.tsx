import { useEffect, useState } from "react";
import type { TickData } from "../types";
import "./NeuralActivityChart.css";

const HISTORY_LENGTH = 80;
const CHART_WIDTH = 120;
const CHART_HEIGHT = 32;

/** Mean spiking rate across the fly's tracked brain regions this tick (0..1), preferring
 * the fine-grained per-neuropil breakdown and falling back to the coarser approach/
 * avoidance/arousal pools when no neuropil data is available. */
function overallFiringRate(tick: TickData): number {
  const neuropilRates = Object.values(tick.neuropilActivity);
  const rates = neuropilRates.length > 0 ? neuropilRates : Object.values(tick.regionActivity);
  if (rates.length === 0) return 0;
  return rates.reduce((sum, rate) => sum + rate, 0) / rates.length;
}

/** Builds an SVG polyline's `points` attribute from a firing-rate history, scaling the
 * y-axis to the history's own peak so the trend stays visible at any activity level. */
function buildSparklinePoints(firingRateHistory: number[]): string {
  if (firingRateHistory.length === 0) return "";
  const peakRate = Math.max(...firingRateHistory, 0.01);
  return firingRateHistory
    .map((rate, index) => {
      const x = (index / Math.max(firingRateHistory.length - 1, 1)) * CHART_WIDTH;
      const y = CHART_HEIGHT - (rate / peakRate) * CHART_HEIGHT;
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");
}

/** Live-updating panel: the fly's current overall firing rate and arousal, plus a rolling
 * sparkline of firing rate over the last HISTORY_LENGTH ticks (~4s at the default tick rate). */
export function NeuralActivityChart({ tick }: { tick: TickData }) {
  const [firingRateHistory, setFiringRateHistory] = useState<number[]>([]);

  useEffect(() => {
    setFiringRateHistory((previousHistory) => {
      const nextHistory = [...previousHistory, overallFiringRate(tick)];
      return nextHistory.length > HISTORY_LENGTH ? nextHistory.slice(-HISTORY_LENGTH) : nextHistory;
    });
  }, [tick]);

  const currentFiringRate = firingRateHistory[firingRateHistory.length - 1] ?? 0;
  const arousal = tick.regionActivity.arousal ?? 0;
  const sparklinePoints = buildSparklinePoints(firingRateHistory);
  const latestPoint = sparklinePoints.split(" ").at(-1)?.split(",").map(Number);

  return (
    <div className="neural-activity-panel">
      <span className="hud-label">Live Neural Activity</span>
      <div className="neural-activity-stats">
        <div className="neural-activity-stat">
          <span className="neural-activity-stat-value">{(currentFiringRate * 100).toFixed(0)}%</span>
          <span className="neural-activity-stat-label">Feuerrate</span>
        </div>
        <div className="neural-activity-stat">
          <span className="neural-activity-stat-value">{(arousal * 100).toFixed(0)}%</span>
          <span className="neural-activity-stat-label">Erregung</span>
        </div>
      </div>
      <svg
        className="neural-activity-sparkline"
        viewBox={`0 0 ${CHART_WIDTH} ${CHART_HEIGHT}`}
        preserveAspectRatio="none"
      >
        <polyline points={sparklinePoints} fill="none" stroke="#7fd4ff" strokeWidth="2" strokeLinecap="round" />
        {latestPoint && <circle cx={latestPoint[0]} cy={latestPoint[1]} r="2" fill="#7fd4ff" />}
      </svg>
    </div>
  );
}
