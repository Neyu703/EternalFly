import { useEffect, useState } from "react";
import type { TickData } from "../types";
import "./NeuralActivityChart.css";

const HISTORY_LENGTH = 80;
const CHART_WIDTH = 300;
const CHART_HEIGHT = 46;

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
function buildSparklinePoints(firingRateHistory: number[], peakRate: number): string {
  if (firingRateHistory.length === 0) return "";
  return firingRateHistory
    .map((rate, index) => {
      const x = (index / Math.max(firingRateHistory.length - 1, 1)) * CHART_WIDTH;
      const y = CHART_HEIGHT - (rate / peakRate) * CHART_HEIGHT;
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");
}

function formatPercent(fraction: number): string {
  return `${Math.round(fraction * 100)}%`;
}

/** Live-updating status card: the fly's current dopamine rating, overall firing rate and
 * arousal, plus a rolling sparkline of firing rate over the last HISTORY_LENGTH ticks
 * (~4s at the default tick rate), auto-scaled to whatever peak that window has seen. */
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
  const trackedRegionCount = Object.keys(tick.neuropilActivity).length;
  const peakFiringRate = Math.max(...firingRateHistory, 0.01);
  const sparklinePoints = buildSparklinePoints(firingRateHistory, peakFiringRate);
  const latestPoint = sparklinePoints.split(" ").at(-1)?.split(",").map(Number);

  return (
    <div className="neural-activity-panel">
      <div className="neural-activity-header">
        <span className="neural-activity-title">EternalFly · Live Neural Activity</span>
        <span className="neural-activity-live-dot" />
      </div>

      <div className="neural-activity-stats">
        <div className="neural-activity-stat">
          <span className="neural-activity-stat-label">Dopamin-Level</span>
          <span className="neural-activity-stat-value neural-activity-stat-value--rating">
            {tick.rating0To10.toFixed(1)} <span className="neural-activity-stat-unit">/10</span>
          </span>
          <span className="neural-activity-stat-caption">Bewertung des Buchs</span>
        </div>
        <div className="neural-activity-stat">
          <span className="neural-activity-stat-label">Feuerrate</span>
          <span className="neural-activity-stat-value neural-activity-stat-value--firing-rate">
            {formatPercent(currentFiringRate)}
          </span>
          <span className="neural-activity-stat-caption">
            {trackedRegionCount > 0 ? `${trackedRegionCount} Hirnregionen` : "approach/avoidance/arousal"}
          </span>
        </div>
        <div className="neural-activity-stat">
          <span className="neural-activity-stat-label">Erregung</span>
          <span className="neural-activity-stat-value neural-activity-stat-value--arousal">
            {formatPercent(arousal)}
          </span>
          <span className="neural-activity-stat-caption">Arousal-Pool</span>
        </div>
      </div>

      <div className="neural-activity-chart-caption">
        <span>Feuerrate-Verlauf</span>
        <span>0–{formatPercent(peakFiringRate)} · auto-scale</span>
      </div>
      <svg
        className="neural-activity-sparkline"
        viewBox={`0 0 ${CHART_WIDTH} ${CHART_HEIGHT}`}
        preserveAspectRatio="none"
      >
        <polyline
          points={sparklinePoints}
          fill="none"
          stroke="#c3e86b"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
        {latestPoint && <circle cx={latestPoint[0]} cy={latestPoint[1]} r="2.5" fill="#c3e86b" />}
      </svg>
    </div>
  );
}
