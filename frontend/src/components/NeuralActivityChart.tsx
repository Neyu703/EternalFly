import { useEffect, useState } from "react";
import type { TickData } from "../types";
import "./NeuralActivityChart.css";

const HISTORY_LENGTH = 80;
const CHART_WIDTH = 300;
const CHART_HEIGHT = 28;

/** Mean spiking rate across the fly's tracked brain regions this tick (0..1), preferring
 * the fine-grained per-neuropil breakdown and falling back to the coarser approach/
 * avoidance/arousal pools when no neuropil data is available. */
function overallFiringRate(tick: TickData): number {
  const neuropilRates = Object.values(tick.neuropilActivity);
  const rates = neuropilRates.length > 0 ? neuropilRates : Object.values(tick.regionActivity);
  if (rates.length === 0) return 0;
  return rates.reduce((sum, rate) => sum + rate, 0) / rates.length;
}

/** Builds an SVG polyline's `points` attribute from a value history, scaling the y-axis
 * against maxValue (either a fixed domain ceiling or the history's own peak). */
function buildSparklinePoints(history: number[], maxValue: number): string {
  if (history.length === 0) return "";
  return history
    .map((value, index) => {
      const x = (index / Math.max(history.length - 1, 1)) * CHART_WIDTH;
      const y = CHART_HEIGHT - (value / maxValue) * CHART_HEIGHT;
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");
}

function formatPercent(fraction: number): string {
  return `${Math.round(fraction * 100)}%`;
}

/** One small labeled sparkline: a value history plotted against either a fixed domain
 * (maxValue) or, when maxValue is omitted, the history's own observed peak. */
function Sparkline({
  label,
  history,
  maxValue,
  color,
  formatDomain,
}: {
  label: string;
  history: number[];
  maxValue?: number;
  color: string;
  formatDomain: (value: number) => string;
}) {
  const peak = maxValue ?? Math.max(...history, 0.0001);
  const points = buildSparklinePoints(history, peak);
  const latestPoint = points.split(" ").at(-1)?.split(",").map(Number);
  return (
    <div className="neural-activity-sparkline-block">
      <div className="neural-activity-chart-caption">
        <span>{label}</span>
        <span>
          0–{formatDomain(peak)} {maxValue === undefined && "· auto-scale"}
        </span>
      </div>
      <svg
        className="neural-activity-sparkline"
        viewBox={`0 0 ${CHART_WIDTH} ${CHART_HEIGHT}`}
        preserveAspectRatio="none"
      >
        <polyline points={points} fill="none" stroke={color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
        {latestPoint && <circle cx={latestPoint[0]} cy={latestPoint[1]} r="2.5" fill={color} />}
      </svg>
    </div>
  );
}

/** Live-updating status card: the fly's current dopamine rating, overall firing rate and
 * arousal, each with its own rolling sparkline over the last HISTORY_LENGTH ticks (~4s
 * at the default tick rate). Dopamine and arousal plot against their real fixed 0-10 /
 * 0-100% domain (they're already calibrated to it, see emotion_decoder.py), while firing
 * rate auto-scales to its own observed peak since it isn't ceiling-normalized. */
export function NeuralActivityChart({ tick }: { tick: TickData }) {
  const [firingRateHistory, setFiringRateHistory] = useState<number[]>([]);
  const [ratingHistory, setRatingHistory] = useState<number[]>([]);
  const [arousalHistory, setArousalHistory] = useState<number[]>([]);

  useEffect(() => {
    const pushCapped = (previousHistory: number[], value: number): number[] => {
      const nextHistory = [...previousHistory, value];
      return nextHistory.length > HISTORY_LENGTH ? nextHistory.slice(-HISTORY_LENGTH) : nextHistory;
    };
    setFiringRateHistory((previousHistory) => pushCapped(previousHistory, overallFiringRate(tick)));
    setRatingHistory((previousHistory) => pushCapped(previousHistory, tick.rating0To10));
    setArousalHistory((previousHistory) => pushCapped(previousHistory, tick.regionActivity.arousal ?? 0));
  }, [tick]);

  const currentFiringRate = firingRateHistory[firingRateHistory.length - 1] ?? 0;
  const arousal = tick.regionActivity.arousal ?? 0;
  const trackedRegionCount = Object.keys(tick.neuropilActivity).length;

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

      <Sparkline label="Dopamin-Verlauf" history={ratingHistory} maxValue={10} color="#ffd166" formatDomain={(value) => value.toFixed(0)} />
      <Sparkline label="Erregung-Verlauf" history={arousalHistory} maxValue={1} color="#ef476f" formatDomain={formatPercent} />
      <Sparkline label="Feuerrate-Verlauf" history={firingRateHistory} color="#7fd4ff" formatDomain={formatPercent} />
    </div>
  );
}
