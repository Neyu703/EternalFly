import { useEffect, useState } from "react";
import type { TickData } from "../types";
import { formatPercent } from "../utils/format";
import { Sparkline } from "./Sparkline";
import "./NeuralActivityChart.css";

const HISTORY_LENGTH = 80;

/** Mean spiking rate across the fly's tracked brain regions this tick (0..1), preferring
 * the fine-grained per-neuropil breakdown and falling back to the coarser approach/
 * avoidance/arousal pools when no neuropil data is available. */
function overallFiringRate(tick: TickData): number {
  const neuropilRates = Object.values(tick.neuropilActivity);
  const rates = neuropilRates.length > 0 ? neuropilRates : Object.values(tick.regionActivity);
  if (rates.length === 0) return 0;
  return rates.reduce((sum, rate) => sum + rate, 0) / rates.length;
}

/** Live-updating status card: the fly's current dopamine rating, overall firing rate and
 * arousal, each with its own rolling sparkline over the last HISTORY_LENGTH ticks (~4s
 * at the default tick rate). Dopamine and arousal plot against their real fixed 0-10 /
 * 0-100% domain (they're already calibrated to it, see emotion_decoder.py), while firing
 * rate auto-scales to its own observed peak since it isn't ceiling-normalized. While
 * isPaused, the backend keeps resending the same frozen tick - history stops accepting
 * new points so the sparklines hold their last real trend instead of flattening out into
 * a repeated-value line. */
export function NeuralActivityChart({ tick, isPaused }: { tick: TickData; isPaused: boolean }) {
  const [firingRateHistory, setFiringRateHistory] = useState<number[]>([]);
  const [ratingHistory, setRatingHistory] = useState<number[]>([]);
  const [arousalHistory, setArousalHistory] = useState<number[]>([]);

  useEffect(() => {
    if (isPaused) return;
    const pushCapped = (previousHistory: number[], value: number): number[] => {
      const nextHistory = [...previousHistory, value];
      return nextHistory.length > HISTORY_LENGTH ? nextHistory.slice(-HISTORY_LENGTH) : nextHistory;
    };
    setFiringRateHistory((previousHistory) => pushCapped(previousHistory, overallFiringRate(tick)));
    setRatingHistory((previousHistory) => pushCapped(previousHistory, tick.rating0To10));
    setArousalHistory((previousHistory) => pushCapped(previousHistory, tick.regionActivity.arousal ?? 0));
  }, [tick, isPaused]);

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
