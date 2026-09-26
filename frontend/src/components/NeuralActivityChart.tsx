import { useEffect, useState } from "react";
import type { TickData } from "../types";
import { AROUSAL_METRIC, DOPAMINE_METRIC, FIRING_RATE_METRIC, type MetricDefinition } from "../metrics";
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

/** Footer of the brain stage: the fly's current dopamine rating, arousal and overall
 * firing rate as three tiles, each with its own rolling sparkline over the last
 * HISTORY_LENGTH ticks (~4s at the default tick rate). Dopamine and arousal plot against
 * their real fixed 0-10 / 0-100% domain (they're already calibrated to it, see
 * emotion_decoder.py), while firing rate auto-scales to its own observed peak since it
 * isn't ceiling-normalized. While isPaused, the backend keeps resending the same frozen
 * tick - history stops accepting new points so the sparklines hold their last real trend
 * instead of flattening out into a repeated-value line. */
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
  const trackedRegionCount = Object.keys(tick.neuropilActivity).length;

  return (
    <div className="neural-activity">
      <MetricTile metric={DOPAMINE_METRIC} value={tick.rating0To10} history={ratingHistory} />
      <MetricTile metric={AROUSAL_METRIC} value={tick.regionActivity.arousal ?? 0} history={arousalHistory} />
      <MetricTile
        metric={FIRING_RATE_METRIC}
        value={currentFiringRate}
        history={firingRateHistory}
        detail={trackedRegionCount > 0 ? `${trackedRegionCount} Regionen` : undefined}
      />
    </div>
  );
}

/** One live metric: its color key and label, the current value in text ink, and its
 * rolling sparkline. The metric's description (plus any detail) is a hover tooltip. */
function MetricTile({
  metric,
  value,
  history,
  detail,
}: {
  metric: MetricDefinition;
  value: number;
  history: number[];
  detail?: string;
}) {
  return (
    <div className="metric-tile" title={detail ? `${metric.description} (${detail})` : metric.description}>
      <div className="metric-tile-header">
        <span className="swatch" style={{ background: metric.color }} />
        <span className="overline">{metric.label}</span>
      </div>
      <div className="metric-tile-value">
        {metric.formatValue(value)}
        {metric.unit && <span className="metric-tile-unit">{metric.unit}</span>}
      </div>
      <Sparkline
        history={history}
        maxValue={metric.maxValue}
        color={metric.color}
        formatValue={metric.formatValue}
        height={22}
      />
    </div>
  );
}
