import { useEffect, useState } from "react";
import type { TickData } from "../types";
import { formatPercent } from "../utils/format";
import { normalizeRegionActivity } from "../regionActivity";
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
 * at the default tick rate), plus the achieved reading speed and overall spike rate.
 * Dopamine plots against its real fixed 0-10 domain (already calibrated to it, see
 * emotion_decoder.py); arousal is rescaled against its own measured ceiling (see
 * regionActivity.ts - the raw region_activity rate barely moves off 0 otherwise), while
 * firing rate auto-scales to its own observed peak since it isn't ceiling-normalized.
 * While isPaused, the backend keeps resending the same frozen tick - history stops
 * accepting new points so the sparklines hold their last real trend instead of
 * flattening out into a repeated-value line. */
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
    setArousalHistory((previousHistory) => pushCapped(previousHistory, normalizeRegionActivity(tick.regionActivity.arousal ?? 0, "arousal")));
  }, [tick, isPaused]);

  const currentFiringRate = firingRateHistory[firingRateHistory.length - 1] ?? 0;
  const arousal = normalizeRegionActivity(tick.regionActivity.arousal ?? 0, "arousal");
  const trackedRegionCount = Object.keys(tick.neuropilActivity).length;
  // The raw injected teaching signal (never changes across a session) vs. the fly's
  // actual current opinion (real MBON approach-avoidance activity, shaped over time by
  // dopamine-gated KC->MBON plasticity - see plasticity.py) - both on the same -1..1
  // scale, so a session where the fly has genuinely learned something shows the two
  // visibly diverging.
  const instinctValence =
    normalizeRegionActivity(tick.regionActivity.reward ?? 0, "reward") -
    normalizeRegionActivity(tick.regionActivity.punishment ?? 0, "punishment");

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
        <div className="neural-activity-stat">
          <span className="neural-activity-stat-label">Lesetempo</span>
          <span className="neural-activity-stat-value neural-activity-stat-value--wpm">
            {Math.round(tick.achievedWordsPerMinute)}
          </span>
          <span className="neural-activity-stat-caption">erreichte WPM</span>
        </div>
        <div className="neural-activity-stat">
          <span className="neural-activity-stat-label">Spikes</span>
          <span className="neural-activity-stat-value neural-activity-stat-value--spikes">
            {tick.spikesPerSecond >= 1000 ? `${(tick.spikesPerSecond / 1000).toFixed(1)}k` : Math.round(tick.spikesPerSecond)}
          </span>
          <span className="neural-activity-stat-caption">Spikes/s, ganzes Hirn</span>
        </div>
        <div className="neural-activity-stat">
          <span className="neural-activity-stat-label">Instinkt</span>
          <span className="neural-activity-stat-value neural-activity-stat-value--instinct">
            {instinctValence >= 0 ? "+" : ""}
            {instinctValence.toFixed(2)}
          </span>
          <span className="neural-activity-stat-caption">Belohnung/Strafe-Signal</span>
        </div>
        <div className="neural-activity-stat">
          <span className="neural-activity-stat-label">Gelernt</span>
          <span className="neural-activity-stat-value neural-activity-stat-value--learned">
            {tick.learnedValence >= 0 ? "+" : ""}
            {tick.learnedValence.toFixed(2)}
          </span>
          <span className="neural-activity-stat-caption">MBON-Bewertung, plastisch</span>
        </div>
      </div>

      <Sparkline label="Dopamin-Verlauf" history={ratingHistory} maxValue={10} color="#ffd166" formatDomain={(value) => value.toFixed(0)} />
      <Sparkline label="Erregung-Verlauf" history={arousalHistory} maxValue={1} color="#ef476f" formatDomain={formatPercent} />
      <Sparkline label="Feuerrate-Verlauf" history={firingRateHistory} color="#7fd4ff" formatDomain={formatPercent} />
    </div>
  );
}
