import { useState } from "react";
import type { TickData } from "../types";
import { AROUSAL_METRIC, DOPAMINE_METRIC, FIRING_RATE_METRIC, type MetricDefinition } from "../metrics";
import { appendCapped } from "../utils/array";
import { arousalOf } from "../utils/emotions";
import { Sparkline } from "./Sparkline";
import { Swatch } from "./Swatch";
import "./NeuralActivityChart.css";

const HISTORY_LENGTH = 80;
const TILE_SPARKLINE_HEIGHT_PX = 22;

/** The live metrics shown as tiles, in order, each with how to read its value off a tick. */
const LIVE_METRICS: { metric: MetricDefinition; valueOf: (tick: TickData) => number }[] = [
  { metric: DOPAMINE_METRIC, valueOf: (tick) => tick.rating0To10 },
  { metric: AROUSAL_METRIC, valueOf: (tick) => arousalOf(tick.regionActivity) },
  { metric: FIRING_RATE_METRIC, valueOf: (tick) => tick.firingRateHz },
];

/** Footer of the brain stage: the fly's current dopamine rating, arousal and overall
 * firing rate as three tiles, each with its own rolling sparkline over the last
 * HISTORY_LENGTH ticks (~4s at the default tick rate). Dopamine and arousal plot against
 * their real fixed 0-10 / 0-100% domain (they're already calibrated to it, see
 * emotion_decoder.py), while firing rate auto-scales to its own observed peak since it
 * isn't ceiling-normalized. While isPaused, the backend keeps resending the same frozen
 * tick - history stops accepting new points so the sparklines hold their last real trend
 * instead of flattening out into a repeated-value line. */
export function NeuralActivityChart({ tick, isPaused }: { tick: TickData; isPaused: boolean }) {
  const [histories, setHistories] = useState<number[][]>(() => LIVE_METRICS.map(() => []));
  const [recordedTick, setRecordedTick] = useState<TickData | null>(null);

  // Records each new tick while rendering (React's "adjust state when a prop changes"
  // pattern) instead of in an effect, so the tiles never render a stale history first.
  if (!isPaused && tick !== recordedTick) {
    setRecordedTick(tick);
    setHistories((previousHistories) =>
      LIVE_METRICS.map(({ valueOf }, index) => appendCapped(previousHistories[index], valueOf(tick), HISTORY_LENGTH)),
    );
  }

  return (
    <div className="neural-activity">
      {LIVE_METRICS.map(({ metric, valueOf }, index) => (
        <MetricTile key={metric.label} metric={metric} value={valueOf(tick)} history={histories[index]} />
      ))}
    </div>
  );
}

/** One live metric: its color key and label, the current value in text ink, and its
 * rolling sparkline. The metric's description is a hover tooltip. */
function MetricTile({ metric, value, history }: { metric: MetricDefinition; value: number; history: number[] }) {
  return (
    <div className="metric-tile" title={metric.description}>
      <div className="metric-tile-header">
        <Swatch color={metric.color} />
        <span className="overline">{metric.label}</span>
      </div>
      <div className="metric-tile-value">
        {metric.formatValue(value)}
        {metric.unit && <span className="unit-suffix">{metric.unit}</span>}
      </div>
      <Sparkline
        history={history}
        maxValue={metric.maxValue}
        color={metric.color}
        formatValue={metric.formatValue}
        height={TILE_SPARKLINE_HEIGHT_PX}
      />
    </div>
  );
}
