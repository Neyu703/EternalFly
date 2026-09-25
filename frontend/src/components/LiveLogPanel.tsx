import { useEffect, useRef, useState } from "react";
import type { TickData } from "../types";
import { normalizeRegionActivity } from "../regionActivity";
import "./LiveLogPanel.css";

const MAX_LOG_LINES = 200;

// A behavior readout above this counts as "the fly did this" for the log line's event
// label - same threshold Fly.tsx uses to trigger its own escape-jump animation.
const BEHAVIOR_EVENT_THRESHOLD = 0.5;
const BEHAVIOR_EVENT_LABELS: Record<string, string> = {
  escape: "⚡ Giant Fiber → Fluchtsprung",
  feeding: "🪰 Rüssel raus",
  backing: "↩️ MDN → weicht zurück",
  turn_left: "↰ dreht links",
  turn_right: "↱ dreht rechts",
};

/** One logged reaction to a single word, kept small enough to render cheaply in a list. */
type LogLine = {
  word: string;
  rating0To10: number;
  regionActivity: Record<string, number>;
  behaviorEvents: string[];
};

function formatPercent(fraction: number): string {
  return `${Math.round(fraction * 100)}%`;
}

/** Which real descending/motor readouts crossed BEHAVIOR_EVENT_THRESHOLD this frame,
 * as their display labels - so the log reads "the fly actually reacted here", not just
 * a running percentage. */
function activeBehaviorEvents(behaviors: Record<string, number>): string[] {
  return Object.entries(behaviors)
    .filter(([, value]) => value > BEHAVIOR_EVENT_THRESHOLD)
    .map(([name]) => BEHAVIOR_EVENT_LABELS[name] ?? name);
}

/** Live-scrolling log of the fly's per-word reaction: appends one line whenever the
 * current word changes (not every simulation tick), so the user can follow along and
 * see the rating/region activity actually move as new words are read. */
export function LiveLogPanel({ tick }: { tick: TickData }) {
  const [logLines, setLogLines] = useState<LogLine[]>([]);
  const lastLoggedWordsReadRef = useRef<number | null>(null);
  const scrollContainerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!tick.currentWord || tick.wordsRead === lastLoggedWordsReadRef.current) return;
    lastLoggedWordsReadRef.current = tick.wordsRead;
    setLogLines((previousLines) => {
      const nextLines = [
        ...previousLines,
        {
          word: tick.currentWord!,
          rating0To10: tick.rating0To10,
          regionActivity: tick.regionActivity,
          behaviorEvents: activeBehaviorEvents(tick.behaviors),
        },
      ];
      return nextLines.length > MAX_LOG_LINES ? nextLines.slice(-MAX_LOG_LINES) : nextLines;
    });
  }, [tick]);

  useEffect(() => {
    const container = scrollContainerRef.current;
    if (container) container.scrollTop = container.scrollHeight;
  }, [logLines]);

  return (
    <div className="live-log-panel">
      <span className="live-log-title">Live-Log</span>
      <div className="live-log-scroll" ref={scrollContainerRef}>
        {logLines.length === 0 && <span className="live-log-empty">Wartet auf Wörter…</span>}
        {logLines.map((line, index) => (
          <div key={index} className="live-log-line">
            <span className="live-log-word">{line.word}</span>
            <span className="live-log-rating">{line.rating0To10.toFixed(2)}/10</span>
            <span className="live-log-activity">
              R {formatPercent(normalizeRegionActivity(line.regionActivity.reward ?? 0, "reward"))} · P{" "}
              {formatPercent(normalizeRegionActivity(line.regionActivity.punishment ?? 0, "punishment"))} · E{" "}
              {formatPercent(normalizeRegionActivity(line.regionActivity.arousal ?? 0, "arousal"))}
            </span>
            {line.behaviorEvents.length > 0 && (
              <span className="live-log-behaviors">{line.behaviorEvents.join(" · ")}</span>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
