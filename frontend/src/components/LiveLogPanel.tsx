import { useEffect, useRef, useState } from "react";
import type { TickData } from "../types";
import "./LiveLogPanel.css";

const MAX_LOG_LINES = 200;

/** One logged reaction to a single word, kept small enough to render cheaply in a list. */
type LogLine = {
  word: string;
  rating0To10: number;
  regionActivity: Record<string, number>;
};

function formatPercent(fraction: number): string {
  return `${Math.round(fraction * 100)}%`;
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
      const nextLines = [...previousLines, { word: tick.currentWord!, rating0To10: tick.rating0To10, regionActivity: tick.regionActivity }];
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
              A {formatPercent(line.regionActivity.approach ?? 0)} · V {formatPercent(line.regionActivity.avoidance ?? 0)} · E{" "}
              {formatPercent(line.regionActivity.arousal ?? 0)}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}
