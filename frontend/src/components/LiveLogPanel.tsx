import { useEffect, useRef, useState } from "react";
import type { TickData } from "../types";
import { formatDecimal, formatPercent } from "../utils/format";
import "./LiveLogPanel.css";

const MAX_LOG_LINES = 200;
// How close to the bottom (px) still counts as "following along", so new lines auto-scroll.
const FOLLOW_THRESHOLD_PX = 24;

/** One logged reaction to a single word, kept small enough to render cheaply in a list. */
type LogLine = {
  word: string;
  rating0To10: number;
  regionActivity: Record<string, number>;
};

/** HUD card with a live-scrolling table of the fly's per-word reaction: appends one row
 * whenever the current word changes (not every simulation tick), so the user can follow
 * along and see the rating/region activity actually move as new words are read. Only
 * auto-scrolls while the user is at the bottom, so scrolling up to read stays put. */
export function LiveLogPanel({ tick }: { tick: TickData }) {
  const [logLines, setLogLines] = useState<LogLine[]>([]);
  const lastLoggedWordsReadRef = useRef<number | null>(null);
  const scrollContainerRef = useRef<HTMLDivElement>(null);
  const isFollowingRef = useRef(true);

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
    if (container && isFollowingRef.current) container.scrollTop = container.scrollHeight;
  }, [logLines]);

  /** Remembers whether the user is at the bottom (following) or has scrolled up to read. */
  function handleScroll() {
    const container = scrollContainerRef.current;
    if (!container) return;
    isFollowingRef.current = container.scrollHeight - container.scrollTop - container.clientHeight < FOLLOW_THRESHOLD_PX;
  }

  return (
    <section className="card live-log" aria-labelledby="live-log-title">
      <div className="card-header">
        <h2 id="live-log-title" className="overline">
          Live-Log
        </h2>
        <span className="card-meta">Reaktion pro Wort</span>
      </div>
      <div className="live-log-scroll" ref={scrollContainerRef} onScroll={handleScroll}>
        {logLines.length === 0 ? (
          <p className="live-log-empty">Wartet auf Wörter…</p>
        ) : (
          <table className="live-log-table">
            <thead>
              <tr>
                <th scope="col">Wort</th>
                <th scope="col" title="Dopamin-Bewertung (0–10)">
                  Dopamin
                </th>
                <th scope="col" title="Annäherung (Approach-Pool)">
                  Annäh.
                </th>
                <th scope="col" title="Vermeidung (Avoidance-Pool)">
                  Vermeid.
                </th>
                <th scope="col" title="Erregung (Arousal-Pool)">
                  Erreg.
                </th>
              </tr>
            </thead>
            <tbody>
              {logLines.map((line, index) => (
                <tr key={index}>
                  <td className="live-log-word">{line.word}</td>
                  <td>{formatDecimal(line.rating0To10, 2)}</td>
                  <td>{formatPercent(line.regionActivity.approach ?? 0)}</td>
                  <td>{formatPercent(line.regionActivity.avoidance ?? 0)}</td>
                  <td>{formatPercent(line.regionActivity.arousal ?? 0)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </section>
  );
}
