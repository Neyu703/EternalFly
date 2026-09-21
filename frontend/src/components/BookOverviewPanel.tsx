import { EMOTION_NAMES } from "../types";
import type { BookHistoryBuckets, BookHistoryPoint } from "../hooks/useBookHistory";
import { formatPercent } from "../utils/format";
import { EMOTION_COLORS } from "../emotionColors";
import { Sparkline } from "./Sparkline";
import "./BookOverviewPanel.css";

/** Keeps only the buckets a word actually landed in, in book-progress order, so the
 * overview charts plot a continuous trend instead of gaps for untouched buckets. */
function compactHistory(history: BookHistoryBuckets): BookHistoryPoint[] {
  return history.filter((point): point is BookHistoryPoint => point !== null);
}

/** End-of-book summary: how the fly's dopamine level, arousal, overall firing rate and
 * all 8 Plutchik emotions developed across the whole book, shown once when it finishes
 * (see useBookHistory). Progress-bucketed rather than time-bucketed, so it reads the
 * same shape regardless of how long or short the book was. */
export function BookOverviewPanel({
  history,
  onDismiss,
}: {
  history: BookHistoryBuckets;
  onDismiss: () => void;
}) {
  const points = compactHistory(history);
  const ratingHistory = points.map((point) => point.rating0To10);
  const arousalHistory = points.map((point) => point.arousal);
  const firingRateHistory = points.map((point) => point.firingRate);

  return (
    <div className="book-overview-backdrop" onClick={onDismiss}>
      <div className="book-overview-panel" onClick={(event) => event.stopPropagation()}>
        <div className="book-overview-header">
          <span className="book-overview-title">Buch beendet · Verlauf</span>
          <button className="book-overview-close" onClick={onDismiss}>
            ✕
          </button>
        </div>

        <div className="book-overview-main-charts">
          <Sparkline label="Dopamin-Verlauf" history={ratingHistory} maxValue={10} color="#ffd166" formatDomain={(value) => value.toFixed(0)} />
          <Sparkline label="Erregung-Verlauf" history={arousalHistory} maxValue={1} color="#ef476f" formatDomain={formatPercent} />
          <Sparkline label="Feuerrate-Verlauf" history={firingRateHistory} color="#7fd4ff" formatDomain={formatPercent} />
        </div>

        <div className="book-overview-emotion-grid">
          {EMOTION_NAMES.map((emotionName) => (
            <Sparkline
              key={emotionName}
              label={emotionName}
              history={points.map((point) => point.emotions[emotionName] ?? 0)}
              maxValue={1}
              color={EMOTION_COLORS[emotionName]}
              formatDomain={formatPercent}
            />
          ))}
        </div>
      </div>
    </div>
  );
}
