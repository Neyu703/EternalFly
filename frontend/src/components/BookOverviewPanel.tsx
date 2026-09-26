import { useEffect, useRef } from "react";
import { EMOTION_LABELS, EMOTION_NAMES } from "../types";
import type { BookHistoryBuckets, BookHistoryPoint } from "../hooks/useBookHistory";
import { AROUSAL_METRIC, DOPAMINE_METRIC, FIRING_RATE_METRIC, type MetricDefinition } from "../metrics";
import { EMOTION_COLORS } from "../emotionColors";
import { strongestEmotion } from "../utils/emotions";
import { formatPercent } from "../utils/format";
import { Sparkline } from "./Sparkline";
import { CloseIcon } from "./icons";
import "./BookOverviewPanel.css";

/** Keeps only the buckets a word actually landed in, in book-progress order, so the
 * overview charts plot a continuous trend instead of gaps for untouched buckets. */
function compactHistory(history: BookHistoryBuckets): BookHistoryPoint[] {
  return history.filter((point): point is BookHistoryPoint => point !== null);
}

/** Arithmetic mean of values, 0 for an empty list. */
function mean(values: number[]): number {
  return values.length === 0 ? 0 : values.reduce((sum, value) => sum + value, 0) / values.length;
}

/** End-of-book summary dialog: headline numbers first (average dopamine, peak arousal,
 * strongest emotion), then how dopamine, arousal, overall firing rate and all 8 Plutchik
 * emotions developed across the whole book, shown once when it finishes (see
 * useBookHistory). Progress-bucketed rather than time-bucketed, so it reads the same shape
 * regardless of how long or short the book was. A native modal <dialog>: Escape, the
 * close button, the "Weiter" button and a backdrop click all dismiss it. */
export function BookOverviewPanel({
  history,
  onDismiss,
}: {
  history: BookHistoryBuckets;
  onDismiss: () => void;
}) {
  const dialogRef = useRef<HTMLDialogElement>(null);
  const continueButtonRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    const dialog = dialogRef.current;
    // Guarded because StrictMode re-runs this effect on an already-open dialog.
    if (dialog && !dialog.open) dialog.showModal();
    // preventScroll keeps a scrollable dialog at its headline instead of jumping to the footer button.
    continueButtonRef.current?.focus({ preventScroll: true });
  }, []);

  const points = compactHistory(history);
  const ratingHistory = points.map((point) => point.rating0To10);
  const arousalHistory = points.map((point) => point.arousal);
  const metricHistories: [MetricDefinition, number[]][] = [
    [DOPAMINE_METRIC, ratingHistory],
    [AROUSAL_METRIC, arousalHistory],
    [FIRING_RATE_METRIC, points.map((point) => point.firingRate)],
  ];
  const meanEmotionByName = Object.fromEntries(
    EMOTION_NAMES.map((emotionName) => [emotionName, mean(points.map((point) => point.emotions[emotionName] ?? 0))]),
  );
  const dominantEmotion = strongestEmotion(meanEmotionByName);

  return (
    <dialog
      ref={dialogRef}
      className="modal"
      aria-labelledby="book-overview-title"
      onClose={onDismiss}
      onClick={(event) => {
        // The content wrapper fills the dialog, so only a click on the backdrop targets the dialog itself.
        if (event.target === event.currentTarget) onDismiss();
      }}
    >
      <div className="modal-content">
        <header className="modal-header">
          <div className="modal-heading">
            <span className="overline">Buch beendet</span>
            <h2 id="book-overview-title" className="modal-title">
              So hat die Fliege das Buch erlebt
            </h2>
          </div>
          <button className="icon-button" onClick={onDismiss} aria-label="Schließen">
            <CloseIcon />
          </button>
        </header>

        <div className="book-overview-summary">
          <SummaryStat
            label="Ø Dopamin"
            value={DOPAMINE_METRIC.formatValue(mean(ratingHistory))}
            unit={DOPAMINE_METRIC.unit}
            color={DOPAMINE_METRIC.color}
          />
          <SummaryStat
            label="Höchste Erregung"
            value={AROUSAL_METRIC.formatValue(Math.max(0, ...arousalHistory))}
            color={AROUSAL_METRIC.color}
          />
          <SummaryStat
            label="Stärkste Emotion"
            value={dominantEmotion ? EMOTION_LABELS[dominantEmotion] : "—"}
            color={dominantEmotion ? EMOTION_COLORS[dominantEmotion] : undefined}
          />
        </div>

        <section className="book-overview-section" aria-labelledby="book-overview-metrics-title">
          <h3 id="book-overview-metrics-title" className="overline">
            Verlauf über das Buch
          </h3>
          {metricHistories.map(([metric, metricHistory]) => (
            <Sparkline
              key={metric.label}
              label={metric.label}
              history={metricHistory}
              maxValue={metric.maxValue}
              color={metric.color}
              formatValue={metric.formatValue}
              height={40}
            />
          ))}
          <div className="book-overview-axis" aria-hidden="true">
            <span>Anfang</span>
            <span>Ende</span>
          </div>
        </section>

        <section className="book-overview-section" aria-labelledby="book-overview-emotions-title">
          <h3 id="book-overview-emotions-title" className="overline">
            Emotionen · jeweils 0–100 %
          </h3>
          <div className="book-overview-emotions">
            {EMOTION_NAMES.map((emotionName) => (
              <Sparkline
                key={emotionName}
                label={EMOTION_LABELS[emotionName]}
                history={points.map((point) => point.emotions[emotionName] ?? 0)}
                maxValue={1}
                color={EMOTION_COLORS[emotionName]}
                formatValue={formatPercent}
                showDomain={false}
              />
            ))}
          </div>
        </section>

        <footer className="modal-footer">
          <button ref={continueButtonRef} className="button button--primary" onClick={onDismiss}>
            Weiter
          </button>
        </footer>
      </div>
    </dialog>
  );
}

/** One headline number of the summary row, keyed by an optional color swatch. */
function SummaryStat({ label, value, unit, color }: { label: string; value: string; unit?: string; color?: string }) {
  return (
    <div className="summary-stat">
      <span className="summary-stat-label">
        {color && <span className="swatch" style={{ background: color }} />}
        {label}
      </span>
      <span className="summary-stat-value">
        {value}
        {unit && <span className="summary-stat-unit">{unit}</span>}
      </span>
    </div>
  );
}
