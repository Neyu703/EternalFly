import { useEffect, useRef } from "react";
import { EMOTION_NAMES, type EmotionName } from "../types";
import type { BookHistoryBuckets, BookHistoryPoint } from "../hooks/useBookHistory";
import { AROUSAL_METRIC, DOPAMINE_METRIC, FIRING_RATE_METRIC, type MetricDefinition } from "../metrics";
import { EMOTION_DEFINITIONS } from "../emotionDefinitions";
import { strongestEmotion } from "../utils/emotions";
import { formatPercent } from "../utils/format";
import { Sparkline } from "./Sparkline";
import { CloseIcon } from "./icons";
import "./BookOverviewPanel.css";

// The overview's metric charts are taller than the live tiles', for the longer history.
const OVERVIEW_SPARKLINE_HEIGHT_PX = 40;

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
 * strongest emotion), then how dopamine, arousal, overall firing rate and the fly's
 * emotional states (reward, aversion, arousal) developed across the whole book, shown once when it finishes (see
 * useBookHistory). Progress-bucketed rather than time-bucketed, so it reads the same shape
 * regardless of how long or short the book was. A native modal <dialog>: Escape, the
 * close button, the "Continue" button and a backdrop click all dismiss it. */
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
  const emotionHistories = Object.fromEntries(
    EMOTION_NAMES.map((emotionName) => [emotionName, points.map((point) => point.emotions[emotionName] ?? 0)]),
  ) as Record<EmotionName, number[]>;
  const meanEmotionByName = Object.fromEntries(
    EMOTION_NAMES.map((emotionName) => [emotionName, mean(emotionHistories[emotionName])]),
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
            <span className="overline">Book finished</span>
            <h2 id="book-overview-title" className="modal-title">
              How the fly experienced the book
            </h2>
          </div>
          <button className="icon-button" onClick={onDismiss} aria-label="Close">
            <CloseIcon />
          </button>
        </header>

        <div className="book-overview-summary">
          <SummaryStat
            label="Avg. dopamine"
            value={DOPAMINE_METRIC.formatValue(mean(ratingHistory))}
            unit={DOPAMINE_METRIC.unit}
            color={DOPAMINE_METRIC.color}
          />
          <SummaryStat
            label="Peak arousal"
            value={AROUSAL_METRIC.formatValue(Math.max(0, ...arousalHistory))}
            color={AROUSAL_METRIC.color}
          />
          <SummaryStat
            label="Strongest emotion"
            value={dominantEmotion ? EMOTION_DEFINITIONS[dominantEmotion].label : "—"}
            color={dominantEmotion ? EMOTION_DEFINITIONS[dominantEmotion].color : undefined}
          />
        </div>

        <section className="book-overview-section" aria-labelledby="book-overview-metrics-title">
          <h3 id="book-overview-metrics-title" className="overline">
            Across the book
          </h3>
          {metricHistories.map(([metric, metricHistory]) => (
            <Sparkline
              key={metric.label}
              label={metric.label}
              history={metricHistory}
              maxValue={metric.maxValue}
              color={metric.color}
              formatValue={metric.formatValue}
              height={OVERVIEW_SPARKLINE_HEIGHT_PX}
            />
          ))}
          <div className="book-overview-axis" aria-hidden="true">
            <span>Start</span>
            <span>End</span>
          </div>
        </section>

        <section className="book-overview-section" aria-labelledby="book-overview-emotions-title">
          <h3 id="book-overview-emotions-title" className="overline">
            Emotions · each 0–100%
          </h3>
          <div className="book-overview-emotions">
            {EMOTION_NAMES.map((emotionName) => (
              <Sparkline
                key={emotionName}
                label={EMOTION_DEFINITIONS[emotionName].label}
                history={emotionHistories[emotionName]}
                maxValue={1}
                color={EMOTION_DEFINITIONS[emotionName].color}
                formatValue={formatPercent}
                showDomain={false}
              />
            ))}
          </div>
        </section>

        <footer className="modal-footer">
          <button ref={continueButtonRef} className="button button--primary" onClick={onDismiss}>
            Continue
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
      <span className="summary-stat-value truncate">
        {value}
        {unit && <span className="unit-suffix">{unit}</span>}
      </span>
    </div>
  );
}
