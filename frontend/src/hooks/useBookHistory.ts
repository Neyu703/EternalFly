import { useCallback, useEffect, useRef, useState } from "react";
import type { TickData } from "../types";
import { arousalOf } from "../utils/emotions";

const NUM_BUCKETS = 200;
const RESTART_PROGRESS_DROP_THRESHOLD = 0.01;

/** The fly's state at one point (0..1 pageProgress) across a book, used to build the
 * end-of-book overview chart. */
export type BookHistoryPoint = {
  rating0To10: number;
  arousal: number;
  firingRate: number;
  emotions: Record<string, number>;
};

/** Book-progress-indexed history: index i covers page progress [i/NUM_BUCKETS, (i+1)/NUM_BUCKETS).
 * A bucket stays null until a word lands in it. */
export type BookHistoryBuckets = (BookHistoryPoint | null)[];

function emptyBuckets(): BookHistoryBuckets {
  return new Array(NUM_BUCKETS).fill(null);
}

/** The bucket a 0..1 page progress falls into (the last one for a finished book). */
function bucketIndexOf(pageProgress: number): number {
  return Math.min(NUM_BUCKETS - 1, Math.floor(pageProgress * NUM_BUCKETS));
}

/** The part of a tick the end-of-book overview charts. */
function toHistoryPoint(tick: TickData): BookHistoryPoint {
  return {
    rating0To10: tick.rating0To10,
    arousal: arousalOf(tick.regionActivity),
    firingRate: tick.firingRateHz,
    emotions: tick.emotions,
  };
}

/**
 * Tracks the fly's rating/arousal/firing-rate/emotions across an entire book's progress
 * (0..100%), bucketed into a fixed NUM_BUCKETS points regardless of how long the book is,
 * so a 50-word text and a 1,000,000-word one both produce a chartable-sized history. Once
 * the book finishes, the accumulated buckets are frozen into finishedBookHistory (until
 * dismissFinishedBookHistory() clears it) so an end-of-book overview can be shown even
 * though autoplay may already be loading the next book by the following tick. Resets
 * automatically when a new book starts: a different totalWords, or pageProgress moving
 * backwards (a manual or autoplay restart of the same book).
 */
export function useBookHistory(tick: TickData | null): {
  finishedBookHistory: BookHistoryBuckets | null;
  dismissFinishedBookHistory: () => void;
} {
  // A ref, not state: the buckets are never rendered, only copied into finishedBookHistory,
  // and a ref also holds the latest tick's point when that copy is taken.
  const bucketsRef = useRef<BookHistoryBuckets>(emptyBuckets());
  const [finishedBookHistory, setFinishedBookHistory] = useState<BookHistoryBuckets | null>(null);
  const lastWordsReadRef = useRef<number | null>(null);
  const lastProgressRef = useRef(0);
  const lastTotalWordsRef = useRef<number | null>(null);
  const wasBookFinishedRef = useRef(false);

  useEffect(() => {
    if (!tick) return;

    const isNewBook = lastTotalWordsRef.current !== null && lastTotalWordsRef.current !== tick.totalWords;
    const isRestart = tick.pageProgress < lastProgressRef.current - RESTART_PROGRESS_DROP_THRESHOLD;
    if (isNewBook || isRestart) {
      bucketsRef.current = emptyBuckets();
      wasBookFinishedRef.current = false;
    }
    lastTotalWordsRef.current = tick.totalWords;
    lastProgressRef.current = tick.pageProgress;

    if (tick.wordsRead !== lastWordsReadRef.current) {
      lastWordsReadRef.current = tick.wordsRead;
      bucketsRef.current[bucketIndexOf(tick.pageProgress)] = toHistoryPoint(tick);
    }

    if (tick.bookFinished && !wasBookFinishedRef.current) {
      wasBookFinishedRef.current = true;
      setFinishedBookHistory([...bucketsRef.current]);
    } else if (!tick.bookFinished) {
      wasBookFinishedRef.current = false;
    }
  }, [tick]);

  // Stable identity, so the overview dialog's Escape-key listener isn't re-registered on every tick.
  const dismissFinishedBookHistory = useCallback(() => setFinishedBookHistory(null), []);

  return { finishedBookHistory, dismissFinishedBookHistory };
}
