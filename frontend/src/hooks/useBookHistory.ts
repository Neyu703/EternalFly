import { useEffect, useRef, useState } from "react";
import type { TickData } from "../types";

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

/** Mean spiking rate across the fly's tracked brain regions this tick (0..1), preferring
 * the fine-grained per-neuropil breakdown and falling back to the coarser approach/
 * avoidance/arousal pools when no neuropil data is available. */
function overallFiringRate(tick: TickData): number {
  const neuropilRates = Object.values(tick.neuropilActivity);
  const rates = neuropilRates.length > 0 ? neuropilRates : Object.values(tick.regionActivity);
  if (rates.length === 0) return 0;
  return rates.reduce((sum, rate) => sum + rate, 0) / rates.length;
}

function emptyBuckets(): BookHistoryBuckets {
  return new Array(NUM_BUCKETS).fill(null);
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
  const [buckets, setBuckets] = useState<BookHistoryBuckets>(emptyBuckets);
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
      setBuckets(emptyBuckets());
      wasBookFinishedRef.current = false;
    }
    lastTotalWordsRef.current = tick.totalWords;
    lastProgressRef.current = tick.pageProgress;

    if (tick.wordsRead !== lastWordsReadRef.current) {
      lastWordsReadRef.current = tick.wordsRead;
      const bucketIndex = Math.min(NUM_BUCKETS - 1, Math.floor(tick.pageProgress * NUM_BUCKETS));
      setBuckets((previousBuckets) => {
        const nextBuckets = [...previousBuckets];
        nextBuckets[bucketIndex] = {
          rating0To10: tick.rating0To10,
          arousal: tick.regionActivity.arousal ?? 0,
          firingRate: overallFiringRate(tick),
          emotions: tick.emotions,
        };
        return nextBuckets;
      });
    }

    if (tick.bookFinished && !wasBookFinishedRef.current) {
      wasBookFinishedRef.current = true;
      setFinishedBookHistory(buckets);
    } else if (!tick.bookFinished) {
      wasBookFinishedRef.current = false;
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tick]);

  function dismissFinishedBookHistory() {
    setFinishedBookHistory(null);
  }

  return { finishedBookHistory, dismissFinishedBookHistory };
}
