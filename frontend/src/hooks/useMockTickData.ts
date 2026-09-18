import { useEffect, useState } from "react";
import { EMOTION_NAMES, type TickData } from "../types";

const DEMO_WORDS = ["the", "dragon", "roared", "and", "the", "castle", "shook", "with", "fear"];

/** Deterministic 0..1 pseudo-value from a string, used to fake per-region activity per word. */
function hashToUnitInterval(text: string): number {
  let hash = 0;
  for (let index = 0; index < text.length; index += 1) {
    hash = (hash * 31 + text.charCodeAt(index)) % 100000;
  }
  return hash / 100000;
}

/**
 * Placeholder tick data that cycles on its own, standing in for the real WebSocket feed
 * (eternalfly/server.py) until Milestone 5 wires the live simulation in. Same shape as the
 * real feed, so swapping this hook out later doesn't require touching any consuming component.
 */
export function useMockTickData(): TickData {
  const [wordIndex, setWordIndex] = useState(0);
  const [neuropilNames, setNeuropilNames] = useState<string[]>([]);

  useEffect(() => {
    const intervalId = setInterval(() => setWordIndex((index) => index + 1), 350);
    return () => clearInterval(intervalId);
  }, []);

  useEffect(() => {
    fetch("/models/neuropil-centroids.json")
      .then((response) => response.json())
      .then((centroids: Record<string, unknown>) => setNeuropilNames(Object.keys(centroids)));
  }, []);

  const cyclePosition = wordIndex % DEMO_WORDS.length;
  const currentWord = DEMO_WORDS[cyclePosition];

  const emotions = Object.fromEntries(
    EMOTION_NAMES.map((name, index) => [name, 0.3 + 0.3 * Math.sin(wordIndex * 0.6 + index)])
  );

  // Each word deterministically "excites" a different subset of regions, standing in for real
  // per-region firing rates until the WebSocket feed replaces this hook.
  const neuropilActivity = Object.fromEntries(
    neuropilNames.map((regionName) => [regionName, hashToUnitInterval(currentWord + regionName)])
  );

  return {
    currentWord,
    pageProgress: (cyclePosition + 1) / DEMO_WORDS.length,
    emotions,
    rating0To10: 5 + 3 * Math.sin(wordIndex * 0.4),
    regionActivity: { approach: 0.4, avoidance: 0.2, arousal: 0.5 },
    wantsNewBook: false,
    neuropilActivity,
  };
}
