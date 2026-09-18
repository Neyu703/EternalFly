import { useEffect, useState } from "react";
import { EMOTION_NAMES, type TickData } from "../types";

const DEMO_WORDS = ["the", "dragon", "roared", "and", "the", "castle", "shook", "with", "fear"];

/**
 * Placeholder tick data that cycles on its own, standing in for the real WebSocket feed
 * (eternalfly/server.py) until Milestone 5 wires the live simulation in. Same shape as the
 * real feed, so swapping this hook out later doesn't require touching any consuming component.
 */
export function useMockTickData(): TickData {
  const [wordIndex, setWordIndex] = useState(0);

  useEffect(() => {
    const intervalId = setInterval(() => setWordIndex((index) => index + 1), 2000);
    return () => clearInterval(intervalId);
  }, []);

  const cyclePosition = wordIndex % DEMO_WORDS.length;
  const emotions = Object.fromEntries(
    EMOTION_NAMES.map((name, index) => [
      name,
      0.3 + 0.3 * Math.sin(wordIndex * 0.6 + index),
    ])
  );

  return {
    currentWord: DEMO_WORDS[cyclePosition],
    pageProgress: (cyclePosition + 1) / DEMO_WORDS.length,
    emotions,
    rating0To10: 5 + 3 * Math.sin(wordIndex * 0.4),
    regionActivity: { approach: 0.4, avoidance: 0.2, arousal: 0.5 },
    wantsNewBook: false,
  };
}
