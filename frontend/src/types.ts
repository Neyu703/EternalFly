/** Mirrors the backend's TickResult JSON schema (see eternalfly/server.py). */
export type TickData = {
  currentWord: string | null;
  pageProgress: number;
  wordsRead: number;
  totalWords: number;
  emotions: Record<string, number>;
  rating0To10: number;
  regionActivity: Record<string, number>;
  wantsNewBook: boolean;
  /** Per-neuropil live firing rate (0..1), keyed the same as neuropil-centroids.json. */
  neuropilActivity: Record<string, number>;
  bookFinished: boolean;
};

export const EMOTION_NAMES = [
  "joy",
  "trust",
  "fear",
  "surprise",
  "sadness",
  "disgust",
  "anger",
  "anticipation",
] as const;

export type EmotionName = (typeof EMOTION_NAMES)[number];

/** German display label per Plutchik emotion; the backend's keys stay English. */
export const EMOTION_LABELS: Record<EmotionName, string> = {
  joy: "Freude",
  trust: "Vertrauen",
  fear: "Angst",
  surprise: "Überraschung",
  sadness: "Traurigkeit",
  disgust: "Ekel",
  anger: "Wut",
  anticipation: "Erwartung",
};
